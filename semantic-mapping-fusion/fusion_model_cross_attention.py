import os, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset

from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from transformers import Dinov2Model
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------
# Dispositivo
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

AMP_DTYPE = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else torch.float16
)

# -------------------------------------------------------
# Hiperparâmetros de perdas auxiliares
# -------------------------------------------------------
TAU = 0.1            # temperatura para InfoNCE
LAMBDA_CL = 0.10     # peso da perda contrastiva
LAMBDA_ORTHO = 1e-4  # peso da perda de ortogonalidade

# -------------------------------------------------------
# Transforms pareados (mantendo Resize(224,224))
# -------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class PairedTrainTransform224:
    def __init__(self, size=224, do_vflip=True, max_rot=15):
        self.size = size
        self.do_vflip = do_vflip
        self.max_rot = max_rot

    def __call__(self, rgb, dep):
        if random.random() < 0.5:
            rgb = F.hflip(rgb); dep = F.hflip(dep)
        if self.do_vflip and random.random() < 0.2:
            rgb = F.vflip(rgb); dep = F.vflip(dep)
        angle = random.uniform(-self.max_rot, self.max_rot)
        rgb = F.rotate(rgb, angle, InterpolationMode.BICUBIC, expand=False)
        dep = F.rotate(dep, angle, InterpolationMode.NEAREST, expand=False)

        rgb = F.resize(rgb, (self.size, self.size), InterpolationMode.BICUBIC)
        dep = F.resize(dep, (self.size, self.size), InterpolationMode.NEAREST)

        rgb = transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)(rgb)

        rgb = F.to_tensor(rgb)
        dep_t = F.to_tensor(dep)              # (1,H,W)
        dep  = dep_t.expand(3, *dep_t.shape[1:])

        rgb = F.normalize(rgb, IMAGENET_MEAN, IMAGENET_STD)
        dep = F.normalize(dep, IMAGENET_MEAN, IMAGENET_STD)
        return rgb, dep

class PairedEvalTransform224:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, rgb, dep):
        rgb = F.resize(rgb, (self.size, self.size), InterpolationMode.BICUBIC)
        dep = F.resize(dep, (self.size, self.size), InterpolationMode.NEAREST)

        rgb = F.to_tensor(rgb)
        dep_t = F.to_tensor(dep)              # (1,H,W)
        dep  = dep_t.expand(3, *dep_t.shape[1:])

        rgb = F.normalize(rgb, IMAGENET_MEAN, IMAGENET_STD)
        dep = F.normalize(dep, IMAGENET_MEAN, IMAGENET_STD)
        return rgb, dep

# -------------------------------------------------------
# Dataset RGBD com transform pareado
# -------------------------------------------------------
class RGBDDataset(Dataset):
    def __init__(self, rgb_path, depth_path, paired_transform=None,
                 rgb_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
                 depth_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
                 strict_classes=True):
        self.rgb_root = Path(rgb_path)
        self.depth_root = Path(depth_path)
        self.paired_transform = paired_transform

        tmp = datasets.ImageFolder(str(self.rgb_root))
        self.classes = tmp.classes
        self.class_to_idx = tmp.class_to_idx

        if strict_classes:
            depth_classes = sorted([p.name for p in self.depth_root.iterdir() if p.is_dir()])
            if depth_classes != self.classes:
                raise ValueError(f"As classes em depth diferem das de RGB.\nRGB:{self.classes}\nDEPTH:{depth_classes}")

        depth_index = {}
        for cls in self.classes:
            cdir = self.depth_root / cls
            if not cdir.exists():
                continue
            for p in cdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in depth_exts:
                    rel = p.relative_to(cdir).with_suffix("")
                    depth_index[(cls, str(rel))] = p

        self.pairs = []
        for rgb_path, label in tmp.samples:
            rgb_path = Path(rgb_path)
            cls = rgb_path.parent.name
            rel = rgb_path.relative_to(self.rgb_root / cls).with_suffix("")
            dpath = depth_index.get((cls, str(rel)))
            if dpath is not None and rgb_path.suffix.lower() in rgb_exts:
                self.pairs.append((rgb_path, dpath, label))

        if len(self.pairs) == 0:
            raise RuntimeError("Nenhum par RGB-Depth encontrado. Verifique nomes e diretórios.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_img_path, depth_img_path, label = self.pairs[idx]
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        depth_image = Image.open(depth_img_path).convert("L")

        if self.paired_transform is not None:
            rgb_image, depth_image = self.paired_transform(rgb_image, depth_image)
        else:
            rgb_image  = F.resize(rgb_image, (224,224), InterpolationMode.BICUBIC)
            depth_image = F.resize(depth_image, (224,224), InterpolationMode.NEAREST)

            rgb_image = F.to_tensor(rgb_image)
            depth_t   = F.to_tensor(depth_image)
            depth_image = depth_t.expand(3, *depth_t.shape[1:])

            rgb_image = F.normalize(rgb_image, IMAGENET_MEAN, IMAGENET_STD)
            depth_image = F.normalize(depth_image, IMAGENET_MEAN, IMAGENET_STD)

        return (rgb_image, depth_image), label

# -------------------------------------------------------
# Bloco de Atenção Cruzada com pré-norm + residual + MLP
# -------------------------------------------------------
class CrossAttnBlock(nn.Module):
    def __init__(self, dim=768, heads=8, p_attn=0.1, p_mlp=0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=p_attn, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(p_mlp),
            nn.Linear(dim*4, dim), nn.Dropout(p_mlp),
        )

    def forward(self, q_seq, kv_seq):
        q = self.ln_q(q_seq)
        kv = self.ln_kv(kv_seq)
        attn_out, _ = self.attn(q, kv, kv)
        x = q_seq + torch.sigmoid(self.gate) * attn_out
        x = x + self.mlp(self.ln_mlp(x))
        return x

# -------------------------------------------------------
# Funções de perdas auxiliares
# -------------------------------------------------------
def info_nce_cosine(h_a: torch.Tensor, h_b: torch.Tensor, tau: float = 0.1) -> float:
    """
    h_a, h_b: (B, D) NORMALIZADOS (norma L2=1)
    InfoNCE simétrica (A->B e B->A) com temperatura tau.
    """
    # Similaridades (B,B)
    logits = (h_a @ h_b.t()) / tau
    labels = torch.arange(h_a.size(0), device=h_a.device)
    ce = nn.CrossEntropyLoss()
    loss_ab = ce(logits, labels)
    loss_ba = ce(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)

def orthogonality_loss(Ha: torch.Tensor, Hb: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Ha, Hb: (B, D) (podem estar normalizados)
    Penaliza correlação linear entre lotes (forma Frobenius normalizada).
    """
    Ha_c = Ha - Ha.mean(dim=0, keepdim=True)
    Hb_c = Hb - Hb.mean(dim=0, keepdim=True)
    num = (Ha_c.t() @ Hb_c).pow(2).sum()
    den = (Ha_c.norm() * Hb_c.norm() + eps).pow(2)
    return num / den

# -------------------------------------------------------
# Modelo com Projeções por Modalidade + Atenção Cruzada + Cabeça MLP
# -------------------------------------------------------
class DinoV2_RGBD_CrossAttentionModel(nn.Module):
    def __init__(self, num_classes, dino_hidden_size=768, num_attention_heads=8):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")

        # Fine-tuning parcial
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.encoder.layer[-6:]:
            for p in blk.parameters():
                p.requires_grad = True

        # --- Projeções por Modalidade (Modality-Specific Projections) ---
        self.proj_rgb = nn.Sequential(
            nn.LayerNorm(dino_hidden_size),
            nn.Linear(dino_hidden_size, dino_hidden_size),
            nn.GELU(),
            nn.LayerNorm(dino_hidden_size),
        )
        self.proj_dep = nn.Sequential(
            nn.LayerNorm(dino_hidden_size),
            nn.Linear(dino_hidden_size, dino_hidden_size),
            nn.GELU(),
            nn.LayerNorm(dino_hidden_size),
        )

        # --- Atenção Cruzada Bidirecional ---
        self.rgb_attends_to_depth = CrossAttnBlock(
            dim=dino_hidden_size, heads=num_attention_heads, p_attn=0.1, p_mlp=0.1
        )
        self.depth_attends_to_rgb = CrossAttnBlock(
            dim=dino_hidden_size, heads=num_attention_heads, p_attn=0.1, p_mlp=0.1
        )

        # --- Classificador Final ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(dino_hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(dino_hidden_size * 2, dino_hidden_size),
            nn.GELU(),
            nn.LayerNorm(dino_hidden_size),
            nn.Linear(dino_hidden_size, num_classes)
        )

    def forward(self, rgb_image, depth_image):
        """
        Em modo train (self.training=True), retorna (logits, h_rgb, h_dep)
        Em modo eval, retorna apenas logits (compatibilidade com test/val)
        """
        # --- Extração backbone ---
        rgb_feat_map = self.backbone(rgb_image).last_hidden_state  # (B, L+1, D)
        dep_feat_map = self.backbone(depth_image).last_hidden_state

        # --- Projeções por Modalidade ---
        rgb_proj = self.proj_rgb(rgb_feat_map)  # (B, L+1, D)
        dep_proj = self.proj_dep(dep_feat_map)  # (B, L+1, D)

        # Embeddings [CLS] projetados e NORMALIZADOS (para perdas auxiliares)
        h_rgb = nn.functional.normalize(rgb_proj[:, 0, :], dim=-1)  # (B, D)
        h_dep = nn.functional.normalize(dep_proj[:, 0, :], dim=-1)  # (B, D)

        # --- Atenção Cruzada Bidirecional ---
        refined_rgb   = self.depth_attends_to_rgb(rgb_proj, dep_proj)
        refined_depth = self.rgb_attends_to_depth(dep_proj, rgb_proj)

        # --- Classificação (concat dos CLS refinidos) ---
        cls_rgb   = refined_rgb[:, 0, :]
        cls_depth = refined_depth[:, 0, :]
        fused = torch.cat([cls_rgb, cls_depth], dim=1)
        logits = self.classifier(fused)

        if self.training:
            return logits, h_rgb, h_dep
        else:
            return logits

# -------------------------------------------------------
# Treino + Val com CE + (λ_cl * CL) + (λ_ortho * ORTHO)
# -------------------------------------------------------
from torch.cuda.amp import autocast, GradScaler

def train_and_evaluate(model, train_loader, val_loader, num_epochs=20, lr=1e-4):
    import itertools

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr/10, 'weight_decay': 0.01},
        {'params': itertools.chain(
            model.proj_rgb.parameters(),
            model.proj_dep.parameters(),
            model.rgb_attends_to_depth.parameters(),
            model.depth_attends_to_rgb.parameters(),
            model.classifier.parameters()
        ), 'lr': lr, 'weight_decay': 0.05}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ------------------- TRAIN -------------------
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_cl = 0.0
        running_ortho = 0.0
        correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")
        for (images_rgb, images_depth), labels in pbar:
            images_rgb = images_rgb.to(device, non_blocking=True)
            images_depth = images_depth.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=AMP_DTYPE):
                logits, h_rgb, h_dep = model(images_rgb, images_depth)
                loss_ce = criterion(logits, labels)
                loss_cl = info_nce_cosine(h_rgb, h_dep, tau=TAU)
                loss_ortho = orthogonality_loss(h_rgb, h_dep)
                loss = loss_ce + LAMBDA_CL * loss_cl + LAMBDA_ORTHO * loss_ortho

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_cl += loss_cl.item()
            running_ortho += loss_ortho.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{running_loss/max(1,total):.4f}",
                "acc": f"{100.0*correct/max(1,total):.2f}%"
            })

        scheduler.step()

        train_loss = running_loss / max(1, len(train_loader))
        train_ce   = running_ce   / max(1, len(train_loader))
        train_cl   = running_cl   / max(1, len(train_loader))
        train_or   = running_ortho/ max(1, len(train_loader))
        train_acc  = 100.0 * correct / max(1, total)

        # ------------------- VAL -------------------
        model.eval()
        val_loss_ce = 0.0
        val_loss_cl = 0.0
        val_loss_or = 0.0
        val_correct = val_total = 0
        with torch.no_grad():
            pbarv = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]")
            for (images_rgb, images_depth), labels in pbarv:
                images_rgb = images_rgb.to(device, non_blocking=True)
                images_depth = images_depth.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(dtype=AMP_DTYPE):
                    # Em eval, forward retorna só logits; precisamos dos h_* para métricas? Podemos obtê-los rodando training=False, mas pedindo retorno estendido:
                    # Uma abordagem simples: mude temporariamente para train() sem grad para pegar h_* — porém manteremos avaliação com CE.
                    # Para não alterar o modo, aqui mantemos apenas CE para seleção de modelo.
                    logits = model(images_rgb, images_depth)
                    loss = criterion(logits, labels)

                val_loss_ce += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / max(1, val_total)
        val_loss_ce /= max(1, len(val_loader))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_dino_fusion1.pth')

        print(f"[Epoch {epoch+1}] "
              f"Train: Loss={train_loss:.4f} | CE={train_ce:.4f} | CL={train_cl:.4f} | ORT={train_or:.6f} | Acc={train_acc:.2f}%  ||  "
              f"Val: CE={val_loss_ce:.4f} | Acc={val_acc:.2f}%  (best {best_val_acc:.2f}%)")

    model.load_state_dict(torch.load('best_model_dino_fusion1.pth', map_location=device))
    return model

# -------------------------------------------------------
# Teste
# -------------------------------------------------------
def test_model(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for (images_rgb, images_depth), labels in tqdm(test_loader, desc="Testing"):
            images_rgb = images_rgb.to(device, non_blocking=True)
            images_depth = images_depth.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(dtype=AMP_DTYPE):
                logits = model(images_rgb, images_depth)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100.0 * correct / max(1,total):.2f}%\n")

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-4

    # Treinos
    TRAIN_PATH_RGB1 = 'COLD/Saarbrücken/cloudy/cloudy1'
    TRAIN_PATH_DEPTH1 = 'COLD_depth/Saarbrücken/cloudy/cloudy1'
    TRAIN_PATH_RGB2 = 'COLD/Saarbrücken/cloudy/cloudy2'
    TRAIN_PATH_DEPTH2 = 'COLD_depth/Saarbrücken/cloudy/cloudy2'

    # Teste
    TEST_PATH_RGB1 = 'COLD/Saarbrücken/cloudy/cloudy3'
    TEST_PATH_DEPTH1 = 'COLD_depth/Saarbrücken/cloudy/cloudy3'

    # Transforms pareados mantendo Resize 224
    train_tf = PairedTrainTransform224(size=IMG_SIZE, do_vflip=True, max_rot=15)
    eval_tf  = PairedEvalTransform224(size=IMG_SIZE)

    print("Carregando datasets de treino individuais...")
    train_dataset1 = RGBDDataset(TRAIN_PATH_RGB1, TRAIN_PATH_DEPTH1, paired_transform=train_tf)
    train_dataset2 = RGBDDataset(TRAIN_PATH_RGB2, TRAIN_PATH_DEPTH2, paired_transform=train_tf)

    full_train_dataset = ConcatDataset([train_dataset1, train_dataset2])

    print("Carregando datasets de teste...")
    test_dataset = RGBDDataset(TEST_PATH_RGB1, TEST_PATH_DEPTH1, paired_transform=eval_tf)

    num_classes = len(train_dataset1.classes)

    # Split train/val
    val_ratio = 0.2
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Resumo
    print('-' * 50)
    print("SETUP DO EXPERIMENTO")
    print(f"Treino combinado (RGB): {TRAIN_PATH_RGB1} + {TRAIN_PATH_RGB2}")
    print(f"Número de classes: {num_classes}")
    print(f"Total de imagens de treino (antes do split): {len(full_train_dataset)}")
    print(f"  -> Imagens de treino (após split): {len(train_dataset)}")
    print(f"  -> Imagens de validação: {len(val_dataset)}")
    print(f"Total de imagens de teste 1: {len(test_dataset)}")
    print('-' * 50)

    model = DinoV2_RGBD_CrossAttentionModel(num_classes).to(device)
    trained_model = train_and_evaluate(model, train_loader, val_loader,
                                       num_epochs=EPOCHS, lr=LEARNING_RATE)

    print('\n- - - Avaliando o modelo no conjunto de teste - - -')
    print(f"Teste 1: {TEST_PATH_RGB1}")
    test_model(trained_model, test_loader)

if __name__ == "__main__":
    main()

