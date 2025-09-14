import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from timm import create_model
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd

def load_data(data_path, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Carrega o dataset completo
    dataset = datasets.ImageFolder(data_path, transform=transform)
    num_classes = len(dataset.classes)

    return dataset, num_classes

def define_model(num_classes):
    #model = create_model('convnext_base', pretrained=True, num_classes=num_classes)
    #model = create_model('convformer_s18', pretrained=True, num_classes=num_classes)
    #model = create_model('beit_base_patch16_224', pretrained=True, num_classes=num_classes)
    #model = create_model('deit_base_patch16_224', pretrained=True, num_classes=num_classes)
    model = create_model('mixer_b16_224', pretrained=True, num_classes=num_classes)
    #model = create_model('swinv2_base_window12_192_22k', pretrained=True, num_classes=num_classes)
    #model = create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    #model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model

def train_and_validate(model, train_loader, val_loader, device, epochs=10, lr=5e-5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / len(val_loader.dataset)
        print(f"Validation Accuracy: {val_acc:.2f}%")

        accuracies.append(val_acc)
    
    print('- - - - - - - - - - - - - - - - -')

    return np.mean(accuracies) # retorna a acurácia média de cada época

def main():
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    DATA_PATH = 'KTHIDOL/dumbo/cloudy' 
    EPOCHS = 10 
    LEARNING_RATE = 0.001 
    N_SPLITS = 5

    print('- - - - - Carregando os dados - - - - - -')
    dataset, num_classes = load_data(DATA_PATH, IMG_SIZE)
    kf = StratifiedKFold(n_splits=N_SPLITS)

    print('- - - - - - - - - - - - - - - - - - - - -')
    print(f"Número de classes: {num_classes}")
    print(f"Quantidade de imagens do dataset: {len(dataset)}")
    print('- - - - - - - - - - - - - - - - - - - - -')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo usado: {device}")
    
    accuracies = []
	
    print('- - - - - Treinando o modelo - - - - -')
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.imgs,dataset.targets)):
        print(f"- - - - Fold {fold+1}/{N_SPLITS} - - - -")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        model = define_model(num_classes)
        val_acc = train_and_validate(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE)
        accuracies.append(val_acc)
        print(f"Acurácia Média - fold[{fold+1}]: {val_acc:.2f}%")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print(f"- - - - Resultados da Validação Cruzada - - - -")
    print('- - - - - convnext_base - - - - -\n')
    print(f"Acurácia Média: {mean_acc:.2f}%")
    print(f"Desvio Padrão da Acurácia: {std_acc:.4f}%")
    print('- - - - - Treinamento finalizado - - - - -')

    #salva os resultados
    results = {
        'Fold': range(1, N_SPLITS+1),
        'Accuracy': accuracies
    }
    df = pd.DataFrame(results)
    df.to_csv('cross_validation_results.csv', index=False)

if __name__ == "__main__":
    main()
