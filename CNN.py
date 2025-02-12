import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import string
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
CHARS = string.ascii_letters + string.digits  # All possible characters in captcha
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHARS)

class CaptchaDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Get label from filename (assuming filename is the label)
        label = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load and convert image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label_tensor = torch.zeros(len(label), dtype=torch.long)
        for i, char in enumerate(label):
            label_tensor[i] = CHAR_TO_IDX[char]

        return image, label_tensor

class CaptchaCNN(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaCNN, self).__init__()
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 10, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_chars * NUM_CLASSES)
        )
        
        self.num_chars = num_chars

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = x.view(x.size(0), self.num_chars, NUM_CLASSES)
        return x

def train_model(model, train_loader, valid_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))
        
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
                valid_loss += loss.item()
        valid_losses.append(valid_loss / len(valid_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}')
    
    return train_losses, valid_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 2)
            
            # Compare each character
            correct += (predicted == labels).sum().item()
            total += labels.numel()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset parameters
    img_height = 32
    img_width = 80
    num_chars = 5  # Adjust based on your CAPTCHA length
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    dataset_path = 'captcha_images'  # Change this to your dataset path
    if not os.path.exists(dataset_path):
        print(f"Please place your CAPTCHA images in a folder named '{dataset_path}'")
        return
    
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split dataset
    train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    train_paths, valid_paths = train_test_split(train_paths, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = CaptchaDataset(train_paths, transform=transform)
    valid_dataset = CaptchaDataset(valid_paths, transform=transform)
    test_dataset = CaptchaDataset(test_paths, transform=transform)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = CaptchaCNN(num_chars).to(device)
    
    # Train model
    print("Training model...")
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, device)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'captcha_model.pth')
    print("\nModel saved as 'captcha_model.pth'")

if __name__ == '__main__':
    main()
