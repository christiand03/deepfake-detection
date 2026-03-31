import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class DeepFakeVideoDataset(Dataset):
    def __init__(self, video_folders, labels, transform=None):
        """
        video_folders: Liste von Pfaden zu den Ordnern (z.B. ['Processed_Dataset/frames/Real/001', ...])
        labels: Liste von Labels (0 für Real, 1 für Fake)
        transform: PyTorch Transforms (z.B. Normalisierung)
        """
        self.video_folders = video_folders
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Gibt zurück, wie viele Videos wir insgesamt haben
        return len(self.video_folders)

    def __getitem__(self, idx):
        folder_path = self.video_folders[idx]
        label = self.labels[idx]

        # Finde alle .jpg Bilder im Ordner und sortiere sie alphabetisch!
        # (Extrem wichtig, damit die zeitliche Reihenfolge der 16 Frames stimmt)
        frame_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        
        frames = []
        for frame_path in frame_paths:
            # Bild laden und von BGR (OpenCV Format) zu RGB (Standard) konvertieren
            img = Image.open(frame_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img) # Wandelt in Tensor um und normalisiert
                
            frames.append(img)

        # frames ist jetzt eine Liste von 16 Tensoren der Form (3, 224, 224).
        # Wir stapeln sie übereinander zu EINEM Tensor der Form (16, 3, 224, 224)
        video_tensor = torch.stack(frames)

        # PyTorch erwartet Labels oft als Float-Tensoren für bestimmte Loss-Funktionen
        return video_tensor, torch.tensor(label, dtype=torch.float32)

def get_dataloaders(base_dir="Processed_Dataset/frames", batch_size=4):
    """
    Sucht alle Daten, macht einen Train/Test Split und erstellt die DataLoader.
    """
    all_folders = []
    all_labels = []

    # 1. Real-Videos sammeln (Label 0)
    real_dir = os.path.join(base_dir, "Real")
    for folder_name in os.listdir(real_dir):
        all_folders.append(os.path.join(real_dir, folder_name))
        all_labels.append(0)

    # 2. Fake-Videos sammeln (Label 1)
    fake_dir = os.path.join(base_dir, "Fake")
    for folder_name in os.listdir(fake_dir):
        all_folders.append(os.path.join(fake_dir, folder_name))
        all_labels.append(1)

    # 3. Train / Validation Split (80% Training, 20% Validierung/Test)
    # random_state=42 sorgt dafür, dass die Aufteilung bei jedem Start gleich bleibt
    X_train, X_val, y_train, y_val = train_test_split(
        all_folders, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # 4. Image Transforms definieren
    # Wir nutzen die Standard-ImageNet Normalisierung. Fast alle vortrainierten 
    # Transformer (ViT, VideoMAE) erwarten exakt diese Werte!
    transform = transforms.Compose([
        transforms.ToTensor(), # Macht aus Pixeln (0-255) Kommazahlen (0.0-1.0)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. Datasets erstellen
    train_dataset = DeepFakeVideoDataset(X_train, y_train, transform=transform)
    val_dataset = DeepFakeVideoDataset(X_val, y_val, transform=transform)

    # 6. DataLoader erstellen
    # shuffle=True im Training ist extrem wichtig, damit das Modell nicht auswendig lernt
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)

# --- Test-Block ---
if __name__ == "__main__":
    print("Erstelle DataLoader...")
    train_loader, val_loader, train_size, val_size = get_dataloaders(batch_size=2)
    
    print(f"Trainings-Videos: {train_size}")
    print(f"Validierungs-Videos: {val_size}")

    # Lade exakt EINEN Batch (2 Videos) aus dem DataLoader zum Testen
    video_batch, label_batch = next(iter(train_loader))
    
    print("\n--- Test erfolgreich! ---")
    print(f"Form des Video-Tensors: {video_batch.shape}")
    print(f"Form der Labels: {label_batch.shape}")
    print(f"Labels in diesem Batch: {label_batch}")
    
    # Die Form des Video-Tensors sollte sein:
    # [2, 16, 3, 224, 224] -> [Batch-Size, Frames, Farbkanäle, Höhe, Breite]