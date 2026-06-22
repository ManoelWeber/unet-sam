import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Importação da nova biblioteca
import segmentation_models_pytorch as smp


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = r"C:\Projetos\SAM_V2"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "mascaras", "train")

VALID_IMG_DIR = os.path.join(BASE_DIR, "valid")
VALID_MASK_DIR = os.path.join(BASE_DIR, "mascaras", "valid")

MODEL_DIR = os.path.join(BASE_DIR, "modelos")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "unet_desgaste.pth")

IMG_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATASET
# ============================================================

class DesgasteDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        imagens = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        self.pares = []

        for img_name in imagens:
            name_no_ext = os.path.splitext(img_name)[0]
            mask_name = name_no_ext + "_mask.png"
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                self.pares.append((img_name, mask_name))
            else:
                print(f"Aviso: máscara não encontrada, ignorando: {img_name}")

        print(f"Total de pares imagem/máscara encontrados: {len(self.pares)}")

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        img_name, mask_name = self.pares[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Máscara não encontrada: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        mask = (mask > 0.5).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask


# ============================================================
# LOSS E MÉTRICAS
# ============================================================

# Substituímos as funções complexas manuais por uma integração limpa com a SMP
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    dice = (2.0 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )

    return dice.item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


# ============================================================
# TREINO E VALIDAÇÃO
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for images, masks in tqdm(loader, desc="Treino"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validação"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)
            total_iou += iou_score(outputs, masks)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)

    return avg_loss, avg_dice, avg_iou


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Usando dispositivo: {DEVICE}")

    train_dataset = DesgasteDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_SIZE)
    valid_dataset = DesgasteDataset(VALID_IMG_DIR, VALID_MASK_DIR, IMG_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # A grande mudança: Chamamos a U-Net pronta da biblioteca com Transfer Learning
    model = smp.Unet(
        encoder_name="resnet34",        # Arquitetura base consolidada
        encoder_weights="imagenet",     # Pesos pré-treinados para visão computacional
        in_channels=3,                  # Imagens RGB
        classes=1                       # Saída binária (0 fundo, 1 desgaste)
    ).to(DEVICE)

    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_iou = 0

    for epoch in range(EPOCHS):
        print(f"\nÉpoca {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        valid_loss, valid_dice, valid_iou = validate(model, valid_loader, loss_fn)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Dice Score: {valid_dice:.4f}")
        print(f"IoU Score:  {valid_iou:.4f}")

        if valid_iou > best_iou:
            best_iou = valid_iou
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Modelo salvo: {MODEL_PATH}")

    print("\nTreinamento finalizado.")
    print(f"Melhor IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()