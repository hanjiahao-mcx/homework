import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import googlenet, GoogLeNet_Weights
import pandas as pd

# -------------------------------
# 自定义 Dataset（numpy 转 tensor）
# -------------------------------
class CatsDogsDataset(Dataset):
    def __init__(self, img_dir, train=True):
        self.img_dir = img_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.train = train

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to open {img_path}: {e}")
            image = Image.new("RGB", (224,224))

        # Resize
        image = image.resize((224,224))

        # 随机水平翻转（训练集）
        if self.train and np.random.rand() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # PIL -> numpy -> tensor，HWC -> CHW
        image = np.array(image).astype(np.float32) / 255.0  # 归一化到 [0,1]
        image = torch.from_numpy(image).permute(2,0,1)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image = (image - mean) / std

        # 标签
        if self.train:
            label = 0 if idx < 12500 else 1
        else:
            label = -1

        return image, label, img_name  # test 时返回文件名

# -------------------------------
# 参数
# -------------------------------
train_dir = "/root/cats-vs-dogs/train"
test_dir  = "/root/cats-vs-dogs/test"

batch_size = 32
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 数据加载
# -------------------------------
train_dataset = CatsDogsDataset(train_dir, train=True)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CatsDogsDataset(test_dir, train=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# GoogLeNet 模型
# -------------------------------
weights = GoogLeNet_Weights.DEFAULT
model = googlenet(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 二分类
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 训练函数
# -------------------------------
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc  = running_corrects.double() / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# -------------------------------
# 开始训练
# -------------------------------
model = train_model(model, train_loader, criterion, optimizer, num_epochs)

# -------------------------------
# 测试预测并输出 CSV
# -------------------------------
model.eval()
results = []
with torch.no_grad():
    for inputs, _, img_names in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        for img_name, pred in zip(img_names, preds):
            results.append({"filename": img_name, "label": pred})

df = pd.DataFrame(results)
df.to_csv("test_predictions.csv", index=False)
print("预测结果已保存到 test_predictions.csv")
