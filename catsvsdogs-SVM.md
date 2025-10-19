import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -------------------------------
# 数据集路径
# -------------------------------
train_dir = "/Users/qijinnian/Desktop/learn/statics learning method/dogs-vs-cats-redux-kernels-edition/train"

# -------------------------------
# 读取图片并转换为向量
# -------------------------------
image_size = (128, 128)  # 缩小尺寸加快 SVM 训练

X = []
y = []

img_files = sorted(os.listdir(train_dir))
for idx, img_name in enumerate(img_files):
    img_path = os.path.join(train_dir, img_name)
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(image_size)
        img_array = np.array(img).flatten()  # 展平为一维向量
        X.append(img_array)
        # 标签：前12500张是猫（0），后12500张是狗（1）
        label = 0 if idx < 500 else 1
        y.append(label)
    except Exception as e:
        print(f"Warning: Failed to open {img_path}: {e}")

X = np.array(X)
y = np.array(y)

# -------------------------------
# 数据拆分
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 特征标准化
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------------------
# SVM 训练
# -------------------------------
svm_clf = SVC(kernel='linear')  # 线性核
svm_clf.fit(X_train, y_train)

# -------------------------------
# 预测和准确率
# -------------------------------
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"SVM 分类准确率: {acc:.4f}")
