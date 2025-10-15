#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
# 1️⃣ 读取数据
file_path = '/Users/qijinnian/Desktop/data/统计学习数据/UniversalBank.csv'
df = pd.read_csv(file_path)
print(df.head(10))
# 2️⃣ 删除无关特征
df = df.drop(['ID', 'ZIP Code'], axis=1)
# 3️⃣ 分离特征与标签
X = df.drop('Personal Loan', axis=1)
y = df['Personal Loan']
# 4️⃣ 划分训练集与测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 5️⃣ 标准化数据（对连续变量进行缩放）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 6️⃣ 建立SVM模型（使用RBF核函数）
clf = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
clf.fit(X_train_scaled, y_train)
# 7️⃣ 在测试集上预测
y_pred = clf.predict(X_test_scaled)
# 8️⃣ 计算准确率和混淆矩阵
accuracy = accuracy_score(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
print("✅ 测试集准确率：{:.4f}".format(accuracy))
print("\n混淆矩阵：\n", matrix)
