from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt

# 月亮形状数据（非线性）
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# 绘图
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
plt.title("Moon-shaped data (non-linear)")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. 生成非线性数据
# -------------------------------
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 2. 定义线性核 SVM 和 RBF 核 SVM
# -------------------------------
linear_svm = SVC(kernel='linear')
rbf_svm    = SVC(kernel='rbf', gamma='scale')

linear_svm.fit(X_train, y_train)
rbf_svm.fit(X_train, y_train)

# -------------------------------
# 3. 绘制决策边界的函数
# -------------------------------
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k')
    plt.title(title)
    plt.show()

# -------------------------------
# 4. 展示线性核与RBF核的区别
# -------------------------------
plot_decision_boundary(linear_svm, X, y, "Linear SVM on Moon Data")
plot_decision_boundary(rbf_svm, X, y, "RBF Kernel SVM on Moon Data")

# -------------------------------
# 5. 输出准确率对比
# -------------------------------
print("Linear SVM Accuracy:", linear_svm.score(X_test, y_test))
print("RBF SVM Accuracy:", rbf_svm.score(X_test, y_test))
