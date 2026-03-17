import numpy as np
import matplotlib.pyplot as plt

# 设定随机种子
np.random.seed(42)
# 生成类别0的数据：中心在(1, 1)
class_0 = np.random.randn(50, 2) + np.array([1, 1])
y0 = np.zeros((50, 1))
# 生成类别1的数据：中心在(3, 3)
class_1 = np.random.randn(50, 2) + np.array([3, 3])
y1 = np.ones((50, 1))

# 合并数据
X = np.vstack((class_0, class_1))
y = np.vstack((y0, y1))

# 打乱顺序
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='RdBu', edgecolor='k')
plt.title("Manual Generated Binary Classes")
plt.show()


# 逻辑回归的灵魂：Sigmoid函数
def sigmoid(z):
    # 将任何实数映射到(0, 1)之间
    return 1 / (1 + np.exp(-z))


class SimpleLogisticRegression:
    def __init__(self, lr=0.1, iterations=10000):
        self.lr = lr
        self.iterations = iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape
        # 权重初始化
        self.w = np.random.randn(n, 1)
        self.b = 0

        for i in range(self.iterations):
            # 1.计算线性组合
            z = np.dot(X, self.w) + self.b
            # 2.激活函数转换：预测概率
            y_pred = sigmoid(z)

            # 3.计算梯度
            # dw = X.T * (y_pred - y) / m
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            # 4.更新
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if i % 20 == 0:
                # 交叉熵损失(Cross-Entropy)
                loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1-y_pred + 1e-15))
                print(f"Iteration {i}: Loss = {loss:.4f}")


# 实例化与训练
model = SimpleLogisticRegression()
model.fit(X, y)

# 绘制结果
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='RdBu', edgecolors='k')
# 绘制决策边界线: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
x1_boundary = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_boundary = -(model.w[0]*x1_boundary + model.b) / model.w[1]
plt.plot(x1_boundary, x2_boundary, color='green', label='Decision Boundary')
plt.legend()
plt.show()