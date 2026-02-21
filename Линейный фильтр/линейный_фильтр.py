import numpy as np
import matplotlib.pyplot as plt

# ------------- ✦ Признаки ✦ -------------
X = np.array([[3, 2], [2, 2], [1, 1], [2, 1.5], [1, 4], [3, 5], [5, 5], [2, 4]])
# ------------- ✦ Метки класов ✦ -------------
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


# ------------- ✦ Линейный классификатор ✦ -------------
class LinearClassifier:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for i in range(num_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                predicted = 1 if linear_output >= 0 else 0
                update = self.learning_rate * (y[i] - predicted) * 2
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predicted = np.where(linear_output >= 0, 1, 0)
        return predicted


# ------------- ✦ Инициализация и обучение линейного классификатора ✦ -------------
classifier = LinearClassifier(learning_rate=0.01, num_iterations=1000)
classifier.fit(X, y)

# ------------- ✦ Прогнозирование меток классов ✦ -------------
predictions = classifier.predict(X)

# ------------- ✦ Вывод результатов ✦ -------------
print("Предсказанные метки классов:", predictions)
print("Истинные метки классов:", y)

# ------------- ✦ Визуализация ✦ -------------
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, marker="o")
plt.xlabel("x1")
plt.ylabel("x2")

# ------------- ✦ Построение разделяющей прямой ✦ -------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu)
plt.show()
