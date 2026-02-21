import numpy as np
import matplotlib.pyplot as plt

# ------------- ✦ Признаки ✦ -------------
X = np.array([[3, 2], [2, 2], [1, 1], [2, 1.5], [1, 4], [3, 5], [5, 5], [2, 4]])
# ------------- ✦ Метки класов ✦ -------------
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


# ------------- ✦ Логистическая регрессия ✦ -------------
class LogisticRegression:
    def __init__(self, input_dim, learning_rate=0.01):
        self.weights = np.random.randn(input_dim)
        self.bias = 0
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cross_entropy_loss(self, y_true, y_pred):
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def train(self, X, y_true, num_epochs=100):
        for epoch in range(num_epochs):
            scores = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(scores)
            loss = np.mean(self.cross_entropy_loss(y_true, y_pred))
            grad_y_pred = y_pred - y_true
            grad_weights = np.dot(X.T, grad_y_pred)
            grad_bias = np.sum(grad_y_pred)
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(scores)
        return np.round(y_pred)


# ------------- ✦ Инициализация и обучение логистической регрессии ✦ -------------
logistic_regression = LogisticRegression(input_dim=2, learning_rate=0.01)
logistic_regression.train(X, y)

# ------------- ✦ Прогнозирование меток классов ✦ -------------
predictions = logistic_regression.predict(X)

# ------------- ✦ Вывод результатов ✦ -------------
print("Предсказанные метки классов:", predictions)
print("Истинные метки классов:", y)
