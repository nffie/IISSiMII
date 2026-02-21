from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------- ✦ Загрузка датасета о классификации вин ✦ -------------
wine = load_wine()
X = wine.data
y = wine.target

# ------------- ✦ Разделение данных на обучающий и тестовый наборы ✦ -------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------- ✦ Создание и обучение MLP классификатора ✦ -------------
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, solver='sgd')
mlp.fit(X_train, y_train)

# ------------- ✦ Предсказание на тестовом наборе ✦ -------------
predictions = mlp.predict(X_test)

# ------------- ✦ Оценка точности модели ✦ -------------
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
