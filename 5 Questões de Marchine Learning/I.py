# 1: Implemente um classificador KNN para o conjunto de dados Iris.
# Divida os dados em conjuntos de treino e teste, treine o modelo e avalie a precisão.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados I Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o classificador KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)

# Avaliar a precisão
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo KNN: {accuracy:.2f}')