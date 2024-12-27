# 2: Utilize a regressão linear para prever o preço de casas com base no conjunto de dados Boston Housing.
# Divida os dados, treine o modelo e avalie o desempenho usando o erro quadrático médio (MSE).

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar o conjunto de dados Boston Housing
boston = load_boston()
X = boston.data
y = boston.target

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo de regressão linear
lr = LinearRegression()

# Treinar o modelo
lr.fit(X_train, y_train)

# Fazer previsões
y_pred = lr.predict(X_test)

# Avaliar o desempenho
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio (MSE): {mse:.2f}')