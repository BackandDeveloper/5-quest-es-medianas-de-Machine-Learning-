# 5: Aplique a Análise de Componentes Principais (PCA) ao conjunto de dados Digits.
# Reduza a dimensionalidade para 2 componentes principais e visualize os dados.

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Digits
digits = load_digits()
X = digits.data
y = digits.target

# Aplicar PCA para reduzir a dimensionalidade
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualizar os dados reduzidos
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=40)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA do Conjunto de Dados Digits')
plt.colorbar()
plt.show()