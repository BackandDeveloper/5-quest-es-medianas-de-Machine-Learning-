# 4:Utilize o algoritmo K-Means para agrupar o conjunto de dados Iris.
# Visualize os clusters resultantes.

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data

# Criar o modelo K-Means
kmeans = KMeans(n_clusters=3, random_state=42)

# Treinar o modelo
kmeans.fit(X)

# Obter os rótulos dos clusters
labels = kmeans.labels_

# Visualizar os clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.title('Clusterização K-Means do Conjunto de Dados Iris')
plt.show()