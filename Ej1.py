import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.linalg
from sklearn.decomposition import PCA
import seaborn as sns

# Load the CSV file
file_path = "C:/Users/iegre/Downloads/dataset02.csv" 
data = pd.read_csv(file_path)

# Access all values excluding the first column and the first row
dataset = data.iloc[0:, 1:]

# Calcular la matriz de similitud en el espacio original
def calculate_similarity_matrix(X, sigma):
    pairwise_sq_dists = np.square(np.linalg.norm(X[:, np.newaxis] - X, axis=2))
    similarity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return similarity_matrix

# Suponemos un valor de sigma
sigma = 1.0
X = dataset.values

similarity_matrix_original = calculate_similarity_matrix(X, sigma)

# Aplicar PCA y reducir la dimensionalidad
def apply_pca(X, d):
    pca = PCA(n_components=d)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

# Reducir a d = 2, 6, 10
d_values = [2, 6, 10]
X_reduced_list = []
for d in d_values:
    X_reduced, pca = apply_pca(X, d)
    X_reduced_list.append((d, X_reduced, pca))

# Calcular matrices de similitud en los espacios reducidos
similarity_matrices_reduced = []
for d, X_reduced, pca in X_reduced_list:
    similarity_matrix_reduced = calculate_similarity_matrix(X_reduced, sigma)
    similarity_matrices_reduced.append((d, similarity_matrix_reduced))


# Función para graficar matrices de similitud
def plot_similarity_matrices(original_matrix, reduced_matrices, d_values):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(d_values) + 1, 1)
    sns.heatmap(original_matrix, cmap='viridis')
    plt.title('Original')
    
    for i, (d, reduced_matrix) in enumerate(reduced_matrices):
        plt.subplot(1, len(d_values) + 1, i + 2)
        sns.heatmap(reduced_matrix, cmap='viridis')
        plt.title(f'Reduced d={d}')
    
    plt.tight_layout()
    plt.show()

# Graficar las matrices de similitud
plot_similarity_matrices(similarity_matrix_original, similarity_matrices_reduced, d_values)



