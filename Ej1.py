import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.linalg
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

# Load the CSV file
csv_file_path = "C:/Users/iegre/OneDrive/Escritorio/repositorio Git/Metodos numericos/TP3/TP3_Metodos/dataset02.csv" 
txt_file_path = "C:/Users/iegre/OneDrive/Escritorio/repositorio Git/Metodos numericos/TP3/TP3_Metodos/y2.txt"
data = pd.read_csv(csv_file_path)

# Access all values excluding the first column and the first row
dataset = data.iloc[0:, 1:]

#Acceder a todos los valores de y.txt
def read_text_file(file_path):
    data = []
    with open(file_path, mode='r') as file:
        for line in file:
            data.append(float(line.strip()))
    return data

# Calcular la matriz de similitud en el espacio original
def calculate_similarity_matrix(X, sigma):
    pairwise_sq_dists = np.square(np.linalg.norm(X[:, np.newaxis] - X, axis=2))
    similarity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return similarity_matrix
    # pairwise_dists = squareform(pdist(X, 'euclidean'))
    # K = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
    # return K

# Suponemos un valor de sigma
sigma = 1.0
X = dataset.values
Y = read_text_file(txt_file_path)


similarity_matrix_original = calculate_similarity_matrix(X, sigma)

#Aplicar SVD
def apply_SVD(X):
    u, s, vt = np.linalg.svd(X,full_matrices=False)
    return u, s, vt

# Aplicar PCA y reducir la dimensionalidad
def apply_pca(X, d):
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = apply_SVD(X_centered)
    V_d = Vt.T[:, :d]
    X_reduced = X_centered @ V_d

    return X_reduced

def errorDePrediccion(X,beta,y):
    return np.linalg.norm((X @ beta)-y) #puede que aca haya que restar el mean de y

def resolverSistemaConPFeatures(X,y):
    u, s, vt = apply_SVD(X)
    sd = s[slice(102)]
    ud = u[:,:102]
    vd = vt.T[:,:102]
    beta = vd @ np.linalg.inv(np.diag(sd)) @ ud.T @ y #preguntar porque aca me quedan 106 valores en vez de 102
    return beta

def resolverSistemaConDFeatures(X,y,d):
    X_centered = X - np.mean(X, axis=0)
    y_centered = y - np.mean(y)
    u, s, vt = apply_SVD(X_centered)
    sd = s[slice(d)]
    ud = u[:,:d]
    beta = np.linalg.inv(np.diag(sd)) @ ud.T @ y_centered
    return beta

# Reducir a d = 2, 6, 10
d_values = [2, 6, 10]
X_reduced_list = []
for d in d_values:
    X_reduced = apply_pca(X, d)
    X_reduced_list.append((d, X_reduced))

#Hacer SVD
u, s, vt = apply_SVD(X)

#Remover valores singulares que son 0
s_matrix = s[slice(102)]
u_matrix = u[:,:102]
vt_matrix = vt[:102,:]


#Resolver sistema
betaP = resolverSistemaConPFeatures(X,Y)
beta1 = resolverSistemaConDFeatures(X,Y,1)
beta2 = resolverSistemaConDFeatures(X,Y,2)
beta3 = resolverSistemaConDFeatures(X,Y,3)
beta6 = resolverSistemaConDFeatures(X,Y,6)
beta10 = resolverSistemaConDFeatures(X,Y,10)
print(beta1)
print(beta2)
print(beta3)
print(beta6)
print(beta10)
print(betaP)

# Calcular matrices de similitud en los espacios reducidos
similarity_matrices_reduced = []
for d, X_reduced in X_reduced_list:
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

#Graficar los autovalores mas importantes
numeroDeValoresSingulares = np.arange(0,102)
plt.figure(figsize=(8,6))
plt.plot(numeroDeValoresSingulares,s_matrix,'o-')
plt.xlabel("Dimension")
plt.ylabel("Valor de valor singular")
plt.grid(True)
plt.show()

#Graficar PC1 vs PC2 vs PC3

fig3 = plt.figure()
ax1 = fig3.add_subplot(111,projection='3d')
x1s = []
y1s = []
z1s = []

for j in range(X.shape[0]):
    x1s.append (vt[0,:] @ X[j,:].T)
    y1s.append (vt[1,:] @ X[j,:].T)
    z1s.append ( vt[2,:] @ X[j,:].T)

ax1.scatter(x1s,y1s,z1s,c=Y)

plt.show()

#Graficar PC1 vs PC2
fig4 = plt.figure()
ax2 = fig4.add_subplot(111)
x2s = []
y2s = []
for j in range(X.shape[0]):
    x2s.append (vt[0,:] @ X[j,:].T)
    y2s.append (vt[1,:] @ X[j,:].T)

ax2.scatter(x2s,y2s,c=Y)
plt.show()

#Graficar error a medida que aumentan las dimensiones
fig5 = plt.figure()
ax3 = fig5.add_subplot(111)
errores = []
dimensiones = np.arange(2,103)
for i in range(2,102):
    beta_i = resolverSistemaConDFeatures(X,Y,i)
    errores.append (errorDePrediccion(apply_pca(X,i),beta_i,Y)) #preguntar si tengo que cnetralizar Y aca tambien y porque con dimension 1 me da un error altisimo
errores.append(errorDePrediccion(X,resolverSistemaConPFeatures(X,Y),Y))
ax3.plot(dimensiones,errores,'o-')
plt.xlabel("Dimensionse")
plt.ylabel("Error")
plt.show()


#Graficar el Y dado contra el y calculado con X*Beta
numeroDeValores = np.arange(0,2000)
fig6 = plt.figure()
ax4 = fig6.add_subplot(111)
ax4.scatter(numeroDeValores, Y, color = "r", label = 'Soluciones exactas')
ax4.scatter(numeroDeValores, X @ betaP,marker='x', color="g", label='Soluciones estimadas' )
plt.title("Soluciones exactas y aproximadas")
plt.show()