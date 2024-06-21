import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

np.random.seed(12345)

N = 5  # filas
D = 100  # columnas
A = np.random.rand(N, D)
b = np.random.rand(N)
num_iter = 1000

def F1(x):
    return np.linalg.norm(A @ x - b)**2

def F2(x,delta):
    return F1(x, A, b) + delta * np.linalg.norm(x)**2

def hessianoF1():
    return 2 * A.T @ A

def hessianoF2(delta):
    return hessianoF1(A) + 2 * delta * np.eye(A.shape[1])

def gradienteF1(x):
    return 2 * A.T @ (A @ x - b)

def gradienteF2(x,delta):
    return gradienteF1(x, A, b) + 2 * delta * x

def gradiente_descendente( num_iter,valorDelta):
    D = A.shape[1]
    x_F = x_F2 = np.random.rand(D)
     
    U, S, Vt = np.linalg.svd(A)
    sigma_max = max(S)
    delta2 = valorDelta * sigma_max ** 2
    lambda_max = S[0]**2
    step = 1 / lambda_max
    x_k = np.zeros((D,num_iter)) #Cada columna es el vector x de la iteración k

    x_k[:,0] = np.random.rand(D)# Condiciones iniciales aleatorias
    f_x = np.zeros(num_iter) #F(x) de cada iteración
    f_x[0] = F1(x_k[:,0])

    #Gradient descent
    for i in range(num_iter-1):
        x_k[:,i+1] = x_k[:,i] - step * gradienteF1(x_k[:,i])
        f_x[i+1] = F1(x_k[:,i+1])

    return x_k, f_x

def resolver_con_svd(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b 


x_F, history_F, = gradiente_descendente(num_iter, 1e-2)


# Graficar la evolución del costo
plt.figure(figsize=(12, 6))
plt.loglog(history_F, label='F1(x)')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.title('Evolución del costo durante las iteraciones')
plt.legend()
plt.show()

x_svd = resolver_con_svd(A, b)



