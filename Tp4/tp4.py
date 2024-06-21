import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

np.random.seed(12345)

def F1(x, A, b):
    return np.linalg.norm(A @ x - b)**2

def F2(x, A, b, delta):
    return F1(x, A, b) + delta * np.linalg.norm(x)**2

def hessianoF1(A):
    return 2 * A.T @ A

def hessianoF2(A, delta):
    return hessianoF1(A) + 2 * delta * np.eye(A.shape[1])

def gradienteF1(x, A, b):
    return 2 * A.T @ (A @ x - b)

def gradienteF2(x, A, b, delta):
    return gradienteF1(x, A, b) + 2 * delta * x

def gradiente_descendente(A, b, F, grad_F, F2, grad_F2, num_iter,valorDelta):
    D = A.shape[1]
    x_F = x_F2 = np.random.rand(D)
     
    U, S, Vt = np.linalg.svd(A)
    sigma_max = max(S)
    delta2 = valorDelta * sigma_max ** 2
    H1 = hessianoF1(A)
    lambda_max = max(np.linalg.eigvals(H1))
    step = 1 / lambda_max
    history_F = []
    history_F2 = []
    norm_residual_F = []
    norm_residual_F2 = []
    normsDeX1 = []
    normsDeX2 = []
    history_F.append(F(x_F, A, b))
    norm_residual_F.append(np.linalg.norm(A @ x_F - b)**2)
    normsDeX1.append(np.linalg.norm(x_F))
    history_F2.append(F2(x_F2, A, b, delta2))
    norm_residual_F2.append(np.linalg.norm(A @ x_F2 - b)**2)
    normsDeX2.append(np.linalg.norm(x_F2))


    
    # Gradiente descendente para F(x)
    for _ in range(num_iter):
        grad_F_value = grad_F(x_F, A, b)
        x_F = x_F - step * grad_F_value
        history_F.append(F(x_F, A, b))
        norm_residual_F.append(np.linalg.norm(A @ x_F - b)**2)
        normsDeX1.append(np.linalg.norm(x_F))
    
    # Gradiente descendente para F2(x)
    for _ in range(num_iter):
        grad_F2_value = grad_F2(x_F2, A, b, delta2)
        x_F2 = x_F2 - step * grad_F2_value
        history_F2.append(F2(x_F2, A, b, delta2))
        norm_residual_F2.append(np.linalg.norm(A @ x_F2 - b)**2)
        normsDeX2.append(np.linalg.norm(x_F2))

    
    return x_F, x_F2, history_F, history_F2, norm_residual_F, norm_residual_F2,normsDeX1,normsDeX2

def resolver_con_svd(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b 

def main():
    N = 5  # filas
    D = 100  # columnas
    A = np.random.rand(N, D)
    b = np.random.rand(N)
    num_iter = 1000
    
    x_F, x_F2, history_F, history_F2, norm_residual_F, norm_residual_F2,normaDeX1,normaDeX2 = gradiente_descendente(A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-2)

    # Graficar la evolución del costo
    plt.figure(figsize=(12, 6))
    plt.loglog(history_F, label='F1(x)')
    plt.loglog(history_F2, label='F2(x)')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.title('Evolución del costo durante las iteraciones')
    plt.legend()
    plt.show()
    
    x_svd = resolver_con_svd(A, b)

    plt.figure(figsize=(12, 6))
    plt.plot(x_F, label='Solución GD F1(x)', linestyle='--')
    plt.plot(x_F2, label='Solución GD F2(x)', linestyle=':')
    plt.plot(x_svd, label='Solución SVD', linestyle='-')
    plt.xlabel('Índice de la variable')
    plt.ylabel('Valor de la variable')
    plt.title('Comparación de soluciones obtenidas')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.loglog(norm_residual_F, label='Norma al cuadrado de Ax-b (F1)')
    plt.loglog(norm_residual_F2, label='Norma al cuadrado de Ax-b (F2)')
    plt.xlabel('Iteraciones')
    plt.ylabel('Norma al cuadrado')
    plt.title('Evolución de la norma al cuadrado de Ax-b durante las iteraciones')
    plt.legend()
    plt.show()
    
    # Calcular errores relativos
    # error_relativo_F = [np.linalg.norm(x_F - x_svd) / np.linalg.norm(x_svd) for x_F in history_F]
    # error_relativo_F2 = [np.linalg.norm(x_F2 - x_svd) / np.linalg.norm(x_svd) for x_F2 in history_F2]
    error_relativo_F = np.abs((x_F - x_svd) / x_svd)
    error_relativo_F2 = np.abs((x_F2 - x_svd) / x_svd) 
    
    # Graficar errores relativos
    plt.figure(figsize=(12, 6))
    plt.loglog(error_relativo_F, label='Error relativo (F1)')
    plt.loglog(error_relativo_F2, label='Error relativo (F2)')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error relativo')
    plt.title('Evolución del error relativo respecto a la solución SVD')
    plt.legend()
    plt.show()

    # Norma de los vectores x en cada iteracion
    plt.figure(figsize=(12, 6))
    plt.loglog(normaDeX1, label='Norma de x1')
    plt.loglog(normaDeX2, label='Norma de x2')
    plt.xlabel('Iteraciones')
    plt.ylabel('Valor de la norma')
    plt.title('Valores de la norma de x durante las iteraciones')
    plt.legend()
    plt.show()

    # Graficar F2(x) co distintos valores de delta
    x_F, x_F2, history_F_001, history_F2_001, norm_residual_F, norm_residual_F2,normaDeX1,normaDeX2 = gradiente_descendente(A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-2)
    x_F, x_F2, history_F_0001, history_F2_0001, norm_residual_F, norm_residual_F2,normaDeX1,normaDeX2 = gradiente_descendente(A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-3)
    x_F, x_F2, history_F_00001, history_F2_00001, norm_residual_F, norm_residual_F2,normaDeX1,normaDeX2 = gradiente_descendente(A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-4)
    x_F, x_F2, history_F_000001, history_F2_000001, norm_residual_F, norm_residual_F2,normaDeX1,normaDeX2 = gradiente_descendente(A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-5)
    plt.figure(figsize=(12, 6))
    plt.loglog(history_F2_001, label='F2(x) 0,01')
    plt.loglog(history_F2_0001, label='F2(x) 0,001')
    plt.loglog(history_F2_00001, label='F2(x) 0,0001')
    plt.loglog(history_F2_000001, label='F2(x) 0,00001')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.title('Evolución del costo durante las iteraciones')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
