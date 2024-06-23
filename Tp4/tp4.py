import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

np.random.seed(0)
def F1(x, A, b):
    return np.linalg.norm(A @ x - b)**2

def F2(x, A, b, delta):
    return F1(x, A, b) + delta * np.linalg.norm(x)**2

def hessianoF1(A):
    return 2 * A.T @ A

def gradienteF1(x, A, b):
    return 2 * A.T @ (A @ x - b)

def gradienteF2(x, A, b, delta):
    return gradienteF1(x, A, b) + 2 * delta * x

def gradiente_descendente(x, A, b, F, grad_F, F2, grad_F2, num_iter,valorDelta):
    x_F = x_F2 = x.copy()
    U, S, Vt = np.linalg.svd(A)
    sigma_max = max(S)
    delta2 = valorDelta * sigma_max 
    H1 = hessianoF1(A)
    # lambda_max = max(np.linalg.eigvals(H1))
    lambda_max = 2*sigma_max **2
    step = 1 / lambda_max
    history_F = []
    history_F2 = []
    error_residual_F = []
    norm_residual_F2 = []
    normsDeX1 = []
    normsDeX2 = []
    history_F.append(F(x_F, A, b))
    error_residual_F.append(np.linalg.norm(A @ x_F - b))
    normsDeX1.append(np.linalg.norm(x_F))
    history_F2.append(F2(x_F2, A, b, delta2))
    norm_residual_F2.append(np.linalg.norm(A @ x_F2 - b))
    normsDeX2.append(np.linalg.norm(x_F2))



    
    # Gradiente descendente para F(x)
    for _ in range(num_iter):   
        grad_F_value = grad_F(x_F, A, b)
        x_F = x_F - step * grad_F_value
        history_F.append(F(x_F, A, b))
        error_residual_F.append(np.linalg.norm(A @ x_F - b))
        normsDeX1.append(np.linalg.norm(x_F))
    
    # Gradiente descendente para F2(x)
    for _ in range(num_iter):
        grad_F2_value = grad_F2(x_F2, A, b, delta2)
        x_F2 = x_F2 - step * grad_F2_value
        history_F2.append(F2(x_F2, A, b, delta2))
        norm_residual_F2.append(np.linalg.norm(A @ x_F2 - b))
        normsDeX2.append(np.linalg.norm(x_F2))

    
    return x_F, x_F2, history_F, history_F2, error_residual_F, norm_residual_F2,normsDeX1,normsDeX2

def resolver_con_svd(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b 

def main():
    N = 5  # filas
    D = 100  # columnas
    A = np.random.randn(N, D)
    b = np.random.randn(N)
    x = np.random.randn(D)
    num_iter = 1000
    
    x_F, x_F2, history_F, history_F2, error_residual_F, error_residual_F2,normaDeX1,normaDeX2 = gradiente_descendente(x ,A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-2)
    x_svd = resolver_con_svd(A, b)

    # Graficar la evolución del costo
    np.arange(len(history_F))
    plt.figure(figsize=(12, 6))
    plt.semilogy(history_F, label='F1(x)')
    plt.semilogy(history_F2, label='F2(x)')
    plt.hlines(F1(x_svd,A,b), xmin = 0, xmax = num_iter, color = 'r', linestyle = '--', label = 'F(x*)')    
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.title('Evolución del costo durante las iteraciones')
    plt.legend()
    plt.show()
    

    plt.figure(figsize=(12, 6))
    plt.plot(x_F, label='Solución GD F1(x)', linestyle='--')
    plt.plot(x_F2, label='Solución GD F2(x)', linestyle=':')
    plt.plot(x_svd, label='Solución SVD', linestyle='-')
    plt.xlabel('Índice de la variable')
    plt.ylabel('Valor de la variable')
    plt.title('Comparación de soluciones obtenidas')
    plt.legend()
    plt.show()
    
    #error de residuo
    plt.figure(figsize=(12, 6))
    plt.semilogy(error_residual_F, label='Error residual F1')
    plt.semilogy(error_residual_F2, label='Error residual de F2')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error residuo')
    plt.title('Evolución del error de residuo')
    plt.legend()
    plt.show()
    

    # Norma de los vectores x en cada iteracion
    plt.figure(figsize=(12, 6))
    plt.semilogy(normaDeX1, label='Norma de x1')
    plt.semilogy(normaDeX2, label='Norma de x2')
    plt.xlabel('Iteraciones')
    plt.ylabel('Valor de la norma')
    plt.title('Valores de la norma de x durante las iteraciones')
    plt.legend()
    plt.show()

    # Graficar F2(x) con distintos valores de delta
    x_F, x_F2_01, history_F_01, history_F2_01, norm_residual_F_01, error_residual_F2_01,normaDeX1,normaDeX2_01 = gradiente_descendente(x,A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-1)
    x_F, x_F2_001, history_F_001, history_F2_001, norm_residual_F_001, error_residual_F2_001,normaDeX1,normaDeX2_001 = gradiente_descendente(x,A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-2)
    x_F, x_F2_0001, history_F_0001, history_F2_0001, norm_residual_F_0001, error_residual_F2_0001,normaDeX1,normaDeX2_0001 = gradiente_descendente(x,A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-3)
    x_F, x_F2_00001, history_F_00001, history_F2_00001, norm_residual_F_00001, error_residual_F2_00001,normaDeX1,normaDeX2_00001 = gradiente_descendente(x,A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-4)
    x_F, x_F2_000001, history_F_000001, history_F2_000001, norm_residual_F_000001, error_residual_F2_000001,normaDeX1,normaDeX2_000001 = gradiente_descendente(x,A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-5)
    x_F, x_F2_0000001, history_F_0000001, history_F2_0000001, norm_residual_F_0000001, error_residual_F2_0000001,normaDeX1,normaDeX2_0000001 = gradiente_descendente(x,A, b, F1, gradienteF1, F2, gradienteF2, num_iter, 1e-6)


    # como varia F2
    plt.figure(figsize=(12, 6))
    plt.semilogy(history_F2_01, label='delta = 0,1 * maxSigma')    
    plt.semilogy(history_F2_001, label='delta = 0,01 * maxSigma')
    plt.semilogy(history_F2_0001, label='delta = 0,001 * maxSigma')
    plt.semilogy(history_F2_00001, label='delta = 0,0001 * maxSigma')
    plt.semilogy(history_F2_000001, label='delta = 0,00001 * maxSigma')
    plt.semilogy(history_F2_0000001, label='delta = 0,000001 * maxSigma')
    plt.semilogy(history_F, label='F1(x)')      
    plt.xlabel('Iteraciones')
    plt.ylabel('F2(x)')
    plt.title('Evolución de F2(x) durante las iteraciones con distintos valores de delta')
    plt.legend()
    plt.show()


    plt.figure(figsize=(12, 6))
    plt.semilogy(error_residual_F2_01, label='Error residual (F2), delta= 0,1*max_sigma')
    plt.semilogy(error_residual_F2_001, label='Error residual (F2), delta= 0,01*max_sigma')
    plt.semilogy(error_residual_F2_0001, label='Error residual (F2), delta= 0,001*max_sigma')
    plt.semilogy(error_residual_F2_00001, label='Error residual (F2), delta= 0,0001*max_sigma')
    plt.semilogy(error_residual_F2_000001, label='Error residual (F2), delta= 0,00001*max_sigma')
    plt.semilogy(error_residual_F2_0000001, label='Error residual (F2), delta= 0,000001*max_sigma')
    plt.semilogy(error_residual_F, label='Error residual F1')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error residual')
    plt.title('Evolución del error residual durante las iteraciones con distintos valores de delta')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.semilogy(normaDeX2_01, label='Norma de x2, delta= 0,1*max_sigma')
    plt.semilogy(normaDeX2_001, label='Norma de x2, delta= 0,01*max_sigma')
    plt.semilogy(normaDeX2_0001, label='Norma de x2, delta= 0,001*max_sigma')
    plt.semilogy(normaDeX2_00001, label='Norma de x2, delta= 0,0001*max_sigma')
    plt.semilogy(normaDeX2_000001, label='Norma de x2, delta= 0,00001*max_sigma')
    plt.semilogy(normaDeX2_0000001, label='Norma de x2, delta= 0,000001*max_sigma')
    plt.semilogy(normaDeX1, label='Norma de x1')
    plt.xlabel('Iteraciones')
    plt.ylabel('Valor de la norma')
    plt.title('Valores de la norma de x durante las iteraciones con distintos valores de delta')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
