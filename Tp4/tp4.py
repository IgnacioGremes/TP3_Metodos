import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg
from sklearn.decomposition import PCA
import seaborn as sns
np.random.seed(12345)


N = 5
D = 100

A = np.random.rand(N,D)
B = np.random.rand(N)

def gradienteF1(x):
    return 2 * A.T @ ( A @ x - B) 


def gradiente_decendiente(f,iteraciones,condInic):

    xAnt = condInic
    lam = 0
    s = 1 / lam
    for i in range(iteraciones):
        xSig = xAnt 
