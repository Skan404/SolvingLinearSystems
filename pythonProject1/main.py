import numpy as np
import matplotlib.pyplot as plt
from math import sin
import time

# Zadanie A
def create_matrix(N, a1, a2, a3):
    A = np.zeros((N, N))
    for i in range(N):
        A[i][i] = a1
        if i < N - 1:
            A[i][i+1] = a2
            A[i+1][i] = a2
        if i < N - 2:
            A[i][i+2] = a3
            A[i+2][i] = a3
    return A

def create_vector(N, f):
    b = [sin(n * (f + 1)) for n in range(1, N+1)]
    return np.array(b)

# Metoda Jacobiego
def jacobi(A, b, max_iter=1000, tol=1e-9):
    x = np.zeros_like(b)
    res_norms = []
    for it_count in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        res_norm = np.linalg.norm(A @ x_new - b)
        res_norms.append(res_norm)
        if res_norm < tol:
            break
        x = x_new
    return x, it_count+1, res_norms

# Metoda Gaussa-Seidla
def gauss_seidel(A, b, max_iter=1000, tol=1e-9):
    x = np.zeros_like(b)
    res_norms = []
    for it_count in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        res_norm = np.linalg.norm(A @ x_new - b)
        res_norms.append(res_norm)
        if res_norm < tol:
            break
        x = x_new
    return x, it_count+1, res_norms

# Metoda faktoryzacji LU
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1
        for j in range(i+1):
            s1 = sum(U[k][i] * L[j][k] for k in range(j))
            U[j][i] = A[j][i] - s1
        for j in range(i, n):
            s2 = sum(U[k][i] * L[j][k] for k in range(i))
            L[j][i] = (A[j][i] - s2) / U[i][i]
    return L, U

def solve_upper_triangle(U, b):
    n = len(U)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    return x

def solve_lower_triangle(L, b):
    n = len(L)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / L[i][i]
    return x

# Przykładowe wartości
e = 4
c = 3
d = 2
f = 5

N = 932
a1 = 5 + e
a2 = a3 = -1

A = create_matrix(N, a1, a2, a3)
b = create_vector(N, f)

print("Zadanie A:")
print("Macierz A:")
print(A)
print("\nWektor b:")
print(b)

# Zadanie B
start = time.time()
x_jacobi, iterations_jacobi, res_norms_jacobi = jacobi(A, b)
end = time.time()
print("\nZadanie B:")
print(f"Metoda Jacobiego: {iterations_jacobi} iteracji, czas: {end - start} sekund")

start = time.time()
x_gauss_seidel, iterations_gauss_seidel, res_norms_gauss_seidel = gauss_seidel(A, b)
end = time.time()
print(f"Metoda Gaussa-Seidla: {iterations_gauss_seidel} iteracji, czas: {end - start} sekund")

# Wykres normy residuum
plt.figure(figsize=(10, 6))
plt.plot(res_norms_jacobi[:50], label='Jacobi')
plt.plot(res_norms_gauss_seidel[:50], label='Gauss-Seidel')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Residual norm')
plt.legend()
plt.savefig('residuum_B.png')

# Zadanie C
a1 = 3
A = create_matrix(N, a1, a2, a3)

print("\nZadanie C:")
print("Macierz A:")
print(A)

start = time.time()
x_jacobi, iterations_jacobi, res_norms_jacobi = jacobi(A, b)
end = time.time()
print(f"Metoda Jacobiego: {iterations_jacobi} iteracji, czas: {end - start} sekund")

start = time.time()
x_gauss_seidel, iterations_gauss_seidel, res_norms_gauss_seidel = gauss_seidel(A, b)
end = time.time()
print(f"Metoda Gaussa-Seidla: {iterations_gauss_seidel} iteracji, czas: {end - start} sekund")

# Wykres normy residuum
plt.figure(figsize=(10, 6))
plt.plot(res_norms_jacobi[:50], label='Jacobi')
plt.plot(res_norms_gauss_seidel[:50], label='Gauss-Seidel')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Residual norm')
plt.legend()
plt.savefig('residuum_C.png')

# Zadanie D
print("\nZadanie D:")
L, U = lu_decomposition(A)
y = solve_lower_triangle(L, b)
x_lu = solve_upper_triangle(U, y)
res_norm_lu = np.linalg.norm(A @ x_lu - b)
print(f"Metoda LU: Norma residuum: {res_norm_lu}")

a1 = 5 + e
a2 = a3 = -1
A = create_matrix(N, a1, a2, a3)

# Zadanie E
N_values = [100, 500, 1000, 2000]
times_jacobi = []
times_gauss_seidel = []
times_lu = []

for N in N_values:
    A = create_matrix(N, a1, a2, a3)
    b = create_vector(N, f)

    start = time.time()
    x_jacobi, _, _ = jacobi(A, b)
    end = time.time()
    times_jacobi.append(end - start)

    start = time.time()
    x_gauss_seidel, _, _ = gauss_seidel(A, b)
    end = time.time()
    times_gauss_seidel.append(end - start)

    start = time.time()
    L, U = lu_decomposition(A)
    y = solve_lower_triangle(L, b)
    x_lu = solve_upper_triangle(U, y)
    end = time.time()
    times_lu.append(end - start)

print("\nZadanie E:")
plt.figure(figsize=(10, 6))
plt.plot(N_values, times_jacobi, label='Jacobi')
plt.xlabel('N')
plt.ylabel('Time (s)')
plt.legend()
plt.show()
plt.savefig('time_jacobi.png')

plt.figure(figsize=(10, 6))
plt.plot(N_values, times_gauss_seidel, label='Gauss-Seidel')
plt.xlabel('N')
plt.ylabel('Time (s)')
plt.legend()
plt.show()
plt.savefig('time_gauss_seidel.png')

plt.figure(figsize=(10, 6))
plt.plot(N_values, times_lu, label='LU')
plt.xlabel('N')
plt.ylabel('Time (s)')
plt.legend()
plt.show()
plt.savefig('time_lu.png')
