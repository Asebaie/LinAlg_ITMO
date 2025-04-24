from matrix_v3 import SparseMatrix
import math


def load_data(file_path):
    X = SparseMatrix(file_path)
    return [[X.get_element(i, j) for j in range(X.num_cols)] for i in range(X.num_rows)]


def center_data(X):
    rows, cols = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(rows)) / rows for j in range(cols)]
    return [[X[i][j] - means[j] for j in range(cols)] for i in range(rows)]


def covariance_matrix(X_centered):
    n = len(X_centered)
    Xt = [[X_centered[i][j] for i in range(n)] for j in range(len(X_centered[0]))]
    C = [[sum(Xt[i][k] * X_centered[k][j] for k in range(n)) for j in range(len(Xt[0]))] for i in range(len(Xt))]
    return [[C[i][j] / (n - 1) for j in range(len(C[0]))] for i in range(len(C))]


def gauss_solver(A, b):
    n = len(A)
    aug = [[A[i][j] for j in range(n)] + [b[i][0]] for i in range(n)]
    for i in range(n):
        max_el, max_row = max((abs(aug[k][i]), k) for k in range(i, n))
        if max_el < 1e-10:
            raise ValueError("Система несовместна")
        if max_row != i:
            aug[i], aug[max_row] = aug[max_row], aug[i]
        pivot = aug[i][i]
        for j in range(i, n + 1):
            aug[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]
    sol = [[aug[i][n]] for i in range(n)]
    if all(abs(x[0]) < 1e-10 for x in sol):
        for j in range(n - 1, -1, -1):
            if any(abs(aug[i][j]) > 1e-10 for i in range(n)):
                return [[0 if k != j else 1] for k in range(n)]
    return sol


def find_eigenvalues(C, tol=1e-6):
    n = len(C)
    eigenvalues = []

    def det_lambda(lambd):
        A = [[C[i][j] - (lambd if i == j else 0) for j in range(n)] for i in range(n)]
        det = 1.0
        for i in range(n):
            max_el, max_row = max((abs(A[k][i]), k) for k in range(i, n))
            if max_el < 1e-12:
                return 0.0
            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                det *= -1
            det *= A[i][i]
            for j in range(i + 1, n):
                factor = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]
        return det

    trace = sum(C[i][i] for i in range(n))
    a, step = 0, 0.1
    while a < trace + step:
        b = a + step
        fa, fb = det_lambda(a), det_lambda(b)
        if fa * fb > 0:
            a += step
            continue
        while b - a > tol:
            m = (a + b) / 2
            fm = det_lambda(m)
            if abs(fm) < tol:
                break
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        root = (a + b) / 2
        if all(abs(root - val) > tol for val in eigenvalues):
            eigenvalues.append(root)
        a += step

    eigenvalues.sort(reverse=True)
    return eigenvalues


def find_eigenvectors(C, eigenvalues):
    eigvecs = []
    for lambd in eigenvalues:
        A = [[C[i][j] - (lambd if i == j else 0) for j in range(len(C))] for i in range(len(C))]
        v = gauss_solver(A, [[0] for _ in range(len(C))])
        norm = math.sqrt(sum(v[i][0] ** 2 for i in range(len(v))))
        v = [[v[i][0] / norm if norm > 1e-10 else (1 if i == 0 else 0)] for i in range(len(v))]
        eigvecs.append(v)
    if len(eigvecs) == 2:
        v1, v2 = eigvecs
        dot = sum(v1[i][0] * v2[i][0] for i in range(len(v1)))
        if abs(dot) > 1e-6:
            proj = [[dot * v1[i][0]] for i in range(len(v1))]
            v2 = [[v2[i][0] - proj[i][0]] for i in range(len(v2))]
            norm = math.sqrt(sum(v2[i][0] ** 2 for i in range(len(v2))))
            v2 = [[v2[i][0] / norm if norm > 1e-10 else (-v1[1][0] if i == 0 else v1[0][0])] for i in range(len(v2))]
        eigvecs[1] = v2
    return eigvecs


def explained_variance_ratio(eigenvalues, k):
    total = sum(eigenvalues)
    return sum(eigenvalues[:k]) / total if total > 0 else 0.0


if __name__ == "__main__":
    X = load_data("int_3x3.csv")
    X_c = center_data(X)
    C = covariance_matrix(X_c)
    print("\nМатрица ковариаций:")
    for row in C:
        print([round(x, 4) for x in row])
    eigvals = find_eigenvalues(C)
    print("\nСобственные значения:", [round(x, 4) for x in eigvals])
    eigvecs = find_eigenvectors(C, eigvals)
    print("\nСобственные векторы:")
    for vec in eigvecs:
        print([round(vec[i][0], 4) for i in range(len(vec))])
    print(f"\nДоля объяснённой дисперсии (k=1): {round(explained_variance_ratio(eigvals, 1), 4)}")