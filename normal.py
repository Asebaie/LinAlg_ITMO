from matrix_v3 import SparseMatrix

def subtract_lambda_identity(X, lambd):
    """
    Вычитает λ * I из матрицы X, возвращает обычную плотную матрицу (list of lists)
    """
    result = []
    for i in range(X.num_rows):
        row = []
        for j in range(X.num_cols):
            value = X.get_element(i, j)
            if i == j:
                value -= lambd
            row.append(value)
        result.append(row)
    return result

def determinant(matrix):
    """
    Определитель квадратной матрицы (list of lists), без библиотеки copy
    """
    n = len(matrix)
    A = [row[:] for row in matrix]  # глубокая копия вручную
    det = 1.0
    for i in range(n):
        if abs(A[i][i]) < 1e-12:
            for k in range(i + 1, n):
                if abs(A[k][i]) > 1e-12:
                    A[i], A[k] = A[k], A[i]
                    det *= -1
                    break
            else:
                return 0.0
        det *= A[i][i]
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
    return det

def find_eigenvalues(X, tol=1e-6):
    """
    Поиск собственных значений методом бисекции через SparseMatrix
    """
    n = X.num_cols
    eigenvalues = []

    def f(lambd):
        A = subtract_lambda_identity(X, lambd)
        return determinant(A)

    def bisect(a, b):
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            return None
        while b - a > tol:
            m = (a + b) / 2
            fm = f(m)
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
        return round((a + b) / 2, 6)

    a = -1000
    step = 1.0
    while a < 1000:
        b = a + step
        root = bisect(a, b)
        if root is not None and all(abs(root - val) > tol for val in eigenvalues):
            eigenvalues.append(root)
        a += step

    eigenvalues.sort(reverse=True)
    return eigenvalues

if __name__ == "__main__":
    X = SparseMatrix("int_3x3.csv")
    eigenvalues = find_eigenvalues(X)
    print("Собственные значения:")
    for val in eigenvalues:
        print(val)