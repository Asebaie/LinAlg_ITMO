from matrix_v3 import SparseMatrix

def load_data(file_path):
    X = SparseMatrix(file_path)
    return [[X.get_element(i, j) for j in range(X.num_cols)] for i in range(X.num_rows)]

def gauss_solver(A, b):
    n = len(A)
    aug = [[A[i][j] for j in range(n)] + [b[i][0]] for i in range(n)]
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-10:
            raise ValueError("Система несовместна или имеет бесконечное число решений")
        for j in range(n + 1):
            aug[i][j] = aug[i][j] / pivot
        for r in range(n):
            if r != i:
                factor = aug[r][i]
                for j in range(n + 1):
                    aug[r][j] -= factor * aug[i][j]
    sol = [[aug[i][n]] for i in range(n)]
    if all(x[0] == 0 for x in sol):
        sol = [[1 if j == i else 0] for j in range(n)]
    return sol

def center_data(X):
    rows, cols = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(rows)) / rows for j in range(cols)]
    X_c = [[X[i][j] - means[j] for j in range(cols)] for i in range(rows)]
    return X_c

def transpose_matrix(X):
    rows, cols = len(X), len(X[0])
    return [[X[i][j] for i in range(rows)] for j in range(cols)]

def multiply_matrices(X, Y):
    rows_X, cols_X = len(X), len(X[0])
    rows_Y, cols_Y = len(Y), len(Y[0])
    return [[sum(X[i][k] * Y[k][j] for k in range(cols_X)) for j in range(cols_Y)] for i in range(rows_X)]

def covariance_matrix(X_centered):
    n = len(X_centered)
    Xt = transpose_matrix(X_centered)
    C = multiply_matrices(Xt, X_centered)
    return [[C[i][j] / (n - 1) for j in range(len(C[0]))] for i in range(len(C))]

if __name__ == "__main__":
    X = load_data("int_3x3.csv")
    X_centered = center_data(X)
    C = covariance_matrix(X_centered)
    print("\nИсходная матрица:")
    for row in X:
        print(row)
    print("\nЦентрированная матрица:")
    for row in X_centered:
        print(row)
    print("\nМатрица ковариаций:")
    for row in C:
        print(row)