from matrix_v3 import SparseMatrix

def center_data(X):
    """
    Центрирует данные: вычитает среднее по каждому столбцу.
    Формула: X_centered = X - mean(X)
    """
    means = [sum(X.get_element(i, j) for i in range(X.num_rows)) / X.num_rows
             for j in range(X.num_cols)]
    centered = SparseMatrix()
    centered.num_rows = X.num_rows
    centered.num_cols = X.num_cols
    centered.values = []
    centered.columns = []
    centered.row_ptr = [0]
    for i in range(X.num_rows):
        for j in range(X.num_cols):
            val = X.get_element(i, j) - means[j]
            if val != 0:
                centered.values.append(val)
                centered.columns.append(j)
        centered.row_ptr.append(len(centered.values))
    return centered

def covariance_matrix(X_centered):
    """
    Вычисляет матрицу ковариаций.
    Формула: C = (1/(n-1)) * X_centered^T * X_centered
    """
    n = X_centered.num_rows
    C = [[0.0 for _ in range(X_centered.num_cols)] for _ in range(X_centered.num_cols)]
    for i in range(X_centered.num_cols):
        for j in range(X_centered.num_cols):
            sum_val = 0.0
            for k in range(X_centered.num_rows):
                sum_val += X_centered.get_element(k, i) * X_centered.get_element(k, j)
            C[i][j] = sum_val / (n - 1)
    return C

if __name__ == "__main__":
    X = SparseMatrix("int_3x3.csv")
    X_centered = center_data(X)
    C = covariance_matrix(X_centered)
    print("Матрица ковариаций:")
    for row in C:
        print(row)