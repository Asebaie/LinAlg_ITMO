from CSR_matrix import SparseMatrix
from typing import List

def load_data(file_path: str) -> SparseMatrix:
    """Загружает данные из CSV в SparseMatrix."""
    return SparseMatrix(file_path)

def to_dense(matrix: SparseMatrix) -> List[List[float]]:
    """Преобразует SparseMatrix в плотную матрицу."""
    return [[float(matrix.get_element(i, j)) for j in range(matrix.num_cols)]
            for i in range(matrix.num_rows)]

def to_sparse(dense: List[List[float]], num_rows: int, num_cols: int) -> SparseMatrix:
    """Преобразует плотную матрицу в SparseMatrix."""
    values, columns, row_ptr = [], [], [0]
    for i in range(num_rows):
        for j in range(num_cols):
            if abs(dense[i][j]) > 1e-16:
                values.append(float(dense[i][j]))
                columns.append(j)
        row_ptr.append(len(values))
    return SparseMatrix(values=values, columns=columns, row_ptr=row_ptr,
                       num_rows=num_rows, num_cols=num_cols)

def gauss_solver(A: SparseMatrix, b: SparseMatrix, epsilon: float = 1e-16) -> List[SparseMatrix]:
    """Решает Ax = b методом Гаусса с частичным выбором ведущего элемента."""
    n = A.num_rows
    A_dense = to_dense(A)
    b_dense = [[float(b.get_element(i, 0))] for i in range(n)]
    augmented = [A_dense[i] + [b_dense[i][0]] for i in range(n)]

    # Прямой ход
    pivot_cols = []
    for i in range(n):
        max_el, max_row, pivot_col = 0, i, -1
        for j in range(n):
            if j in pivot_cols:
                continue
            for k in range(i, n):
                if abs(augmented[k][j]) > max_el:
                    max_el, max_row, pivot_col = abs(augmented[k][j]), k, j
        # Проверка на почти нулевую строку
        if max_el < 1e-10:
            continue
        pivot_cols.append(pivot_col)
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        pivot = augmented[i][pivot_col]
        for j in range(n + 1):
            augmented[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = augmented[k][pivot_col]
                for j in range(n + 1):
                    augmented[k][j] -= factor * augmented[i][j]

    # Проверка ранга
    rank = len(pivot_cols)
    print("\nМатрица после прямого хода:")
    for row in augmented:
        print([round(x, 4) for x in row])
    print("Ведущие столбцы:", pivot_cols)
    print("Ранг матрицы:", rank)

    # Проверка на совместность
    for i in range(n):
        if all(abs(augmented[i][j]) < epsilon for j in range(n)) and abs(augmented[i][n]) >= epsilon:
            raise ValueError("Система несовместна")

    # Базис ядра
    free_vars = [j for j in range(n) if j not in pivot_cols]
    print("Свободные переменные:", free_vars)
    solutions = []
    for var in free_vars:
        vec = [0.0] * n
        vec[var] = 1.0
        for i in range(n):
            for j in pivot_cols:
                if abs(augmented[i][j]) >= epsilon:
                    sum_val = sum(augmented[i][k] * vec[k] for k in range(n) if k != j)
                    vec[j] = -sum_val / augmented[i][j] if abs(augmented[i][j]) >= epsilon else 0.0
        if any(abs(x) >= epsilon for x in vec):
            solutions.append([[x] for x in vec])

    if not solutions and rank < n:
        solutions.append([[0.0] for _ in range(n)])

    return [to_sparse(sol, n, 1) for sol in solutions]

def center_data(X: SparseMatrix) -> SparseMatrix:
    """Центрирует данные, вычитая среднее по каждому признаку."""
    n, m = X.num_rows, X.num_cols
    if n == 0 or m == 0:
        return SparseMatrix(values=[], columns=[], row_ptr=[0], num_rows=n, num_cols=m)

    means, counts = [0.0] * m, [0] * m
    for i in range(n):
        for j in range(X.row_ptr[i], X.row_ptr[i + 1]):
            col = X.columns[j]
            means[col] += X.values[j]
            counts[col] += 1
    means = [means[j] / counts[j] if counts[j] > 0 else 0.0 for j in range(m)]

    values, columns, row_ptr = [], [], [0]
    for i in range(n):
        row_vals, row_cols = [], []
        for j in range(X.row_ptr[i], X.row_ptr[i + 1]):
            col = X.columns[j]
            val = X.values[j] - means[col]
            if abs(val) > 1e-16:
                row_vals.append(val)
                row_cols.append(col)
        values.extend(row_vals)
        columns.extend(row_cols)
        row_ptr.append(len(values))
    return SparseMatrix(values=values, columns=columns, row_ptr=row_ptr, num_rows=n, num_cols=m)

def covariance_matrix(X_centered: SparseMatrix) -> SparseMatrix:
    """Вычисляет матрицу ковариаций."""
    n, m = X_centered.num_rows, X_centered.num_cols
    if n <= 1 or m == 0:
        return SparseMatrix(values=[], columns=[], row_ptr=[0] * (m + 1), num_rows=m, num_cols=m)

    X_dense = to_dense(X_centered)
    Xt = [[X_dense[i][j] for i in range(n)] for j in range(m)]
    C = [[sum(Xt[i][k] * X_dense[k][j] for k in range(n)) / (n - 1) for j in range(m)] for i in range(m)]
    return to_sparse(C, m, m)

if __name__ == "__main__":
    X = load_data("int_3x3_2.csv")
    X_centered = center_data(X)
    C = covariance_matrix(X_centered)
    print("\nИсходная матрица:")
    for row in to_dense(X):
        print([round(x, 4) for x in row])
    print("\nЦентрированная матрица:")
    for row in to_dense(X_centered):
        print([round(x, 4) for x in row])
    print("\nМатрица ковариаций:")
    for row in to_dense(C):
        print([round(x, 4) for x in row])
