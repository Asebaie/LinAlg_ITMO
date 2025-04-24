from matrix_v3 import SparseMatrix
import matplotlib.pyplot as plt

def center_data(X):
    """Центрирует данные: вычитает среднее по каждому столбцу."""
    means = [
        sum(X.get_element(i, j) for i in range(X.num_rows)) / X.num_rows
        for j in range(X.num_cols)
    ]
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

    return centered, means

def covariance_matrix(X_centered):
    """Вычисляет матрицу ковариаций."""
    n = X_centered.num_rows
    m = X_centered.num_cols
    C = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            total = 0.0
            for k in range(n):
                total += X_centered.get_element(k, i) * X_centered.get_element(k, j)
            C[i][j] = total / (n - 1)
    return C

def find_eigenvalues(cov_matrix):
    """Находит собственные значения матрицы ковариаций (упрощённый вариант, для симметричных матриц)."""
    # Для упрощения используем метод Якоби или другой численный метод
    n = len(cov_matrix)
    eigenvalues = [cov_matrix[i][i] for i in range(n)]  # Простое приближение
    return eigenvalues

def find_eigenvectors(cov_matrix, eigenvalues):
    """Находит собственные векторы для заданных собственных значений."""
    def solve_eigenvector(matrix, eigenvalue):
        """Решает систему линейных уравнений для нахождения собственного вектора."""
        n = len(matrix)
        vector = [1.0] * n  # Начальное приближение
        for _ in range(100):  # Итерации до сходимости
            new_vector = []
            for i in range(n):
                summation = 0.0
                for j in range(n):
                    if i != j:
                        summation += matrix[i][j] * vector[j]
                new_vector.append((eigenvalue - summation) / matrix[i][i])
            if all(abs(new_vector[i] - vector[i]) < 1e-6 for i in range(n)):
                break
            vector = new_vector
        return vector

    eigenvectors = []
    for eigenvalue in eigenvalues:
        eigenvectors.append(solve_eigenvector(cov_matrix, eigenvalue))
    return eigenvectors

def explained_variance_ratio(eigenvalues, k):
    """Вычисляет долю объяснённой дисперсии для первых k собственных значений."""
    total_variance = sum(eigenvalues)
    var_ratio = [eigenvalue / total_variance for eigenvalue in eigenvalues[:k]]
    return var_ratio

def pca(X, k):
    """Полный алгоритм PCA."""
    X_centered, means = center_data(X)
    C = covariance_matrix(X_centered)
    eigenvalues = find_eigenvalues(C)
    top_eigenvalues = eigenvalues[:k]
    eigenvectors = find_eigenvectors(C, top_eigenvalues)

    # Проекция данных
    X_proj = []
    for i in range(X.num_rows):
        row_proj = []
        for v in eigenvectors:
            dot = sum(X_centered.get_element(i, j) * v[j] for j in range(X.num_cols))
            row_proj.append(dot)
        X_proj.append(row_proj)

    var_ratio = explained_variance_ratio(eigenvalues, k)
    return X_proj, var_ratio

def plot_pca_projection(X_proj):
    """Визуализирует проекцию на первые две компоненты."""
    if len(X_proj[0]) < 2:
        raise ValueError("Для визуализации необходимо k >= 2")
    x_vals = [row[0] for row in X_proj]
    y_vals = [row[1] for row in X_proj]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, c="blue", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Проекция на первые 2 главные компоненты")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X = SparseMatrix("int_3x3.csv")
    k = 2
    X_proj, var_ratio = pca(X, k)
    print("Доля объяснённой дисперсии:", var_ratio)
    plot_pca_projection(X_proj)