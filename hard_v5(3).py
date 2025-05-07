from CSR_matrix import SparseMatrix
from easy_v5 import load_data, center_data, covariance_matrix, to_dense, to_sparse
from normal_v5 import find_eigenvalues, find_eigenvectors, explained_variance_ratio
from typing import Tuple
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def pca(X: SparseMatrix, k: int) -> Tuple[SparseMatrix, float]:
    """
    Выполняет PCA на матрице данных X, возвращает проекцию на k компонент и долю дисперсии.

    Вход:
        X: матрица данных (n×m)
        k: число главных компонент
    Выход:
        X_proj: проекция данных (n×k)
        variance_ratio: доля объяснённой дисперсии
    """
    # Центрирование данных
    X_centered = center_data(X)

    # Вычисление матрицы выборочных ковариаций
    C = covariance_matrix(X_centered)
    C_dense = to_dense(C)

    # Нахождение собственных значений и векторов
    eigenvalues = find_eigenvalues(C_dense)
    eigenvectors = find_eigenvectors(C_dense, eigenvalues)

    # Проекция данных на k главных компонент
    n, m = X_centered.num_rows, X_centered.num_cols
    X_dense = to_dense(X_centered)
    V = [eigenvectors[i] for i in range(min(k, len(eigenvectors)))]
    projected = [[sum(X_dense[i][j] * V[l][j] for j in range(m)) for l in range(len(V))] for i in range(n)]
    X_proj = to_sparse(projected, n, len(V))

    # Доля объяснённой дисперсии
    variance_ratio = explained_variance_ratio(eigenvalues, k)

    return X_proj, variance_ratio


def plot_pca_projection(X_proj: SparseMatrix) -> Figure:
    """
    Визуализирует проекцию данных на первые две главные компоненты.

    Вход:
        X_proj: проекция данных (n×2)
    Выход:
        объект Figure из Matplotlib
    """
    if X_proj.num_cols != 2:
        raise ValueError("X_proj must have exactly 2 columns for visualization")

    X_dense = to_dense(X_proj)
    fig, ax = plt.subplots()
    ax.scatter([row[0] for row in X_dense], [row[1] for row in X_dense], c='blue', marker='o')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA Projection')
    ax.grid(True)

    # Вывод графика вместо сохранения
    plt.show()

    return fig


def reconstruction_error(X_orig: SparseMatrix, X_recon: SparseMatrix) -> float:
    """
    Вычисляет среднеквадратическую ошибку восстановления.

    Вход:
        X_orig: исходные данные (n×m)
        X_recon: восстановленные данные (n×m)
    Выход:
        среднеквадратическая ошибка MSE
    """
    if X_orig.num_rows != X_recon.num_rows or X_orig.num_cols != X_recon.num_cols:
        raise ValueError("X_orig and X_recon must have the same dimensions")

    n, m = X_orig.num_rows, X_orig.num_cols
    X_orig_dense = to_dense(X_orig)
    X_recon_dense = to_dense(X_recon)
    mse = sum((X_orig_dense[i][j] - X_recon_dense[i][j]) ** 2 for i in range(n) for j in range(m)) / (n * m)
    return mse


if __name__ == "__main__":
    # Пример использования
    X = load_data("int_3x3_2.csv")
    k = 2
    X_proj, variance_ratio = pca(X, k)

    # Восстановление данных для расчёта MSE
    X_centered = center_data(X)
    C = covariance_matrix(X_centered)
    C_dense = to_dense(C)
    eigenvectors = find_eigenvectors(C_dense, find_eigenvalues(C_dense))
    V = [eigenvectors[i] for i in range(k)]
    X_proj_dense = to_dense(X_proj)
    X_recon_dense = [[sum(X_proj_dense[i][l] * V[l][j] for l in range(k)) for j in range(X.num_cols)] for i in
                     range(X.num_rows)]
    X_recon = to_sparse(X_recon_dense, X.num_rows, X.num_cols)

    # Вычисление MSE
    mse = reconstruction_error(X_centered, X_recon)

    # Вывод результатов
    print(f"Доля объяснённой дисперсии (k={k}): {round(variance_ratio, 4)}")
    print("Спроецированные данные (k=2):")
    for row in to_dense(X_proj):
        print([round(x, 4) for x in row])
    print(f"Среднеквадратическая ошибка восстановления: {round(mse, 4)}")

    # Визуализация
    plot_pca_projection(X_proj)
