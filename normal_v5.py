from easy_v5 import load_data, center_data, covariance_matrix, to_dense
import math
from typing import List


def find_eigenvalues(C: List[List[float]], tol: float = 1e-6) -> List[float]:
    """Находит собственные значения матрицы C методом бисекции."""
    n = len(C)
    eigenvalues = []

    def det_lambda(lambd: float) -> float:
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
    a, step = -abs(trace), 0.1
    while a < abs(trace):
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
        if abs(root) < 1e-6:
            root = 0.0
        if all(abs(root - val) > tol for val in eigenvalues):
            eigenvalues.append(root)
        a += step

    return sorted(eigenvalues, reverse=True)


def compute_kernel(A: List[List[float]], epsilon: float = 1e-6) -> List[List[float]]:
    """Находит базис ядра матрицы A с высокой численной устойчивостью."""
    n = len(A)
    augmented = [row[:] + [0.0] for row in A]

    # Прямой ход с частичным выбором ведущего элемента
    pivot_cols = []
    row_swaps = 0
    for i in range(n):
        max_el, max_row, pivot_col = 0, i, -1
        for j in range(n):
            if j in pivot_cols:
                continue
            for k in range(i, n):
                if abs(augmented[k][j]) > max_el:
                    max_el, max_row, pivot_col = abs(augmented[k][j]), k, j
        # Проверяем, является ли строка почти нулевой
        if max_el < epsilon:
            # Дополнительная проверка суммы элементов строки
            row_sum = max(sum(abs(augmented[k][j]) for j in range(n)) for k in range(i, n))
            if row_sum < epsilon:
                continue
            # Если сумма значима, ищем другой столбец
            for j in range(n):
                if j in pivot_cols:
                    continue
                for k in range(i, n):
                    if abs(augmented[k][j]) > max_el:
                        max_el, max_row, pivot_col = abs(augmented[k][j]), k, j
            if max_el < epsilon:
                continue
        pivot_cols.append(pivot_col)
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
            row_swaps += 1
        pivot = augmented[i][pivot_col]
        for j in range(n + 1):
            augmented[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = augmented[k][pivot_col]
                for j in range(n + 1):
                    augmented[k][j] -= factor * augmented[i][j]

    # Проверка вырожденности
    rank = len(pivot_cols)

    # Формируем базис ядра
    free_vars = [j for j in range(n) if j not in pivot_cols]
    solutions = []
    for var in free_vars:
        vec = [0.0] * n
        vec[var] = 1.0
        for i in range(min(rank, n)):
            pivot_col = pivot_cols[i]
            sum_val = sum(augmented[i][k] * vec[k] for k in range(n) if k != pivot_col)
            vec[pivot_col] = -sum_val / augmented[i][pivot_col] if abs(augmented[i][pivot_col]) >= epsilon else 0.0
        norm = math.sqrt(sum(x ** 2 for x in vec))
        if norm >= epsilon:
            solutions.append([x / norm for x in vec])

    # Если нет решений и ранг < n, добавляем нулевой вектор
    if not solutions and rank < n:
        solutions.append([0.0] * n)

    return solutions


def find_eigenvectors(C: List[List[float]], eigenvalues: List[float], epsilon: float = 1e-6) -> List[List[float]]:
    """Находит ортонормированные собственные векторы матрицы C."""
    n = len(C)
    eigvecs = []

    for idx, lambd in enumerate(eigenvalues):
        A = [[C[i][j] - (lambd if i == j else 0) for j in range(n)] for i in range(n)]
        solutions = compute_kernel(A, epsilon)
        v = None
        for sol in solutions:
            norm = math.sqrt(sum(x ** 2 for x in sol))
            if norm > epsilon:
                v = sol[:]  # Векторы уже нормированы в compute_kernel
                break
        if v is None:
            v = [1.0 if i == idx else 0.0 for i in range(n)]
        eigvecs.append(v)

    # Ортогонализация Грама-Шмидта
    ortho_eigvecs = []
    for idx, v in enumerate(eigvecs):
        u = v[:]
        for prev_u in ortho_eigvecs:
            proj = sum(prev_u[k] * u[k] for k in range(n))
            u = [u[k] - proj * prev_u[k] for k in range(n)]
        norm = math.sqrt(sum(x ** 2 for x in u))
        if norm > epsilon:
            u = [x / norm for x in u]
        else:
            u = [1.0 if i == idx else 0.0 for i in range(n)]
        ortho_eigvecs.append(u)

    return ortho_eigvecs


def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """Вычисляет долю объяснённой дисперсии для k компонент."""
    total = sum(eigenvalues)
    return sum(eigenvalues[:k]) / total if total > 0 and 0 < k <= len(eigenvalues) else 0.0


if __name__ == "__main__":
    X = load_data("int_3x3_2.csv")
    X_c = center_data(X)
    C = covariance_matrix(X_c)
    C_dense = to_dense(C)
    eigenvalues = find_eigenvalues(C_dense)
    print("\nСобственные значения:", [round(x, 4) for x in eigenvalues])
    eigvecs = find_eigenvectors(C_dense, eigenvalues)
    print("\nСобственные векторы:")
    for vec in eigvecs:
        print([round(x, 4) for x in vec])
    print(f"\nДоля объяснённой дисперсии (k=1): {round(explained_variance_ratio(eigenvalues, 1), 4)}")
