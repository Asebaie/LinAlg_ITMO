class SparseMatrix:
    def __init__(self, file_path=None, values=None, columns=None, row_ptr=None, num_rows=None, num_cols=None):
        if file_path:
            # Считывание из файла
            self.values = []  # Ненулевые значения.
            self.columns = []  # Индексы столбцов ненулевых элементов.
            self.row_ptr = [0]  # Указатели на начало каждой строки.
            self.num_rows = 0  # Количество строк.
            self.num_cols = 0  # Количество столбцов.

            with open(file_path, "r") as file:
                for line in file:
                    row = list(map(int, line.strip().split(";")))
                    self.num_cols = max(self.num_cols, len(row))
                    for col_index, value in enumerate(row):
                        if value != 0:
                            self.values.append(value)
                            self.columns.append(col_index)
                    self.row_ptr.append(len(self.values))
                    self.num_rows += 1
        else:
            # Создание вручную
            self.values = values or []
            self.columns = columns or []
            self.row_ptr = row_ptr or []
            self.num_rows = num_rows or 0
            self.num_cols = num_cols or 0

    def get_element(self, row, col):
        # Выводим элемент по заданной строке и столбцу
        start = self.row_ptr[row]
        end = self.row_ptr[row + 1]
        for i in range(start, end):
            if self.columns[i] == col:
                return self.values[i]
        return 0

    def trace(self):
        # Вычисляет след матрицы.
        print("\nЗадача №1:")
        if self.num_rows != self.num_cols:
            print("\033[31mСлед не определён: матрица не квадратная.\033[0m")
            return

        trace_sum = 0
        for i in range(self.num_rows):
            trace_sum += self.get_element(i, i)
        print(f"\033[32mСлед матрицы: {trace_sum}\033[0m")

    def add(self, other_file_path):
        # Сложение двух матриц. Выводит результат в плотном формате.
        print("\nЗадача №2:")
        other_matrix = SparseMatrix(other_file_path)
        if self.num_rows != other_matrix.num_rows or self.num_cols != other_matrix.num_cols:
            print("\033[31mНевозможно сложить: размеры матриц не совпадают.\033[0m")
            return

        print("\033[32mРезультат сложения матриц:\033[0m")
        for row in range(self.num_rows):
            dense_row = [0] * self.num_cols
            # Обрабатываем текущую строку первой матрицы
            for i in range(self.row_ptr[row], self.row_ptr[row + 1]):
                dense_row[self.columns[i]] += self.values[i]
            # Обрабатываем текущую строку второй матрицы
            for i in range(other_matrix.row_ptr[row], other_matrix.row_ptr[row + 1]):
                dense_row[other_matrix.columns[i]] += other_matrix.values[i]
            # Выводим строку
            print(" ".join(map(str, dense_row)))

    def multiply_scalar(self, scalar):
        # Умножение матрицы на скаляр. Выводит результат в плотном формате.
        print(f"\033[32mРезультат умножения матрицы на скаляр {scalar}:\033[0m")
        for row in range(self.num_rows):
            dense_row = [0] * self.num_cols
            for i in range(self.row_ptr[row], self.row_ptr[row + 1]):
                dense_row[self.columns[i]] = self.values[i] * scalar
            print(" ".join(map(str, dense_row)))

    def multiply_matrix(self, other_file_path):
        # Умножение двух матриц. Выводит результат в плотном формате.
        other_matrix = SparseMatrix(other_file_path)
        if self.num_cols != other_matrix.num_rows:
            print("\033[31mНевозможно перемножить: размеры матриц не подходят.\033[0m")
            return

        print("\033[32m\nРезультат умножения матриц:\033[0m")
        other_dense = [[0] * other_matrix.num_cols for _ in range(other_matrix.num_rows)]
        for row in range(other_matrix.num_rows):
            start = other_matrix.row_ptr[row]
            end = other_matrix.row_ptr[row + 1]
            for i in range(start, end):
                other_dense[row][other_matrix.columns[i]] = other_matrix.values[i]

        for row in range(self.num_rows):
            dense_row = [0] * other_matrix.num_cols
            for i in range(self.row_ptr[row], self.row_ptr[row + 1]):
                self_value = self.values[i]
                self_col = self.columns[i]
                for col in range(other_matrix.num_cols):
                    dense_row[col] += self_value * other_dense[self_col][col]
            print(" ".join(map(str, dense_row)))

    def determinant_and_inverse_check(self):
        # Вычисляет определитель матрицы и проверяет существование обратной матрицы.
        print("\nЗадача №3:")
        if self.num_rows != self.num_cols:
            print("\033[31mОпределитель не определён: матрица не квадратная.\033[0m")
            return

        def determinant(csr_matrix):
            # Рекурсивная функция для вычисления определителя в CSR-формате.
            if csr_matrix.num_rows == 1:
                return csr_matrix.get_element(0, 0)

            if csr_matrix.num_rows == 2:
                a = csr_matrix.get_element(0, 0)
                b = csr_matrix.get_element(0, 1)
                c = csr_matrix.get_element(1, 0)
                d = csr_matrix.get_element(1, 1)
                return a * d - b * c

            det = 0
            for col in range(csr_matrix.num_cols):
                element = csr_matrix.get_element(0, col)
                if element != 0:  # Учитываем только ненулевые элементы
                    minor = minor_matrix(csr_matrix, 0, col)
                    minor_det = determinant(minor)
                    det += ((-1) ** col) * element * minor_det
            return det

        def minor_matrix(csr_matrix, excluded_row, excluded_col):
            # Создаёт минор матрицы в CSR-формате.
            minor_values = []
            minor_columns = []
            minor_row_ptr = [0]
            current_index = 0

            for row in range(csr_matrix.num_rows):
                if row == excluded_row:
                    continue  # Пропускаем исключённую строку
                start = csr_matrix.row_ptr[row]
                end = csr_matrix.row_ptr[row + 1]
                minor_row_values = []

                for i in range(start, end):
                    col = csr_matrix.columns[i]
                    if col != excluded_col:
                        minor_row_values.append((col if col < excluded_col else col - 1, csr_matrix.values[i]))

                # Добавляем значения текущей строки минора
                for col, value in minor_row_values:
                    minor_values.append(value)
                    minor_columns.append(col)
                current_index += len(minor_row_values)
                minor_row_ptr.append(current_index)

            # Создаём минор как новую разреженную матрицу
            return SparseMatrix(
                values=minor_values,
                columns=minor_columns,
                row_ptr=minor_row_ptr,
                num_rows=csr_matrix.num_rows - 1,
                num_cols=csr_matrix.num_cols - 1,
            )

        # Вычисляем определитель основной матрицы
        det = determinant(self)
        print(f"\033[32mОпределитель матрицы: {det}\033[0m")
        if det != 0:
            print("\033[32mСуществует ли обратная матрица: да\033[0m")
        else:
            print("\033[31mСуществует ли обратная матрица: нет\033[0m")


if __name__ == "__main__":
    file1 = "int_3x3.csv"
    file2 = "int_3x3_1.csv"

    sparse_matrix = SparseMatrix(file1)

    sparse_matrix.trace()

    row, col = map(int, input("Введите координаты элемента через пробел: ").split())
    row -= 1
    col -= 1
    if 0 <= row < sparse_matrix.num_rows and 0 <= col < sparse_matrix.num_cols:
        element = sparse_matrix.get_element(row, col)
        print(f"\033[32mЭлемент в строке {row + 1}, столбце {col + 1}: {element}\033[0m")
    else:
        print("\033[31mОшибка: Координаты выходят за пределы матрицы.\033[0m")

    sparse_matrix.add(file2)

    scalar = float(input("\nВведите скаляр: "))
    sparse_matrix.multiply_scalar(scalar)

    sparse_matrix.multiply_matrix(file2)

    sparse_matrix.determinant_and_inverse_check()
