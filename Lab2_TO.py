def simplex_method(c, A, b):
    """
    Реализация симплекс-метода для задачи линейного программирования.
    Максимизация f(x) = c^T * x при ограничениях Ax <= b, x >= 0.

    Аргументы:
        c: Список коэффициентов целевой функции.
        A: Список списков (матрица ограничений).
        b: Список правых частей ограничений.

    Возвращает:
        Оптимальное значение функции и значения переменных.
    """
    import numpy as np

    # Размерности
    num_constraints, num_variables = len(A), len(c)

    # Построение начальной симплекс-таблицы
    # Добавляем базисные переменные (для каждого ограничения)
    tableau = np.zeros((num_constraints + 1, num_constraints + num_variables + 1))

    # Заполняем таблицу
    tableau[:num_constraints, :num_variables] = A  # Коэффициенты ограничений
    tableau[:num_constraints, num_variables:num_variables + num_constraints] = np.eye(num_constraints)  # Базис
    tableau[:num_constraints, -1] = b  # Правая часть ограничений
    tableau[-1, :num_variables] = -np.array(c)  # Целевая функция
    print(tableau)

    # Симплекс-метод
    while True:
        # Шаг 1: Проверка на оптимальность (нет отрицательных значений в строке целевой функции)
        if all(tableau[-1, :-1] >= 0):
            break

        # Шаг 2: Выбор разрешающего столбца (самый отрицательный коэффициент)
        pivot_col = np.argmin(tableau[-1, :-1])

        # Шаг 3: Выбор разрешающей строки (критерий минимального отношения)
        ratios = []
        for i in range(num_constraints):
            if tableau[i, pivot_col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(float('inf'))  # Исключаем строки, где элемент <= 0
        pivot_row = np.argmin(ratios)

        if all(r == float('inf') for r in ratios):
            raise ValueError("Задача не имеет конечного решения (неограниченная функция).")

        # Шаг 4: Обновление таблицы по разрешающему элементу
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(num_constraints + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        print(tableau)

    # Извлечение решения
    solution = np.zeros(num_variables)
    for i in range(num_constraints):
        for j in range(num_variables):
            if tableau[i, j] == 1 and tableau[:, j].tolist().count(1) == 1 and tableau[:, j].tolist().count(0) == num_constraints:
                solution[j] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    return optimal_value, solution


# Пример использования
c = [2, -1, 1, -5]  # Коэффициенты целевой функции
A = [
    [1, 2, 0, 1],  # Ограничение 1
    [-1, 4, 1, 0],  # Ограничение 2
]
b = [5, 3]  # Правая часть ограничений

optimal_value, solution = simplex_method(c, A, b)
print("Максимальное значение функции:", optimal_value)
print("Оптимальные значения переменных:", solution)

