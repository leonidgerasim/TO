import math
import numpy as np
import pandas as pd

# Исходные данные
lambda_ = 5  # Интенсивность входного потока
mu = 10  # Интенсивность обслуживания
m = 6  # Максимальная длина очереди
a = 2  # Параметр стоимости за канал
b = 2.5  # Параметр стоимости за отказ

# Диапазон числа каналов
n_range = range(1, 7)

# Список для хранения результатов
results = []

for n in n_range:
    rho = lambda_ / (n * mu)  # Уровень загрузки

    if rho >= 1:
        results.append([n] + ["Ошибка: уровень загрузки должен быть меньше 1."])
        continue

        # Расчёт вероятностей и характеристик
    P0 = 1 / (
            sum((lambda_ / mu) ** k / math.factorial(k) for k in range(n)) +
            ((lambda_ / mu) ** n / math.factorial(n)) * (1 - rho ** (m + 1)) / (1 - rho)
    )

    P_reject = ((lambda_ / mu) ** n / math.factorial(n)) * rho ** m * P0  # Вероятность отказа
    A = lambda_ * (1 - P_reject)  # Абсолютная пропускная способность

    P_queue = 1 - sum((lambda_ / mu) ** k / math.factorial(k) * P0 for k in range(n))

    L_queue = P_queue * rho / (1 - rho) if rho < 1 else np.inf  # Средняя длина очереди
    W_queue = L_queue / lambda_  # Среднее время ожидания в очереди
    L_system = L_queue + n * rho  # Среднее число заявок в системе
    W_system = L_system / lambda_  # Среднее время пребывания заявки в системе
    W_service = 1 / mu  # Среднее время обслуживания

    L_busy = n * rho  # Среднее число занятых каналов
    C_busy = L_busy / n  # Коэффициент занятости каналов
    C_idle = 1 - C_busy  # Коэффициент простоя каналов

    cost = a * n + b * P_reject  # Издержки

    # Добавление результатов
    results.append([
        n, rho, P0, P_reject, A, P_queue, L_busy, C_busy, C_idle, L_queue,
        W_queue, W_service, W_system, cost
    ])

# Создание DataFrame
columns = [
    "Channels (n)", "Load Level (\u03C1)", "P0", "P_reject", "Throughput (A)", "P_queue",
    "Avg Busy Channels", "Channel Utilization", "Channel Idle", "Avg Queue Length",
    "Avg Queue Wait Time", "Avg Service Time", "Avg System Time", "Costs"
]
df = pd.DataFrame(results, columns=columns)

# Сохранение результатов в Excel
output_file = "results.csv"
df.to_csv(output_file, index=False)

print(f"Расчёт выполнен. Результаты сохранены в \"{output_file}\".")