import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Параметры системы
arrival_rate = 7  # Интенсивность поступления (заявки в минуту)
simulation_time = 1000  # Общее время моделирования (в минутах)
max_queue_length = 6  # Максимальная длина очереди


# Функция моделирования СМО
def simulate_smo(num_channels):
    current_time = 0


    queue = []
    service_channels = [None] * num_channels  # Статусы каналов (None, если свободен)
    service_end_times = [0] * num_channels  # Время завершения обслуживания для каждого канала

    # Статистика
    wait_times = []
    queue_lengths = []
    lost_requests = 0

    # Генерация заявок
    arrival_times = []
    time = 0
    while time < simulation_time:
        time += np.random.exponential(1 / arrival_rate)
        arrival_times.append(time)

        # Имитация работы системы
    for arrival_time in arrival_times:
        current_time = arrival_time

        # Освобождение каналов, если обслуживание завершено
        for i in range(num_channels):
            if service_channels[i] and current_time >= service_end_times[i]:
                service_channels[i] = None

                # Попытка найти свободный канал
        channel_found = False
        for i in range(num_channels):
            if service_channels[i] is None:  # Канал свободен
                service_time = gamma.rvs(a=1)
                service_channels[i] = True
                service_end_times[i] = current_time + service_time
                wait_times.append(0)
                channel_found = True
                break

                # Если каналов нет, заявка встаёт в очередь
        if not channel_found:
            if len(queue) < max_queue_length:
                queue.append(current_time)
            else:
                # Заявка теряется, если очередь переполнена
                lost_requests += 1

                # Обслуживание из очереди
        for i in range(num_channels):
            if service_channels[i] is None and queue:
                start_service_time = queue.pop(0)
                # Генерация времени обслуживания из распределения Хи-квадрат с порядком 3
                service_time = gamma.rvs(a=1)
                service_channels[i] = True
                service_end_times[i] = current_time + service_time
                wait_times.append(current_time - start_service_time)  # Время ожидания фиксируется

    # Запись длины очереди
    queue_lengths.append(len(queue))
    # Возврат статистики
    return {
        "avg_wait_time": np.mean(wait_times) if wait_times else 0,
        "avg_queue_length": np.mean(queue_lengths),
        "lost_requests": lost_requests,
        "loss_rate": lost_requests / len(arrival_times) if arrival_times else 0
    }
# Моделирование для различных чисел каналов
results = {}
for num_channels in range(1, 7):
    results[num_channels] = simulate_smo(num_channels)
    print(f"Каналов: {num_channels}")
    print(f"Среднее время ожидания: {results[num_channels]['avg_wait_time']:.2f} минут")
    print(f"Средняя длина очереди: {results[num_channels]['avg_queue_length']:.2f}")
    print(f"Потерянные заявки: {results[num_channels]['lost_requests']}")
    print(f"Доля потерянных заявок: {results[num_channels]['loss_rate']:.2%}\n")
    # Построение графиков
    channels = list(results.keys())
    wait_times = [results[n]["avg_wait_time"] for n in channels]
    loss_rates = [results[n]["loss_rate"] for n in channels]
    plt.figure(figsize=(10, 5))
    # График среднего времени ожидания
    plt.subplot(1, 2, 1)
    plt.plot(channels, wait_times, marker='o')
    plt.title("Среднее время ожидания")
    plt.xlabel("Количество каналов")
    plt.ylabel("Время (минуты)")
    # График доли потерянных заявок
    plt.subplot(1, 2, 2)
    plt.plot(channels, loss_rates, marker='o', color='r')
    plt.title("Доля потерянных заявок")
    plt.xlabel("Количество каналов")
    plt.ylabel("Доля потерянных заявок")
    plt.tight_layout()
    plt.show()