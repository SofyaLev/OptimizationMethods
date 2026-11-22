import numpy as np


# ----- проверка исходных данных -----

def _prepare(bounds, max_nfe):
    """ проверяет область поиска и бюджет вычислений """

    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("Границы должны быть конечным массивом размера (2, 2)")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Нижняя граница должна быть меньше верхней")
    if not isinstance(max_nfe, (int, np.integer)) or max_nfe < 1:
        raise ValueError("Бюджет должен быть положительным целым числом")
    return bounds[:, 0], bounds[:, 1]


# ----- метод роя частиц -----

def particle_swarm(f, bounds, max_nfe=10_000, seed=None, *,
                   swarm_size=30, inertia=0.7, cognitive=1.5,
                   social=1.5, velocity_fraction=0.2):
    """
    ищет минимум функции f методом роя частиц

    bounds - границы двух координат
    max_nfe - максимальное число вычислений f, включая начальный рой
    seed - зерно генератора для воспроизводимости
    inertia - постоянный коэффициент инерции w
    cognitive, social - притяжение к личному рекорду и лидеру роя
    velocity_fraction - максимальная скорость как доля ширины области

    возвращает словарь с лучшей точкой x, значением f и числом вычислений nfe
    """

    lower, upper = _prepare(bounds, max_nfe)
    if not isinstance(swarm_size, (int, np.integer)) or swarm_size < 1:
        raise ValueError("Размер роя должен быть положительным целым числом")
    parameters = np.array([inertia, cognitive, social, velocity_fraction])
    if not np.all(np.isfinite(parameters)) or np.any(parameters < 0):
        raise ValueError("Параметры PSO должны быть конечными и неотрицательными")
    rng = np.random.default_rng(seed)
    # если бюджет меньше размера роя, оцениваем только допустимое число частиц
    count = min(swarm_size, max_nfe)
    limit = velocity_fraction * (upper - lower)
    positions = rng.uniform(lower, upper, size=(count, 2))
    velocities = rng.uniform(-limit, limit, size=(count, 2))
    values = np.array([float(f(point.copy())) for point in positions])
    nfe = count
    personal_positions = positions.copy()
    personal_values = values.copy()
    best_index = int(np.argmin(values))
    best_position = positions[best_index].copy()
    best_value = float(values[best_index])

    while nfe < max_nfe:
        # скорость складывается из инерции, личной памяти и притяжения к лидеру
        memory = cognitive * rng.random((count, 2)) * (personal_positions - positions)
        attraction = social * rng.random((count, 2)) * (best_position - positions)
        velocities = np.clip(inertia * velocities + memory + attraction, -limit, limit)
        proposed = positions + velocities
        # возвращаем частицу на границу и обнуляем скорость по этой координате
        velocities[(proposed < lower) | (proposed > upper)] = 0.0
        positions = np.clip(proposed, lower, upper)
        active = min(count, max_nfe - nfe)
        # последняя итерация может быть неполной, чтобы не превысить бюджет
        for index in range(active):
            value = float(f(positions[index].copy()))
            nfe += 1
            if value < personal_values[index]:
                personal_values[index] = value
                personal_positions[index] = positions[index]
            if value < best_value:
                best_value = value
                best_position = positions[index].copy()

    return {"x": best_position, "f": best_value, "nfe": nfe}


# ----- имитация отжига -----

def simulated_annealing(f, bounds, max_nfe=10_000, seed=None, *,
                        temperature_start=0.5, temperature_end=1e-6,
                        step_start=0.1, step_end=1e-5):
    """
    ищет минимум функции f методом имитации отжига

    temperature_start, temperature_end - начальная и конечная температуры
    step_start, step_end - начальный и конечный шаги как доли ширины области
    температура и шаг геометрически уменьшаются по мере расходования бюджета

    возвращает лучшую посещенную точку, ее значение и число вычислений
    """

    lower, upper = _prepare(bounds, max_nfe)
    parameters = np.array([temperature_start, temperature_end, step_start, step_end])
    if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0):
        raise ValueError("Температуры и относительные шаги должны быть положительными")
    rng = np.random.default_rng(seed)
    position = rng.uniform(lower, upper)
    value = float(f(position.copy()))
    best_position, best_value = position.copy(), value
    for nfe in range(1, max_nfe):
        # доля израсходованного бюджета задает температуру и масштаб шага
        progress = (nfe - 1) / max(1, max_nfe - 2)
        temperature = temperature_start * (temperature_end / temperature_start) ** progress
        step = step_start * (step_end / step_start) ** progress
        candidate = np.clip(position + rng.normal(size=2) * step * (upper - lower),
                            lower, upper)
        candidate_value = float(f(candidate.copy()))
        difference = candidate_value - value
        # улучшение принимаем всегда, ухудшение - по критерию Метрополиса
        if difference <= 0 or rng.random() < np.exp(-difference / temperature):
            position, value = candidate, candidate_value
        if candidate_value < best_value:
            # сохраняем рекорд отдельно от текущей точки, которая может ухудшаться
            best_position, best_value = candidate.copy(), candidate_value
    return {"x": best_position, "f": best_value, "nfe": max_nfe}
