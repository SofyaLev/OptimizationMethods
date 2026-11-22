import numpy as np

from components import _prepare


# ----- параметры гибрида (вариант 14) -----

DEFAULTS = {
    "swarm_size": 30,
    "inertia": 0.7,
    "cognitive": 1.5,
    "social": 1.5,
    "velocity_fraction": 0.2,
    "temperature_scale": 0.1,
    "temperature_ratio": 1e-8,
    "step_start": 0.05,
    "step_end": 1e-5,
}


# ----- учет вычислений и истории -----

class Budget:
    """ хранит бюджет, лучший результат и данные для рисунков """

    def __init__(self, f, bounds, max_nfe):
        """ проверяет исходные данные и создает счетчик """

        self.lower, self.upper = _prepare(bounds, max_nfe)
        self.f = f
        self.max_nfe = max_nfe
        self.nfe = 0
        self.best_x = None
        self.best_f = np.inf
        self.history = []
        self.trajectory = []
        self.switches = []
        self.snapshots = []

    @property
    def left(self):
        """ возвращает оставшееся число вычислений """

        return self.max_nfe - self.nfe

    def eval(self, point):
        """ вычисляет f в допустимой точке и обновляет общий рекорд """

        if self.left <= 0:
            raise RuntimeError("Бюджет вычислений исчерпан")
        point = np.clip(np.asarray(point, dtype=float), self.lower, self.upper)
        self.nfe += 1
        value = float(self.f(point.copy()))
        if not np.isfinite(value):
            raise ValueError("Целевая функция должна возвращать конечное число")
        if value < self.best_f:
            self.best_x = point.copy()
            self.best_f = value
            self.trajectory.append((self.nfe, point.copy()))
        self.history.append(self.best_f)
        return value

    def snap(self, name, points):
        """ сохраняет положения частиц независимо от дальнейшего движения """

        self.snapshots.append((self.nfe, name, points.copy()))

    def result(self):
        """ возвращает результат в формате задания преподавателя """

        return {
            "x": self.best_x.copy(),
            "f": self.best_f,
            "nfe": self.nfe,
            "history": self.history,
            "trajectory": self.trajectory,
            "switches": self.switches,
            "snapshots": self.snapshots,
        }


# ----- точка входа для запуска и битвы -----

def optimize(f, bounds, max_nfe=10_000, seed=None, constraints=None, **params):
    """
    ищет минимум операторным гибридом PSO и SA для варианта 14

    f - функция двух переменных, доступная только через вычисление значения
    bounds - две пары нижних и верхних границ
    max_nfe - бюджет, включая создание роя и кандидатов в лидеры
    seed - зерно генератора случайных чисел
    constraints - для этого варианта допускается None или пустой список
    params - параметры из DEFAULTS, обычно используются значения по умолчанию

    после обновления роя проверяет одного соседа текущего лидера по Метрополису
    общий рекорд хранится отдельно от лидера и возвращается после поиска
    """

    if constraints:
        raise ValueError("Вариант 14 поддерживает только прямоугольные границы")
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Неизвестные параметры: {sorted(unknown)}")
    settings = {**DEFAULTS, **params}
    size = settings["swarm_size"]
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Размер роя должен быть положительным целым числом")
    coefficients = np.array([settings[name] for name in DEFAULTS if name != "swarm_size"])
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0):
        raise ValueError("Параметры должны быть конечными и неотрицательными")
    if not 0 < settings["temperature_ratio"] <= 1:
        raise ValueError("Отношение температур должно находиться в (0, 1]")
    if settings["temperature_scale"] <= 0 or min(settings["step_start"], settings["step_end"]) <= 0:
        raise ValueError("Температурный масштаб и шаги должны быть положительными")

    budget = Budget(f, bounds, max_nfe)
    rng = np.random.default_rng(seed)
    count = min(size, max_nfe)
    width = budget.upper - budget.lower
    limit = settings["velocity_fraction"] * width
    positions = rng.uniform(budget.lower, budget.upper, size=(count, 2))
    velocities = rng.uniform(-limit, limit, size=(count, 2))
    values = np.array([budget.eval(point) for point in positions])
    personal_positions = positions.copy()
    personal_values = values.copy()
    leader = budget.best_x.copy()
    leader_value = budget.best_f
    budget.snap("Начальный рой", positions)

    # начальная температура зависит от разброса f, а не от известного минимума
    spread = float(np.ptp(values))
    scale_floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
    temperature_start = settings["temperature_scale"] * max(spread, scale_floor)
    remaining_at_start = budget.left
    snapshot_levels = [count + remaining_at_start / 3, count + 2 * remaining_at_start / 3]
    level_index = 0
    accepted = 0
    uphill = 0
    attempts = 0
    swarm_nfe = count
    annealing_nfe = 0
    budget.switches.append((count, "Начало операторного PSO + SA"))

    while budget.left:
        uphill_this_iteration = False
        # все частицы движутся к одному лидеру, который может уступать рекорду
        memory = settings["cognitive"] * rng.random((count, 2)) * (personal_positions - positions)
        attraction = settings["social"] * rng.random((count, 2)) * (leader - positions)
        velocities = np.clip(settings["inertia"] * velocities + memory + attraction, -limit, limit)
        proposed = positions + velocities
        velocities[(proposed < budget.lower) | (proposed > budget.upper)] = 0.0
        positions = np.clip(proposed, budget.lower, budget.upper)
        active = min(count, budget.left)
        for index in range(active):
            value = budget.eval(positions[index])
            swarm_nfe += 1
            if value < personal_values[index]:
                personal_values[index] = value
                personal_positions[index] = positions[index]
            if value < leader_value:
                leader = positions[index].copy()
                leader_value = value

        if budget.left:
            # один шаг отжига меняет ориентир следующего движения роя
            progress = (budget.nfe - count) / max(1, remaining_at_start)
            temperature = temperature_start * settings["temperature_ratio"] ** progress
            step = settings["step_start"] * (settings["step_end"] / settings["step_start"]) ** progress
            candidate = np.clip(leader + rng.normal(size=2) * step * width,
                                budget.lower, budget.upper)
            candidate_value = budget.eval(candidate)
            annealing_nfe += 1
            attempts += 1
            difference = candidate_value - leader_value
            if difference <= 0 or rng.random() < np.exp(-difference / temperature):
                leader, leader_value = candidate.copy(), candidate_value
                accepted += 1
                if difference > 0:
                    uphill += 1
                    uphill_this_iteration = True
                    budget.switches.append((budget.nfe, "SA принял ухудшение лидера"))

        if level_index == 0 and (uphill_this_iteration or budget.nfe >= snapshot_levels[0]):
            name = "Первая ухудшающая замена лидера" if uphill_this_iteration else "Рой, 1/3 бюджета"
            budget.snap(name, positions)
            level_index = 1
        elif level_index == 1 and budget.nfe >= snapshot_levels[1]:
            budget.snap("Рой, 2/3 бюджета", positions)
            level_index = 2

    budget.snap("Конечный рой", positions)
    result = budget.result()
    result["diagnostics"] = {
        "sa_attempts": attempts,
        "sa_accepted": accepted,
        "sa_uphill": uphill,
        "pso_nfe": swarm_nfe,
        "sa_nfe": annealing_nfe,
        "temperature_start": temperature_start,
    }
    return result
