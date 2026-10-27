import tracemalloc
from functools import lru_cache
from itertools import product
from math import floor
from time import perf_counter

# ----------------------
# Исходные данные задачи
# ----------------------

# начальные стоимости активов: ЦБ1, ЦБ2, депозит (в д.е.)
INITIAL_ASSETS = (100.0, 800.0, 400.0)

# свободные денежные средства на старте (в д.е.)
INITIAL_CASH = 600.0

# минимальные остатки, которые должны оставаться у инвестора
MIN_ASSETS = (30.0, 150.0, 100.0)

# размер одного пакета = 1/4 начальной стоимости актива
LOTS = (25.0, 200.0, 100.0)

# комиссии брокера при покупке и при продаже
FEES = (0.04, 0.07, 0.05)

# константа для сравнения дробных чисел
EPS = 1e-8

# флаг отсечения заведомо невыгодных сделок на втором этапе
# True - быстро (использует линейность V3)
# False - полный перебор (медленнее)
USE_PRUNING = True

# сценарии по этапам
# формат: ((вероятность, (множитель ЦБ1, ЦБ2, депозита)), ...)
SCENARIOS = (
    # этап 1
    ((0.60, (1.20, 1.10, 1.07)),   # благоприятная
     (0.30, (1.05, 1.02, 1.03)),   # нейтральная
     (0.10, (0.80, 0.95, 1.00))),  # негативная
    # этап 2
    ((0.30, (1.40, 1.15, 1.01)),
     (0.20, (1.05, 1.00, 1.00)),
     (0.50, (0.60, 0.90, 1.00))),
    # этап 3
    ((0.40, (1.15, 1.12, 1.05)),
     (0.40, (1.05, 1.01, 1.01)),
     (0.20, (0.70, 0.94, 1.00))),
)


# ------------------------
# Проверка исходных данных
# ------------------------

def validate_input():
    """ проверяет корректность исходных данных """
    # все параметры по активам должны быть заданы тройками
    if not (len(INITIAL_ASSETS) == len(MIN_ASSETS) == len(LOTS) == len(FEES) == 3):
        raise ValueError("Для трех активов должны быть заданы три параметра")

    # начальные суммы не могут быть отрицательными
    if INITIAL_CASH < 0 or any(x < 0 for x in INITIAL_ASSETS):
        raise ValueError("Начальные суммы не могут быть отрицательными")

    # размер пакета должен быть положительным
    if any(q <= 0 for q in LOTS):
        raise ValueError("Размер пакета должен быть положительным")

    # комиссия - доля от суммы, значит должна лежать в [0, 1)
    if any(not 0 <= fee < 1 for fee in FEES):
        raise ValueError("Комиссии должны находиться в диапазоне [0, 1)")

    # минимальные остатки не могут быть отрицательными
    if any(mn < 0 for mn in MIN_ASSETS):
        raise ValueError("Минимальные объемы не могут быть отрицательными")

    # в задаче ровно три этапа
    if len(SCENARIOS) != 3:
        raise ValueError("Модель должна содержать три этапа")

    # по каждому этапу: сумма вероятностей = 1, множители > 0
    for scenarios in SCENARIOS:
        if abs(sum(p for p, _ in scenarios) - 1.0) > EPS:
            raise ValueError("Сумма вероятностей ситуаций этапа должна равняться 1")
        for p, mult in scenarios:
            if p < 0 or len(mult) != 3 or any(x <= 0 for x in mult):
                raise ValueError("Некорректная вероятность или множитель")


# ----------------------------------------
# Генерация допустимых управлений на этапе
# ----------------------------------------

def make_actions(state, profitable_factors=None):
    """
    возвращает список допустимых управлений для данного состояния

    состояние state = (ЦБ1, ЦБ2, депозит, деньги)
    управление u = (u_1, u_2, u_3) - сколько пакетов купить/продать по каждому
    активу:
        u[i] > 0 — покупка,
        u[i] < 0 — продажа,
        u[i] = 0 — бездействие.

    profitable_factors - необязательный кортеж ожидаемых множителей
    оставшихся этапов. если он задан, отсекаем заведомо невыгодные
    направления сделок

    обоснование отсечения для нашей задачи:
    на этапе 2 остается только этап 3, а V_3(s) линейна по состоянию
    (V_3 = 1.02*A_1 + 1.04*A_2 + 1.024*D + C), поэтому вклад
    сделки по каждому активу не зависит от сделок по другим активам.
    значит, отсечение (покупать/продавать/бездействовать)
    по каждому активу эквивалентно полному перебору. для других задач
    такое отсечение может быть неверным - там нужен полный перебор
    (profitable_factors=None)

    возвращает список пар (новое_состояние_после_сделок, управление)
    """
    assets, cash = state[:3], state[3]

    # сколько максимум пакетов можно продать по каждому активу,
    # чтобы не уйти ниже минимума
    max_sell = [
        max(0, floor((assets[i] - MIN_ASSETS[i] + EPS) / LOTS[i]))
        for i in range(3)
    ]

    # верхняя оценка денег, которые можно выручить, если продать все,
    # что разрешено. нужна, чтобы понять, сколько пакетов в принципе
    # можно купить в этом же управлении
    sale_funds_upper_bound = sum(
        max_sell[i] * LOTS[i] * (1.0 - FEES[i]) for i in range(3)
    )

    # формируем диапазон допустимых значений u[i] для каждого актива
    ranges = []
    for i in range(3):
        # максимум пакетов на покупку: сколько позволяют деньги
        # (свои + максимальная выручка от продаж) с учетом комиссии
        max_buy = max(
            0,
            floor((cash + sale_funds_upper_bound + EPS)
                  / (LOTS[i] * (1.0 + FEES[i])))
        )
        low, high = -max_sell[i], max_buy

        # если известны ожидаемые множители оставшихся этапов,
        # отсекаем заведомо невыгодные направления
        if profitable_factors is not None:
            factor = profitable_factors[i]
            if factor > 1.0 + FEES[i] + EPS:
                # актив в среднем растет сильнее, чем стоит покупка ->
                # продавать невыгодно, оставляем только покупки и 0
                low = 0
            elif factor < 1.0 - FEES[i] - EPS:
                # актив в среднем падает сильнее, чем теряем на комиссии
                # при продаже -> покупать невыгодно, оставляем продажи и 0
                high = 0
            elif (abs(factor - (1.0 + FEES[i])) <= EPS or
                  abs(factor - (1.0 - FEES[i])) <= EPS):
                # ровно на пороге комиссии - покупка/продажа могут быть
                # равноценны бездействию, поэтому ничего не отсекаем
                pass
            else:
                # множитель между порогами -> любая сделка ухудшает ожидание
                low = high = 0
        ranges.append(range(low, high + 1))

    # перебираем все комбинации u = (u_1, u_2, u_3) из построенных диапазонов
    actions = []
    for u in product(*ranges):
        # новые стоимости активов после сделок
        new_assets = tuple(assets[i] + u[i] * LOTS[i] for i in range(3))
        new_cash = cash

        # пересчитываем деньги с учетом комиссий
        for i, count in enumerate(u):
            amount = count * LOTS[i]
            if count > 0:
                # покупка: платим стоимость пакетов + комиссию
                new_cash -= amount * (1.0 + FEES[i])
            elif count < 0:
                # продажа: получаем стоимость пакетов - комиссию
                new_cash += (-amount) * (1.0 - FEES[i])

        # запрет кредита: денег не может стать отрицательное кол-во
        if new_cash < -EPS:
            continue

        # минимум проверяем при продаже
        if any(u[i] < 0 and new_assets[i] < MIN_ASSETS[i] - EPS for i in range(3)):
            continue

        actions.append((
            tuple(round(x, 8) for x in (*new_assets, new_cash)),
            tuple(u),
        ))
    return actions


# -----------------------------------
# Переход состояния и целевая функция
# -----------------------------------

def apply_situation(state, multipliers):
    """
    применяет сценарий к состоянию: умножает стоимости активов на множители
    свободные деньги не меняются
    """
    return tuple(round(x, 8) for x in (
        state[0] * multipliers[0],
        state[1] * multipliers[1],
        state[2] * multipliers[2],
        state[3],
    ))


def portfolio_value(state):
    """ итоговая стоимость портфеля W(s) = сумма всех активов и денег """
    return sum(state)


def expected_multipliers(stage):
    """
    математическое ожидание множителя каждого актива на данном этапе
    E[m_i] = Σ_j p_j * m_ij
    """
    return tuple(
        sum(p * mult[i] for p, mult in SCENARIOS[stage])
        for i in range(3)
    )


# ---------------------------------------------
# Функция Беллмана для последнего этапа (k = 3)
# ---------------------------------------------

def terminal_stage_value(state):
    """
    оптимум последнего этапа (k = 3)

    на последнем этапе будущего уже нет, поэтому сравниваем
    ожидаемую стоимость сразу после сценария. если ожидаемый множитель
    каждого актива лежит строго между ценой покупки с комиссией
    (1 + c) и чистой выручкой от продажи (1 - c), то любая сделка
    ухудшает ожидание -> наиболее выгодно - ничего не делать

    возвращает (значение, кортеж пар (управление, состояние_после_сделки))
    все пары - это альтернативные оптимальные управления (если есть)
    """
    means = expected_multipliers(2)

    # проверяем, что бездействие строго оптимально
    no_trade_is_optimal = all(
        (means[i] < 1.0 + FEES[i] - EPS) and
        (means[i] > 1.0 - FEES[i] + EPS)
        for i in range(3)
    )
    if no_trade_is_optimal:
        # ожидаемая стоимость без сделок - просто линейная комбинация
        expected = sum(state[i] * means[i] for i in range(3)) + state[3]
        return expected, (((0, 0, 0), tuple(state)),)

    # иначе (например, если комиссии нулевые или множитель на пороге) -
    # перебираем все действия и собираем все оптимумы
    best_value = float("-inf")
    best_actions = []
    for traded_state, u in make_actions(state):
        value = sum(
            p * portfolio_value(apply_situation(traded_state, mult))
            for p, mult in SCENARIOS[2]
        )
        if value > best_value + EPS:
            best_value, best_actions = value, [(u, traded_state)]
        elif abs(value - best_value) <= EPS:
            best_actions.append((u, traded_state))
    return best_value, tuple(best_actions)


# ---------------------------------
# Функция Беллмана для этапов 1 и 2
# ---------------------------------

@lru_cache(maxsize=None)
def bellman(stage, state):
    """
    функция Беллмана

    возвращает:
        best_value - максимальная ожидаемая конечная стоимость портфеля
        best_actions - кортеж пар (управление, состояние_после_сделки), на которых достигается максимум

    lru_cache сохраняет уже посчитанные подзадачи (state, stage),
    чтобы не пересчитывать их повторно
    """
    # на последнем этапе используется отдельная (более быстрая) функция
    if stage == 2:
        return terminal_stage_value(state)

    best_value = float("-inf")
    best_actions = []

    # на этапе 2 можно отсечь заведомо невыгодные сделки (см. make_actions),
    # так как после него остается только этап 3, где делать что-либо невыгодно -> продолжение линейно по объемам
    # если USE_PRUNING = False - перебираем все допустимые управления
    if stage == 1 and USE_PRUNING:
        factors = tuple(
            expected_multipliers(1)[i] * expected_multipliers(2)[i]
            for i in range(3)
        )
        candidates = make_actions(state, profitable_factors=factors)
    else:
        # на первом этапе или при выключенном отсечении перебираем все допустимые действия
        candidates = make_actions(state)

    # перебираем допустимые управления текущего этапа
    for traded_state, u in candidates:
        # ожидаемое значение = Σ_j p_j * V_{k+1}(f(s, u, j))
        value = 0.0
        for p, mult in SCENARIOS[stage]:
            next_state = apply_situation(traded_state, mult)
            future, _ = bellman(stage + 1, next_state)
            value += p * future

        # обновляем максимум (с учетом EPS, чтобы корректно ловить равенства)
        if value > best_value + EPS:
            best_value, best_actions = value, [(u, traded_state)]
        elif abs(value - best_value) <= EPS:
            best_actions.append((u, traded_state))

    if not best_actions:
        raise ValueError(f"Нет допустимых управлений на этапе {stage + 1}: {state}")
    return best_value, tuple(best_actions)


# --------------------------------------------
# Печать одной оптимальной стратегии по дереву
# --------------------------------------------

def print_policy_table():
    """
    рекурсивно собирает и печатает сводную таблицу стратегии для всех достижимых состояний
    """
    print(f"{'Этап':<6} | {'Состояние перед этапом (A_1, A_2, D, C)':<40} | {'Действие u':<15} | {'Состояние после сделки':<30}")
    print("-" * 105)

    initial = tuple((*INITIAL_ASSETS, INITIAL_CASH))

    def walk(stage, state):
        if stage == 3:
            return

        _, actions = bellman(stage, state)
        u, traded_state = actions[0]  # берем первое из возможных оптимальных

        # форматируем состояния для вывода
        state_str = f"({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}, {state[3]:.2f})"
        traded_str = f"({traded_state[0]:.2f}, {traded_state[1]:.2f}, {traded_state[2]:.2f}, {traded_state[3]:.2f})"

        print(f"Этап {stage + 1} | {state_str:<40} | {str(u):<15} | {traded_str:<30}")

        # рекурсивно идем в следующие ветви по всем сценариям
        for p, mult in SCENARIOS[stage]:
            next_state = apply_situation(traded_state, mult)
            walk(stage + 1, next_state)

    walk(0, initial)
    print("-" * 105)


# -------------------------------------
# Замер времени и памяти
# -------------------------------------

def measure_resources():
    """
    повторно прогоняет задачу с чистым кэшем и замеряет время и пиковую память
    возвращает (значение, время_в_секундах, пик_памяти_в_МБ)
    """
    bellman.cache_clear()
    initial = tuple((*INITIAL_ASSETS, INITIAL_CASH))

    tracemalloc.start()
    start = perf_counter()
    value, _ = bellman(0, initial)
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return value, elapsed, peak / (1024 * 1024)


# ----
# main
# ----

def main():
    validate_input()

    initial = tuple((*INITIAL_ASSETS, INITIAL_CASH))
    initial_total = portfolio_value(initial)

    # чистый прогон DP с замером времени и памяти
    value, dp_time, peak_mb = measure_resources()

    _, optimal_actions = bellman(0, initial)

    print("=" * 65)
    print("Результаты")
    print("=" * 65)
    print(f"Начальная стоимость портфеля: {initial_total:.2f} д.е.")
    print(f"Оптимальное управление 1-го этапа: {optimal_actions[0][0]}")
    print(f"Число найденных оптимумов на 1-м этапе: {len(optimal_actions)}")
    print(f"Максимальная ожидаемая стоимость: {value:.6f} д.е.")
    print(f"Максимальный ожидаемый доход:   {value - initial_total:.6f} д.е.")
    print("=" * 65)

    print("\nТаблица оптимальной политики:")
    print_policy_table()

    # промежуточные значения V_k для сверки с ручным расчетом
    print("\nПромежуточные значения V_k:")
    for s2 in [(810.0, 880.0, 428.0, 2.0),
               (708.75, 816.0, 412.0, 2.0),
               (540.0, 760.0, 400.0, 2.0)]:
        v2, _ = bellman(1, s2)
        print(f"  V2{s2} = {v2:.6f}")

    for s3 in [(49.0, 1012.0, 432.28, 746.0),
               (36.75, 880.0, 428.0, 746.0),
               (21.0, 792.0, 428.0, 746.0)]:
        v3, _ = terminal_stage_value(s3)
        print(f"  V3{s3} = {v3:.6f}")

    print("\n" + "=" * 65)
    print(f"Время работы:   {dp_time:.4f} с")
    print(f"Пиковая память:     {peak_mb:.2f} МБ")
    print(f"Состояний в кэше:   {bellman.cache_info().currsize}")
    print("=" * 65)


if __name__ == "__main__":
    main()