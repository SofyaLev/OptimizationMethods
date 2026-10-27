import itertools
from lab4 import (
    INITIAL_ASSETS, INITIAL_CASH, LOTS, FEES, SCENARIOS,
    apply_situation, portfolio_value, bellman, validate_input
)

# названия сценариев по индексам 0, 1, 2
SCEN_NAMES = ("Благопр.", "Нейтр.", "Негатив.")

# начальный капитал = сумма всех активов и свободных денег
INITIAL_CAPITAL = sum(INITIAL_ASSETS) + INITIAL_CASH


# --------------------------------
# Построение дерева решений в файл
# --------------------------------

def print_tree(stage, state, file, depth=0, path_prob=1.0):
    """
    рекурсивно пишет дерево решений найденной стратегии в файл

    в каждом узле - состояние, действие, состояние после сделки
    на каждой ветви - рыночная ситуация и вероятность перехода
    в листьях - конечный капитал, чистый доход и вероятность пути
    """
    indent = "  " * depth

    # лист: этапов больше нет, печатаем итог
    if stage == 3:
        final = portfolio_value(state)
        income = final - INITIAL_CAPITAL
        print(
            f"{indent}[ЛИСТ] W = {final:.2f} д.е., "
            f"доход = {income:+.2f} д.е., "
            f"P пути = {path_prob:.3f}",
            file=file,
        )
        return

    # узел: берем первое оптимальное действие из стратегии
    _, actions = bellman(stage, state)
    u, traded_state = actions[0]

    print(
        f"{indent}Этап {stage + 1}: "
        f"s = ({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}, {state[3]:.2f}) "
        f"| u = {u} "
        f"| s после сделки = "
        f"({traded_state[0]:.2f}, {traded_state[1]:.2f}, "
        f"{traded_state[2]:.2f}, {traded_state[3]:.2f})",
        file=file,
    )

    # ветвимся по трем сценариям текущего этапа
    for idx, (p, mult) in enumerate(SCENARIOS[stage]):
        name = SCEN_NAMES[idx]
        next_state = apply_situation(traded_state, mult)
        print(
            f"{indent}  -> {name} (p = {p:.2f}, "
            f"P пути = {path_prob * p:.3f}): "
            f"s{stage + 2} = "
            f"({next_state[0]:.2f}, {next_state[1]:.2f}, "
            f"{next_state[2]:.2f}, {next_state[3]:.2f})",
            file=file,
        )
        print_tree(stage + 1, next_state, file, depth + 2, path_prob * p)


def save_tree(filename="decision_tree.txt"):
    """ сохраняет полное дерево решений в текстовый файл """
    initial_state = tuple((*INITIAL_ASSETS, INITIAL_CASH))
    with open(filename, "w", encoding="utf-8") as f:
        print_tree(0, initial_state, f)
    print(f"Дерево решений сохранено в {filename}")


# ----------------------------------------
# Анализ всех 27 конечных исходов стратегии
# ----------------------------------------

def run_bonus_analysis():
    initial_state = tuple((*INITIAL_ASSETS, INITIAL_CASH))
    initial_capital = sum(initial_state)

    print("=" * 65)
    print("Полное дерево исходов найденной стратегии")
    print("=" * 65)

    outcomes = []

    # перебираем все 27 возможных сценариев (3 этапа по 3 ситуации)
    # 0 - благоприятная, 1 - нейтральная, 2 - негативная
    for path in itertools.product(range(3), repeat=3):
        state = initial_state
        prob = 1.0
        total_commission = 0.0
        path_desc = []
        actions_desc = []

        # проходим по трем этапам
        for stage in range(3):
            scenario_idx = path[stage]
            p, mult = SCENARIOS[stage][scenario_idx]
            prob *= p

            path_desc.append(f"Эт.{stage + 1}: {SCEN_NAMES[scenario_idx]}")

            # получаем оптимальное действие из нашей стратегии (из bellman)
            _, actions = bellman(stage, state)
            u, traded_state = actions[0]  # берем первое оптимальное

            # считаем комиссии за этот шаг
            step_comm = 0.0
            for i, count in enumerate(u):
                if count != 0:
                    # комиссия берется от абсолютного объема сделки
                    step_comm += abs(count) * LOTS[i] * FEES[i]
            total_commission += step_comm

            actions_desc.append(f"({u[0]},{u[1]},{u[2]})")

            # переходим в следующее состояние
            state = apply_situation(traded_state, mult)

        final_capital = portfolio_value(state)
        net_income = final_capital - initial_capital
        return_pct = (net_income / initial_capital) * 100

        outcomes.append({
            'path': " -> ".join(path_desc),
            'prob': prob,
            'actions': " -> ".join(actions_desc),
            'commission': total_commission,
            'final_capital': final_capital,
            'net_income': net_income,
            'return_pct': return_pct,
        })

    # --- статистика ---
    expected_capital = sum(o['final_capital'] * o['prob'] for o in outcomes)
    expected_income = expected_capital - initial_capital
    expected_return = (expected_income / initial_capital) * 100

    loss_prob = sum(o['prob'] for o in outcomes if o['net_income'] < 0)
    min_income = min(o['net_income'] for o in outcomes)
    max_income = max(o['net_income'] for o in outcomes)

    print(f"Ожидаемый конечный капитал: {expected_capital:.2f} д.е.")
    print(f"Ожидаемый чистый доход:     {expected_income:.2f} д.е.")
    print(f"Ожидаемая доходность:       {expected_return:.2f} %")
    print(f"Вероятность убытка:         {loss_prob * 100:.2f} %")
    print(f"Минимальный доход:          {min_income:.2f} д.е.")
    print(f"Максимальный доход:         {max_income:.2f} д.е.")
    print("-" * 102)

    # --- детализация по всем 27 исходам ---
    print("\nДетализация всех 27 исходов (отсортировано по убыванию капитала):")
    print(
        f"{'Эт1':<3} | {'Эт2':<3} | {'Эт3':<3} | {'Вер.':<5} | "
        f"{'Действия':<32} | {'Комис.':>7} | {'Капитал':>9} | "
        f"{'Доход':>9} | {'Дох-ть':>7}"
    )
    print("-" * 102)

    # сортируем (от лучшего к худшему)
    outcomes.sort(key=lambda x: x['final_capital'], reverse=True)

    # обозначения для сценариев (короткие)
    short_names = {"Благопр.": "Блг", "Нейтр.": "Ней", "Негатив.": "Нег"}

    for o in outcomes:
        # разбиваем сценарий на три части
        parts = o['path'].split(" -> ")
        s1 = short_names[parts[0].split(": ")[1]]
        s2 = short_names[parts[1].split(": ")[1]]
        s3 = short_names[parts[2].split(": ")[1]]

        print(
            f"{s1:<3} | {s2:<3} | {s3:<3} | {o['prob']:.3f} | "
            f"{o['actions']:<22} | {o['commission']:>7.2f} | "
            f"{o['final_capital']:>9.2f} | {o['net_income']:>9.2f} | "
            f"{o['return_pct']:>6.2f}%"
        )

    print("-" * 102)


if __name__ == "__main__":
    validate_input() # проверяем данные из основного файла
    save_tree() # сохраняем полное дерево в decision_tree.txt
    run_bonus_analysis()