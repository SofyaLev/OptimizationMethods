from itertools import product

import sympy as sp


# ----- исходная задача (вариант 11) -----

C = (1, 2, 4, 1)
A = sp.Matrix([[1, 1, 1, 0], [0, 1, 2, 1], [1, 0, 0, 1]])
B = sp.Matrix([10, 6, 2])


# ----- целочисленное решение -----

def solve_integer():
    """ перебирает все допустимые целочисленные точки своего варианта """

    best_value = None
    best_points = []
    feasible_count = 0
    # из ограничений следуют x1 <= 10, x2 <= 6, x3 <= 3
    for x1, x2, x3 in product(range(11), range(7), range(4)):
        # четвертую переменную однозначно определяет ограничение-равенство
        x4 = 6 - x2 - 2 * x3
        if x4 < 0 or x1 + x2 + x3 > 10 or x1 + x4 < 2:
            continue
        point = (x1, x2, x3, x4)
        feasible_count += 1
        value = sum(c * x for c, x in zip(C, point))
        if best_value is None or value > best_value:
            best_value = value
            best_points = [point]
        elif value == best_value:
            best_points.append(point)
    return best_value, best_points, feasible_count


# ----- двойственный сертификат -----

def find_dual(point):
    """
    строит кандидата двойственной задачи по условиям дополняющей нежесткости

    point - найденная целочисленная точка
    после решения системы отдельно проверяет знаки, неравенства и равенство целей
    эти проверки подтверждают оптимальность для непрерывной задачи данного варианта
    """

    variables = sp.symbols("y1 y2 y3", real=True)
    dual = sp.Matrix(variables)
    primal = sp.Matrix(point)
    primal_residuals = A * primal - B
    dual_residuals = A.T * dual - sp.Matrix(C)
    equations = [variables[i] * primal_residuals[i] for i in range(3)]
    equations += [point[j] * dual_residuals[j] for j in range(4)]
    solutions = sp.solve(equations, variables, dict=True)
    candidates = []
    for solution in solutions:
        if not all(variable in solution for variable in variables):
            continue
        values = dual.subs(solution)
        if values[0] < 0 or values[2] > 0:
            continue
        if any(value < 0 for value in A.T * values - sp.Matrix(C)):
            continue
        primal_value = sum(c * x for c, x in zip(C, point))
        dual_value = (B.T * values)[0]
        if sp.simplify(dual_value - primal_value) == 0:
            candidates.append((tuple(values), dual_value))
    return candidates


# ----- вывод результата -----

def main():
    """ решает целочисленную задачу и проверяет двойственный сертификат """

    value, points, count = solve_integer()
    if not points:
        raise RuntimeError("Допустимые целочисленные точки не найдены")
    certificates = find_dual(points[0])
    if not certificates:
        raise RuntimeError("Сертификат непрерывного оптимума не найден")
    print(f"Допустимых целочисленных точек: {count}")
    print(f"Оптимальные целочисленные точки: {points}")
    print(f"Максимум целочисленной задачи: {value}")
    for dual, dual_value in certificates:
        print(f"Двойственная точка: {dual}")
        print(f"Минимум двойственной задачи: {dual_value}")
    print("Прямая и двойственная цели совпали, оптимальность подтверждена")


if __name__ == "__main__":
    main()
