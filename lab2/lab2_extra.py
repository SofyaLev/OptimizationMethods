import sympy as sp

from lab2 import TESTS, evaluate_l, make_function, pyavsky


# ----- настройки дополнительного расчета -----

X = sp.Symbol("x", real=True)
EPS = 0.001


# ----- расчет константы Липшица -----

def find_l(expression, a, b):
    """ ищет максимум модуля первой производной на заданном отрезке """

    first = sp.diff(expression, X)
    second = sp.diff(first, X)
    roots = sp.solveset(second, X, domain=sp.Interval(a, b))
    if second == 0 or roots == sp.EmptySet:
        roots = []
    elif not isinstance(roots, sp.FiniteSet):
        raise ValueError("Не удалось получить полный конечный список корней f''")

    candidates = sorted(set([sp.sympify(a), sp.sympify(b)] + list(roots)), key=float)
    values = [abs(first.subs(X, point)) for point in candidates]
    index = max(range(len(values)), key=lambda i: float(values[i]))
    return first, second, candidates, float(values[index])


# ----- проверка на функции из задания -----

def main():
    """ вычисляет константу и запускает исходный метод ломаных """

    test = TESTS[2]
    expression = sp.sympify(test["expr"], locals={"x": X}, rational=True)
    a, b = test["a"], test["b"]
    first, second, candidates, lipschitz = find_l(expression, a, b)
    function = make_function(test["expr"])
    old_l = evaluate_l(function, a, b)
    # небольшой запас учитывает округление при вычислениях с float
    used_l = lipschitz * (1 + 1e-12)
    result = pyavsky(function, a, b, EPS, used_l)

    print(f"Функция: {test['expr']}, отрезок: [{a}, {b}]")
    print(f"f'(x) = {first}")
    print(f"f''(x) = {second}")
    print(f"Кандидаты: {[float(point) for point in candidates]}")
    print(f"Константа Липшица: {lipschitz:.10f}")
    print(f"Прежняя оценка по сетке с запасом 1.2: {old_l:.10f}")
    print(f"L для метода ломаных: {used_l:.10f}")
    print(f"Минимум: x = {result['u_min']:.10f}, f = {result['W_min']:.10f}")
    print(f"Итераций: {result['iterations']}")


if __name__ == "__main__":
    main()
