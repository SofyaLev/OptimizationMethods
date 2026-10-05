import numpy as np
import matplotlib.pyplot as plt
import time


# ----- элементы метода Пиявского -----

def g(x, u, W_u, L):
    """ вспомогательная функция g(x, u) = W(u) - L*|x - u| """

    return W_u - L * abs(x - u)


def p(x, u_list, W_list, L):
    """ ломаная p_n(x) = max_i g(x, u_i) """

    return max(g(x, u_list[i], W_list[i], L) for i in range(len(u_list)))


def find_min_p(u_list, W_list, L):
    """ поиск u_n = min p(x) среди нижних вершин ломаной """

    best_u, best_val = None, float('inf')

    # кандидаты - концы отрезка и точки пересечения соседних g_i
    candidates = [u_list[0], u_list[-1]]

    for i in range(len(u_list) - 1):
        ui, ui1 = u_list[i], u_list[i + 1]
        Wi, Wi1 = W_list[i], W_list[i + 1]
        # формула со слайда: u_cross = (W(u_q) - W(u_p)) / (2L) + (u_q + u_p) / 2
        u_cross = (Wi - Wi1) / (2 * L) + (ui + ui1) / 2
        candidates.append(u_cross)

    for u in candidates:
        val = p(u, u_list, W_list, L)
        if val < best_val:
            best_val = val
            best_u = u

    return best_u, best_val

def evaluate_l(W, a, b, n=5000, k=1.2):
    """ оценка константы Липшица """

    xs = np.linspace(a, b, n)               # x_i
    ys = np.array([W(x) for x in xs])       # f(x_i)
    slopes = np.abs(np.diff(ys) / np.diff(xs)) # |f(x_{i+1}) - f(x_i)| / |x_{i+1} - x_i|

    return float(np.max(slopes)) * k        # max(...) * k

# ----- основной алгоритм -----

def pyavsky(W, a, b, eps, L, max_iter=10000):
    """ метод Пиявского для поиска глобального минимума на [a, b] """

    t_start = time.time()

    # нулевая итерация: u_0 = a, u_1 = b
    u_list = [a, b]
    W_list = [W(a), W(b)]

    iterations = 0
    while iterations < max_iter:
        iterations += 1

        # u_n = min p_{n-1}(x), p_val - значение ломаной в этой точке
        u_new, p_val = find_min_p(u_list, W_list, L)
        W_new = W(u_new)

        # добавляем новую точку в отсортированный список
        idx = np.searchsorted(u_list, u_new)
        u_list.insert(idx, u_new)
        W_list.insert(idx, W_new)

        # лекционный критерий: W_new - p_val < eps
        if W_new - p_val < eps:
            break

    elapsed = time.time() - t_start
    best_idx = int(np.argmin(W_list))

    return {
        'u_list': u_list,
        'W_list': W_list,
        'u_min': u_list[best_idx],
        'W_min': W_list[best_idx],
        'iterations': iterations,
        'time': elapsed
    }


# ----- выбор функции -----
# можно выбрать для запуска что-то из набора тестовых функций или ввести свою

def choose_function():
    """ выводит меню тестовых функций и возвращает выбранную """

    print("\nвыберите тестовую функцию:")
    print("-" * 70)
    print(f"{'№':>2}  {'имя':<12} {'W(x)':<35} [a, b]")
    print("-" * 70)

    for i, t in enumerate(TESTS, 1):
        print(f"{i:>2}  {t['name']:<12} {t['expr']:<35} [{t['a']}, {t['b']}]")
    print(f"{len(TESTS) + 1:>2}  своя функция")
    print("-" * 70)

    choice = input("номер: ").strip()

    if not choice.isdigit():
        print("нужно ввести номер, попробуйте ещё раз")
        return choose_function()

    n = int(choice)

    if 1 <= n <= len(TESTS):
        t = TESTS[n - 1]
        W = make_function(t["expr"])
        L = evaluate_l(W, t["a"], t["b"])
        print(f"Оцененная L = {L:.4f}")
        return W, t["a"], t["b"], L, t["name"]

    if n == len(TESTS) + 1:
        # своя функция - вводим выражение и параметры вручную
        W = input_validation()
        a = float(input("a = "))
        b = float(input("b = "))
        if a >= b:
            print("ошибка: a должно быть меньше b")
            return choose_function()
        L = evaluate_l(W, a, b)
        print(f"Оцененная L = {L:.4f}")
        return W, a, b, L, "своя функция"

    print("такого номера нет, попробуйте ещё раз")
    return choose_function()


# ----- функции для форматирования вывода -----

def print_result(name, result):
    """ печатает результатов расчета """

    print("\n" + "=" * 45)
    print("результаты:")
    print(f"  u*       = {result['u_min']:.10f}")
    print(f"  W(u*)    = {result['W_min']:.10f}")
    print(f"  итераций = {result['iterations']}")
    print(f"  время    = {result['time']:.4f} сек")
    print("=" * 45)


def plot_result(W, a, b, result, L, title="метод Пиявского"):
    """ строит график: функция, ломаная, галочки, вершины, минимум """

    fig, ax = plt.subplots(figsize=(12, 6))

    # исходная функция
    x_plot = np.linspace(a, b, 1000)
    y_plot = [W(x) for x in x_plot]
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='W(x)', alpha=0.7)

    # ломаная p_n
    u_list = result['u_list']
    W_list = result['W_list']
    p_plot = [p(x, u_list, W_list, L) for x in x_plot]
    ax.plot(x_plot, p_plot, 'r--', linewidth=1.5, label='p_n(x)', alpha=0.8)

    # вспомогательные функции g_i
    for i, u in enumerate(u_list):
        g_plot = [g(x, u, W_list[i], L) for x in x_plot]
        ax.plot(x_plot, g_plot, 'g-', linewidth=0.5, alpha=0.3)

    # верхние вершины (точки, где вычислялась функция)
    ax.scatter(u_list, W_list, color='green', s=5, alpha=0.5,
               zorder=5, label='верхние вершины u_i')

    # найденный минимум
    ax.scatter([result['u_min']], [result['W_min']], color='red', s=40,
               edgecolors='black', linewidths=0.6, zorder=6,
               label=f'минимум: ({result["u_min"]:.4f}, {result["W_min"]:.4f})')

    # ограничиваем ось Y по значениям функции, а не по галочкам
    y_min = min(result['W_list'])
    y_max = max(result['W_list'])
    margin = (y_max - y_min) * 0.2 + 0.5
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('W(x)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----- данные для тестовых функций -----
# каждая функция задана выражением expr, отрезком [a, b]
# чтобы добавить новую функцию, просто дописываем словарь в список

TESTS = [
    {
        "name": "Растригин",
        "expr": "10 + (x - 1)**2 - 10*cos(2*pi*x)",
        "a": -2.0, "b": 3.0,
    },
    {
        "name": "Экли",
        "expr": "cos(x) * exp(-(x - pi)**2)",
        "a": 0.0, "b": 10.0,
    },
{
        "name": "Из задания",
        "expr": "x + sin(3.14159*x)",
        "a": -2.0, "b": 2.0,
    },
]


# разрешаем только математические функции
SAFE_ENV = {
    "np": np,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": abs, "pi": np.pi, "e": np.e,
}


def make_function(expr):
    """ превращает строку с выражением в функцию W(x) """

    code = compile(expr, "<expr>", "eval")
    def W(x):
        env = dict(SAFE_ENV)
        env["x"] = x
        return eval(code, env)
    return W


def input_validation():
    """ запрашивает выражение и проверяет, что оно вычисляется """

    while True:
        expr = input("f(x) = ").strip()
        try:
            W = make_function(expr)
            W(0.0)
            return W
        except Exception as e:
            print(f"ошибка в выражении: {e}, попробуйте ещё раз")


# ----- main -----

def main():
    """ точка входа: выбор функции, ввод точности, запуск и вывод """

    W, a, b, L, name = choose_function()

    while True:
        try:
            eps = float(input("eps = "))
            if eps <= 0:
                print("eps должно быть положительным, попробуйте ещё раз")
                continue
            break
        except ValueError:
            print("нужно ввести число, попробуйте ещё раз")

    print(f"\nфункция: {name}")
    print(f"отрезок: [{a}, {b}]")
    print(f"L = {L:.4f}, eps = {eps}")

    result = pyavsky(W, a, b, eps, L)

    print_result(name, result)
    plot_result(W, a, b, result, L, title=name)


if __name__ == "__main__":
    main()