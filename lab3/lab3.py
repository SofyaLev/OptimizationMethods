import sympy as sp
from itertools import combinations
from typing import Any


# ----- настройки задачи (вариант 14) -----

# исходные переменные
x, y, z = sp.symbols("x y z", real=True)
variables = (x, y, z)

# целевая функция
f = x + 2 * y - z

# ограничения в виде h_j(x) = 0
h = (x**2 + 2 * y**2 + z**2 - 4,)

# точность проверки невязок
EPS = 1e-8

# для варианта 14:
#   условие x^2 + 2*y^2 + z^2 = 4 задает эллипсоид - замкнутое и ограниченное
#   множество в R^3, то есть компакт
#   градиент ограничения не обращается в ноль на эллипсоиде, значит особых
#   точек нет, список кандидатов полный
compact = True
full_list = True


# ----- проверка входных данных -----

def check_input() -> None:
    """ проверяет, что задача задана корректно """

    if not variables:
        raise ValueError("нет переменных")
    if len(set(variables)) != len(variables):
        raise ValueError("переменные повторяются")
    if not h:
        raise ValueError("нет ограничений")
    if EPS <= 0:
        raise ValueError("EPS должно быть положительным")

    # в выражениях не должно быть посторонних символов
    known = set(variables)
    for expr in (f,) + tuple(h):
        extra = expr.free_symbols - known
        if extra:
            names = ", ".join(str(s) for s in extra)
            raise ValueError("лишние символы: " + names)

    # пробуем взять производные - если не получится, задача негладкая
    try:
        for v in variables:
            sp.diff(f, v)
        for c in h:
            for v in variables:
                sp.diff(c, v)
    except (ValueError, TypeError, NotImplementedError) as e:
        raise ValueError(f"производные не вычисляются: {e}")

    print("проверка входных данных: ок")
    print(f"переменных: {len(variables)}, ограничений: {len(h)}")
    if len(h) > len(variables):
        print("предупреждение: ограничений больше, чем переменных")


# ----- решение системы -----

def is_real(v: sp.Expr) -> bool:
    """ возвращает True, если значение действительное и конечное """

    v = sp.simplify(v)
    if v.free_symbols:
        return False
    if v.is_real is True and v.is_finite is not False:
        return True
    if v.is_real is False or v.is_finite is False:
        return False
    try:
        return abs(complex(sp.N(v)).imag) <= EPS
    except (TypeError, ValueError, OverflowError):
        return False


def collect_solutions(raw: Any, unknowns: tuple) -> tuple[list[dict], bool]:
    """ приводит результат solve/nonlinsolve к списку словарей вида:
    [{x: ..., y: ..., lam1: ...}, ...]
    возвращает (решения, найдено_семейство)
    семейство - это когда решатель вернул параметрическое решение,
    его нельзя считать точками
    """

    if raw is None:
        return [], True
    if raw is sp.EmptySet or raw == sp.EmptySet:
        return [], False

    if isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, (list, tuple, set, sp.FiniteSet)):
        items = list(raw)
    else:
        return [], True

    sols = []
    family = False
    for item in items:
        if isinstance(item, dict):
            sol = {k: sp.simplify(v) for k, v in item.items() if k in unknowns}
        elif isinstance(item, (tuple, sp.Tuple)) and len(item) == len(unknowns):
            sol = {u: sp.simplify(v) for u, v in zip(unknowns, item)}
        else:
            family = True
            continue

        # если для какой-то неизвестной нет значения в словаре - значит,
        # она свободна, это семейство
        if any(u not in sol for u in unknowns):
            family = True
            continue
        # если значение содержит свободный символ
        # (напрмер, t) - тоже семейство
        if any(val.free_symbols for val in sol.values()):
            family = True
            continue
        if not all(is_real(val) for val in sol.values()):
            continue
        # отбрасываем дубликаты
        if not any(all(sp.simplify(sol[u] - old[u]) == 0 for u in unknowns) for old in sols):
            sols.append(sol)
    return sols, family


def solve_system(equations: list[sp.Expr], unknowns: tuple, name: str) -> dict:
    """ решает систему через solve и nonlinsolve """

    sols = []
    family = False
    methods = []

    # solve хорошо работает с полиномиальными системами
    try:
        raw = sp.solve(equations, list(unknowns), dict=True, simplify=True)
        new, fam = collect_solutions(raw, unknowns)
        sols.extend(new)
        family = family or fam
        methods.append("solve")
    except Exception as e:
        print(f"  solve не сработал: {e}")

    # nonlinsolve - метод для нелинейных систем
    try:
        raw = sp.nonlinsolve(equations, list(unknowns))
        new, fam = collect_solutions(raw, unknowns)
        for s in new:
            if not any(all(sp.simplify(s[u] - old[u]) == 0 for u in unknowns) for old in sols):
                sols.append(s)
        family = family or fam
        methods.append("nonlinsolve")
    except Exception as e:
        print(f"  nonlinsolve не сработал: {e}")

    if sols:
        status = "конечное"
    elif family:
        status = "семейство"
    elif methods:
        status = "пусто"
    else:
        status = "неизвестно"

    print(f"\n{name}:")
    print("  методы:", ", ".join(methods) if methods else "не удалось")
    print("  статус:", status)
    print("  решений:", len(sols))
    if family:
        print("  внимание: получено семейство, полнота не подтверждена")

    return {"solutions": sols, "status": status, "family": family}


# ----- вспомогательные функции -----

def residual_is_zero(r: sp.Expr) -> bool:
    """ проверяет, что невязка близка к нулю """

    r = sp.simplify(r)
    # если точно знаем, что r = 0
    if r == 0 or r.is_zero is True:
        return True
    if r.is_zero is False:
        # если не 0, но близко - пробуем перевести значение в число
        # и сравнить с EPS
        try:
            return abs(complex(sp.N(r))) <= EPS
        except (TypeError, ValueError, OverflowError):
            return False
    try:
        return abs(complex(sp.N(r))) <= EPS
    except (TypeError, ValueError, OverflowError):
        return False


def get_jacobian() -> sp.Matrix:
    """ матрица Якоби ограничений """

    return sp.Matrix([[sp.diff(c, v) for v in variables] for c in h])


def get_rank(J: sp.Matrix, subs: dict) -> int | None:
    """ ранг матрицы Якоби в точке """

    try:
        return int(J.subs(subs).rank())
    except (ValueError, TypeError, NotImplementedError):
        return None


def singular_equations() -> list[sp.Expr]:
    """ система для поиска особых точек, где ранг Якоби падает
    ранг меньше m тогда и только тогда, когда все миноры порядка m равны нулю
    """

    J = get_jacobian()
    m, n = J.shape
    eqs = list(h)
    if m <= n:
        for cols in combinations(range(n), m):
            eqs.append(sp.factor(J[:, cols].det()))
    return [sp.simplify(e) for e in eqs]


def classify(lagrangian: sp.Expr, subs: dict) -> str:
    """ классификация второго порядка по приведенному гессиану """

    # матрица Якобиана ограничений в точке
    J = get_jacobian().subs(subs)

    try:
        # базис касательного пространства (ядро J)
        basis = J.nullspace()
        if not basis:
            return "изолированная точка, второй порядок неинформативен"
        B = sp.Matrix.hstack(*basis)

        # гессиан лагранжиана, суженный на касательное пространство
        H = sp.hessian(lagrangian, variables).subs(subs)
        H_red = sp.simplify(B.T * H * B)

        # собственные значения приведённого гессиана
        vals = []
        for val, mult in H_red.eigenvals().items():
            for _ in range(mult):
                num = complex(sp.N(val, 30))     # переводим в число
                if abs(num.imag) > EPS:
                    return "комплексное собственное значение"
                vals.append(num.real)

        # знаки собственных значений
        if vals and all(v > EPS for v in vals):
            return "строгий локальный минимум"
        if vals and all(v < -EPS for v in vals):
            return "строгий локальный максимум"
        if any(v > EPS for v in vals) and any(v < -EPS for v in vals):
            return "не экстремум"
        return "второй порядок не дал однозначного ответа"

    except (ValueError, TypeError, NotImplementedError, ZeroDivisionError):
        return "классификация не удалась"


def fmt(mapping: dict, vars_: tuple) -> str:
    """ печатает словарь в виде (x = ..., y = ...) """

    return "(" + ", ".join(f"{v} = {sp.simplify(mapping[v])}" for v in vars_) + ")"


# ----- основная часть -----

def main():
    check_input()

    print("-" * 60)
    print("f =", f)
    for i, c in enumerate(h, 1):
        print(f"h{i} = {c} = 0")

    # множители Лагранжа
    mults = tuple(sp.symbols(f"lam1:{len(h) + 1}", real=True))

    # лагранжиан L = f + sum(lam_j * h_j)
    L = sp.expand(f + sum(m * c for m, c in zip(mults, h)))
    print("\nфункция Лагранжа:")
    print("  L =", L)

    # система: частные производные L по x_i + исходные ограничения
    eqs = [sp.diff(L, v) for v in variables] + list(h)
    unknowns = variables + mults

    print("\nсистема:")
    for i, e in enumerate(eqs, 1):
        print(f"  ({i}) {sp.simplify(e)} = 0")

    # решаем систему Лагранжа
    lag_res = solve_system(eqs, unknowns, "решение системы Лагранжа")

    # ищем особые точки (где ранг матрицы Якоби падает)
    sing_res = solve_system(singular_equations(), variables, "поиск особых точек")

    # собираем кандидатов
    candidates = []
    for sol in lag_res["solutions"]:
        point = {v: sp.simplify(sol[v]) for v in variables}
        lam = {m: sp.simplify(sol[m]) for m in mults}
        if not all(residual_is_zero(c.subs(point)) for c in h):
            print("отброшено (не проходит по ограничению):", point)
            continue
        candidates.append({"point": point, "mult": lam, "kind": "Лагранж"})

    for sol in sing_res["solutions"]:
        point = {v: sp.simplify(sol[v]) for v in variables}
        if not all(residual_is_zero(c.subs(point)) for c in h):
            continue
        if any(all(sp.simplify(point[v] - c["point"][v]) == 0 for v in variables) for c in candidates):
            continue
        candidates.append({"point": point, "mult": None, "kind": "особая"})

    # вывод кандидатов
    J = get_jacobian()
    print("\nградиенты ограничений:")
    for i, c in enumerate(h, 1):
        grad = [sp.diff(c, v) for v in variables]
        print(f"  grad(h{i}) = ({', '.join(map(str, grad))})")

    print("\nкандидаты:")
    if not candidates:
        print("  нет конечных допустимых кандидатов")

    for i, c in enumerate(candidates, 1):
        point = c["point"]
        res = [sp.simplify(cn.subs(point)) for cn in h]
        val = sp.simplify(f.subs(point))
        rank = get_rank(J, point)
        ok = all(residual_is_zero(r) for r in res)

        subs = dict(point)
        if c["mult"] is not None:
            subs.update(c["mult"])
            cls = classify(L, subs)
        else:
            cls = "особая, метод Лагранжа не применим"

        c.update({"value": val, "res": res, "rank": rank, "ok": ok, "cls": cls})

        print(f"\nКандидат {i} ({c['kind']}):")
        print("  точка:", fmt(point, variables))
        if c["mult"] is None:
            print("  множители: не определены")
        else:
            print("  множители:", fmt(c["mult"], mults))
        print("  f(x) =", val)
        print("  невязки:", res)
        print("  допустимость:", "да" if ok else "нет")
        print("  ранг:", rank, "/", len(h))
        print("  классификация:", cls)

    # итог
    print("\n" + "-" * 60)
    print("итог")
    print("-" * 60)

    # глобальный вывод разрешаем только если все проверено
    can_global = (
        compact and full_list
        and candidates
        and lag_res["status"] == "конечное" and not lag_res["family"]
        and sing_res["status"] in ("пусто", "конечное") and not sing_res["family"]
        and all(c["ok"] for c in candidates)
    )

    if can_global:
        mn = min(candidates, key=lambda c: float(sp.N(c["value"])))
        mx = max(candidates, key=lambda c: float(sp.N(c["value"])))
        print("глобальный минимум:")
        print("  точка:", fmt(mn["point"], variables))
        print("  значение:", mn["value"])
        print("глобальный максимум:")
        print("  точка:", fmt(mx["point"], variables))
        print("  значение:", mx["value"])
        print("основание: множество компактно, список кандидатов полон, значения сравнены")
    else:
        print("глобальные экстремумы автоматически не объявлены")
        print("нужен дополнительный анализ задачи")

if __name__ == "__main__":
    main()