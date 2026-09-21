def print_table(table, basis_cols, col_names, title, obj_name="F"):
    """ функция для читаемого вывода таблиц """

    print(f"\n{title}")
    header = ["базис"] + col_names + ["b"]  # шапка: базис, имена переменных, свободный член
    print(" | ".join(f"{x:>8}" for x in header))
    print("-" * (11 * len(header)))

    for r in range(len(basis_cols)):
        b_name = col_names[basis_cols[r]]  # имя базисной переменной в этой строке
        row_str = [f"{b_name:>8}"] + [f"{val:8.3f}" for val in table[r]]
        print(" | ".join(row_str))

    obj_str = [f"{obj_name:>8}"] + [f"{val:8.3f}" for val in table[-1]]  # последняя строка - целевая функция (W или Z)
    print(" | ".join(obj_str))


def pivot(table, row, col):
    """ функция для пересчета таблицы """

    pivot_val = table[row][col]  # фиксируем разрешающий элемент
    # для разрешающей строки делим каждый ее элемент на разрешающий
    table[row] = [x / pivot_val for x in table[row]]
    for r in range(len(table)):
        if r != row:
            # для остальных строк фиксируем элемент,
            # который в ручном решении соответствует a[i][j*]
            coeff = table[r][col]

            # пересчитываем элементы (запись вида ^a означает новый элемент)
            # по формуле: ^a[i][j] = a[i][j] - a[i][j*] * a[i*][j]/a[i*][j*]
            table[r] = [table[r][c] - coeff * table[row][c] for c in range(len(table[0]))]


def solve(c, A, b, constraint_types, free_vars, optimization="max"):
    """ основная функция для решения задачи """

    n_vars, n_constraints = len(c), len(A)
    n_vars_orig = n_vars  # запоминаем исходное число переменных - понадобится при сборке ответа

    # копируем входные данные, чтобы не портить оригиналы
    A = [row[:] for row in A]
    b = list(b)
    constraint_types = list(constraint_types)

    # --- замена свободных переменных ---
    # если переменная свободна (может быть < 0),
    # делаем замену x_k = x_k+ - x_k-
    # в orig_map храним пары (индекс исходной переменной, знак):
    # +1 для x_k+, -1 для x_k-
    orig_map = []
    for j in range(n_vars):
        if j in free_vars:
            orig_map.append((j, +1))
            orig_map.append((j, -1))
        else:
            orig_map.append((j, +1))

    n_new = len(orig_map)  # переменных стало больше, если были свободные
    new_c = [0.0] * n_new
    new_A = [[0.0] * n_new for _ in range(n_constraints)]
    for i in range(n_constraints):
        for k, (j, s) in enumerate(orig_map):
            new_A[i][k] = A[i][j] * s  # коэффициент при x_k+ как был, при x_k- с минусом
    for k, (j, s) in enumerate(orig_map):
        new_c[k] = c[j] * s

    c = new_c
    A = new_A
    n_vars = n_new

    # --- если правая часть отрицательна, умножаем строку на -1 ---
    # при этом знак неравенства меняется на противоположный
    for i in range(n_constraints):
        if b[i] < 0:
            b[i] = -b[i]
            A[i] = [-x for x in A[i]]
            if constraint_types[i] == "<=":
                constraint_types[i] = ">="
            elif constraint_types[i] == ">=":
                constraint_types[i] = "<="

    # --- канонизация: добавляем балансирующие переменные (s_i) / искусственные (a_i) ---
    names = [f"x{j + 1}" for j in range(n_vars)]  # имена исходных переменных
    expanded_A = [row[:] for row in A]  # расширенная матрица
    basis = [-1] * n_constraints  # индекс базисной переменной в каждой строке
    artificial_vars = set()  # индексы искусственных переменных
    col_idx = n_vars  # счетчик новых столбцов

    for i, ctype in enumerate(constraint_types):
        if ctype == "<=":
            # для <= добавляем s_i с коэффициентом +1, она сразу становится базисной
            for r in range(n_constraints):
                expanded_A[r].append(1.0 if r == i else 0.0)
            names.append(f"s{i + 1}")
            basis[i] = col_idx
            col_idx += 1
        elif ctype == ">=":
            # для >= добавляем s_i с -1,
            # потом искусственную с +1 - она становится базисной
            for r in range(n_constraints):
                expanded_A[r].append(-1.0 if r == i else 0.0)
            names.append(f"s{i + 1}")
            col_idx += 1

            for r in range(n_constraints):
                expanded_A[r].append(1.0 if r == i else 0.0)
            names.append(f"a{i + 1}")
            artificial_vars.add(col_idx)
            basis[i] = col_idx
            col_idx += 1
        elif ctype == "=":
            # для = сразу добавляем искусственную с +1
            for r in range(n_constraints):
                expanded_A[r].append(1.0 if r == i else 0.0)
            names.append(f"a{i + 1}")
            artificial_vars.add(col_idx)
            basis[i] = col_idx
            col_idx += 1

    total_vars = col_idx  # общее число переменных после канонизации

    print("\nканонический вид:")
    for i in range(n_constraints):
        print(f"ограничение {i + 1}: {[round(x, 3) for x in expanded_A[i]]} = {b[i]}")
    print("переменные:", names)
    print("базис:", [names[idx] for idx in basis])

    # --- строим таблицу ограничений: к каждой строке приклеиваем b_i ---
    table = [expanded_A[i] + [b[i]] for i in range(n_constraints)]

    # --- строим строку W = сумма искусственных -> min ---
    # искусственные надо выразить через свободные:
    # для этого вычитаем строки с искусственным базисом из нулевой строки
    w_row = [0.0] * (total_vars + 1)
    for i in range(n_constraints):
        if basis[i] in artificial_vars:
            for j in range(total_vars + 1):
                w_row[j] -= table[i][j]
    table.append(w_row)

    # --- этап 1: решаем вспомогательную задачу W -> min ---
    if artificial_vars:
        step = 1
        print_table(table, basis, names, "этап 1: начальная таблица", obj_name="W")
        while True:
            w_coeffs = table[-1][:-1]  # коэффициенты строки W без свободного члена
            min_val = min(w_coeffs)

            if min_val >= -1e-9:  # если все коэффициенты >= 0 - оптимум W найден
                break

            col = w_coeffs.index(min_val)  # разрешающий столбец: минимальный отрицательный коэффициент
            # разрешающая строка: минимальное отношение b/a среди положительных a
            ratios = [(table[r][-1] / table[r][col], r) for r in range(n_constraints) if table[r][col] > 1e-9]

            if not ratios:
                print("\nсистема ограничений несовместна")
                return None

            row = min(ratios)[1]
            print(
                f"шаг {step}: вводим {names[col]}, выводим {names[basis[row]]} "
                f"(разрешающий элемент: [{row + 1}, {col + 1}] = {table[row][col]:.3f})"
            )
            pivot(table, row, col)
            basis[row] = col  # обновляем базис: новая переменная вошла в строку
            print_table(table, basis, names, f"этап 1: таблица {step}", obj_name="W")
            step += 1

        # если W_min != 0 - не все искусственные обнулились, задача несовместна
        if abs(table[-1][-1]) > 1e-9:
            print("\nW_min != 0: задача не имеет допустимых решений")
            return None

        # --- отдельный случай: искусственная осталась в базисе с нулем ---
        # либо выводим ее через ненулевой неискусственный элемент,
        # либо удаляем строку
        to_delete = []
        for i in range(len(basis)):
            if basis[i] in artificial_vars:
                replaced = False
                for j in range(total_vars):
                    if j not in artificial_vars and abs(table[i][j]) > 1e-9:
                        pivot(table, i, j)
                        basis[i] = j
                        replaced = True
                        break
                if not replaced:
                    to_delete.append(i)  # строка линейно зависима -> удаляем ее

        for i in sorted(to_delete, reverse=True):
            del table[i]
            del basis[i]

        n_constraints = len(basis)  # строк стало меньше

    # --- этап 2: убираем искусственные столбцы и решаем основную задачу ---
    non_art_cols = [j for j in range(total_vars) if j not in artificial_vars]
    phase2_names = [names[j] for j in non_art_cols]

    phase2_table = [[table[r][j] for j in non_art_cols] + [table[r][-1]] for r in range(n_constraints)]
    new_basis = [non_art_cols.index(b_var) for b_var in basis]  # переиндексируем базис под новую таблицу
    num_p2_vars = len(non_art_cols)

    # --- строим строку целевой функции -Z (или Z, если задача минимизации) ---
    p2_c = c + [0.0] * (num_p2_vars - n_vars)  # коэффициенты Z для всех переменных, для дополнительных - 0
    if optimization == "min":
        p2_c = [-x for x in p2_c]  # для минимума меняем знак, чтобы все свести к минимизации

    obj_row = [-x for x in p2_c] + [0.0]
    for r in range(n_constraints):
        b_col = new_basis[r]
        coeff = obj_row[b_col]  # исключаем базисные переменные из строки целевой функции
        for c_idx in range(num_p2_vars + 1):
            obj_row[c_idx] -= coeff * phase2_table[r][c_idx]

    phase2_table.append(obj_row)

    # --- цикл симплекс-метода для основной задачи ---
    step = 1
    print_table(phase2_table, new_basis, phase2_names, "этап 2: начальная таблица", obj_name="Z")

    while True:
        obj_coeffs = phase2_table[-1][:-1]
        min_val = min(obj_coeffs)
        if min_val >= -1e-9:  # все коэффициенты >= 0 - оптимум найден
            break

        col = obj_coeffs.index(min_val)  # разрешающий столбец
        ratios = [(phase2_table[r][-1] / phase2_table[r][col], r) for r in range(n_constraints) if
                  phase2_table[r][col] > 1e-9]

        if not ratios:
            print("\nцелевая функция неограничена")
            return None

        row = min(ratios)[1]
        print(
            f"шаг {step}: вводим {phase2_names[col]}, выводим {phase2_names[new_basis[row]]} "
            f"(разрешающий элемент: [{row + 1}, {col + 1}] = {phase2_table[row][col]:.3f})"
        )
        pivot(phase2_table, row, col)
        new_basis[row] = col
        print_table(phase2_table, new_basis, phase2_names, f"этап 2: таблица {step}", obj_name="Z")
        step += 1

    # --- собираем ответ ---
    x_expanded = [0.0] * total_vars
    for r in range(n_constraints):
        # базисные переменные = свободные члены
        x_expanded[new_basis[r]] = phase2_table[r][-1]

    # восстанавливаем исходные переменные: x_k = x_k+ - x_k-
    x_sol = [0.0] * n_vars_orig
    for k, (j, s) in enumerate(orig_map):
        if k < total_vars:
            x_sol[j] += s * x_expanded[k]

    opt_val = phase2_table[-1][-1]
    if optimization == "min":
        opt_val = -opt_val  # снимаем замену знака, которую делали для минимизации

    print("ответ:")
    for i in range(n_vars_orig):
        print(f"x{i + 1} = {x_sol[i]:.4f}")
    print(f"значение Z ({optimization}) = {opt_val:.4f}")
    return [round(v, 4) for v in x_sol], round(opt_val, 4)


def main():
    """ точка входа: ввод данных и запуск решения """

    n = int(input("\nвведите количество переменных целевой функции: "))
    m = int(input("введите количество ограничений: "))

    print(f"коэффициенты при переменных целевой функции через пробел (например: 1 2 4 1):")
    c = list(map(float, input("c: ").split()))
    A = []
    b = []
    constraint_types = []

    print("\nограничения")
    for i in range(m):
        print(f"\nограничение {i + 1}:")
        row = list(map(float, input(f"коэффициенты при x1..x{n} через пробел: ").split()))
        sign = input("знак (<=, >=, =): ").strip()
        b_value = float(input("правая часть: "))

        A.append(row)
        constraint_types.append(sign)
        b.append(b_value)

    # номера свободных переменных вводятся через пробел
    try:
        fs = input("свободные переменные (номера через пробел, Enter если нет): ").strip()
        free_vars = set(int(x) - 1 for x in fs.split()) if fs else set()
    except ValueError:
        free_vars = set()

    print("\nнаправление оптимизации\n")
    print("1 - максимизировать")
    print("2 - минимизировать")
    choice = input("выберите (1 или 2): ").strip()
    optimization = "min" if choice == "2" else "max"

    solve(c, A, b, constraint_types, free_vars, optimization)


if __name__ == "__main__":
    main()