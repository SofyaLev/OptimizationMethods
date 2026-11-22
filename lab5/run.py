import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

# ограничиваем число потоков до импорта NumPy
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from components import particle_swarm, simulated_annealing
from functions import get_function
from optimizer import optimize


# ----- методы и каталоги результатов -----

ROOT = Path(__file__).resolve().parent
METHODS = {"hybrid": optimize, "pso": particle_swarm, "sa": simulated_annealing}


# ----- внешний контроль одного запуска -----

def measure(method, test, max_nfe, seed):
    """
    измеряет результат, время и число обращений к функции снаружи метода

    test - эталонная функция с границами и данными для проверки
    значение известного минимума используется только при оценке результатов
    возвращает строку измерений и полный результат для визуализации
    """

    calls = 0
    first_success = 0
    lower, upper = test.lower, test.upper

    def objective(point):
        """ считает вычисления и проверяет каждую переданную точку """

        nonlocal calls, first_success
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("Некорректная точка")
        if np.any(point < lower) or np.any(point > upper):
            raise ValueError("Выход за границы области")
        if calls >= max_nfe:
            raise RuntimeError("Превышен бюджет целевой функции")
        calls += 1
        value = test(point)
        if first_success == 0 and test.error(value) <= test.eps and test.is_feasible(point):
            first_success = calls
        return value

    started = perf_counter()
    result = method(objective, test.bounds, max_nfe=max_nfe, seed=seed)
    elapsed = perf_counter() - started
    # контрольный пересчет не является частью поиска и выполняется после таймера
    point = np.asarray(result["x"], dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)) or not test.is_feasible(point):
        raise RuntimeError("Метод вернул недопустимое решение")
    value = test(point)
    if calls != result["nfe"] or not np.isclose(value, result["f"], rtol=1e-12, atol=1e-14):
        raise RuntimeError("Значение или счетчик не согласуется с внешней проверкой")
    if elapsed > 60:
        raise RuntimeError("Превышен тайм-аут 60 секунд")
    diagnostics = result.get("diagnostics", {})
    row = {
        "seed": seed,
        "x1": float(point[0]),
        "x2": float(point[1]),
        "f": float(value),
        "error": float(test.error(value)),
        "nfe": calls,
        "time_s": elapsed,
        "success": int(test.error(value) <= test.eps),
        "first_success_nfe": first_success,
        "sa_attempts": diagnostics.get("sa_attempts", 0),
        "sa_accepted": diagnostics.get("sa_accepted", 0),
        "sa_uphill": diagnostics.get("sa_uphill", 0),
        "pso_nfe": diagnostics.get("pso_nfe", 0),
        "sa_nfe": diagnostics.get("sa_nfe", 0),
    }
    return row, result


# ----- серии запусков и статистика -----

def summarize(rows):
    """ вычисляет показатели таблицы по серии независимых запусков """

    values = np.array([row["f"] for row in rows])
    errors = np.array([row["error"] for row in rows])
    log_errors = np.log10(np.maximum(errors, 1e-12))
    return {
        "best": float(values.min()),
        "median": float(np.median(values)),
        "worst": float(values.max()),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_time_s": float(np.mean([row["time_s"] for row in rows])),
        "mean_nfe": float(np.mean([row["nfe"] for row in rows])),
        "median_log_error": float(np.median(log_errors)),
        "iqr_log_error": float(np.quantile(log_errors, 0.75) - np.quantile(log_errors, 0.25)),
    }


def save_rows(path, rows):
    """ сохраняет полные измерения в CSV """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def series(method, test, runs=30, max_nfe=10_000, first_seed=1):
    """ выполняет прогрев и серию с последовательными зернами """

    if runs < 1:
        raise ValueError("Число запусков должно быть положительным")
    if test.constraints:
        raise ValueError("Для варианта 14 выбирайте функцию без дополнительных ограничений")
    method(test, test.bounds, max_nfe=min(100, max_nfe), seed=0)
    rows = []
    for seed in range(first_seed, first_seed + runs):
        row, _ = measure(method, test, max_nfe, seed)
        rows.append(row)
    return rows


# ----- запуск из командной строки -----

def main():
    """ запускает гибрид или отдельный компонент на эталонной функции """

    parser = argparse.ArgumentParser(description="ЛР5, вариант 14, PSO + SA")
    parser.add_argument("--func", default="schaffer2")
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-nfe", type=int, default=10_000)
    args = parser.parse_args()
    try:
        test = get_function(args.func)
        rows = series(METHODS[args.method], test, args.runs, args.max_nfe, args.seed)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    for row in rows:
        status = "успех" if row["success"] else "порог не достигнут"
        print(f"seed={row['seed']}  x*=({row['x1']:.9g}, {row['x2']:.9g})  "
              f"f(x*)={row['f']:.10g}  NFE={row['nfe']}  время={row['time_s']:.4f} с  {status}")
        print("JSON:", json.dumps({"func": args.func, "method": args.method, **row}, ensure_ascii=False))
    if args.runs > 1:
        path = ROOT / "results" / f"{args.method}_{args.func}_runs.csv"
        save_rows(path, rows)
        summary = summarize(rows)
        print("| Метод | Лучшее f | Медиана f | Худшее f | Доля успехов | Среднее время, с | Среднее NFE |")
        print("|---|---|---|---|---|---|---|")
        print(f"| {args.method} | {summary['best']:.6g} | {summary['median']:.6g} | "
              f"{summary['worst']:.6g} | {summary['success_rate']:.4f} | "
              f"{summary['mean_time_s']:.4f} | {summary['mean_nfe']:.0f} |")
        print(f"Измерения сохранены в {path}")


if __name__ == "__main__":
    main()
