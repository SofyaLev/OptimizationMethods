import csv
from pathlib import Path
from time import perf_counter

import matplotlib

# сохраняем рисунки без открытия отдельных окон
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from components import particle_swarm, simulated_annealing


# ----- настройки экспериментов (вариант 14) -----

ROOT = Path(__file__).resolve().parent
BOUNDS = [(-100.0, 100.0), (-100.0, 100.0)]
BUDGET = 10_000
RUNS = 30


def schaffer2(point):
    """ вычисляет функцию Шаффера N.2 по формуле из задания """

    x, y = point
    return 0.5 + (np.sin(x*x - y*y)**2 - 0.5) / (1 + 0.001*(x*x + y*y))**2


# ----- серия запусков и проверка результатов -----

def run_series(method, **parameters):
    """
    выполняет 30 запусков метода с зернами от 1 до 30

    внешняя функция objective считает обращения к f и проверяет границы
    время измеряется вокруг вызова метода
    контрольный пересчет результата выполняется после остановки таймера

    возвращает список измерений для всех запусков
    """

    records = []
    for seed in range(1, RUNS + 1):
        calls = 0

        def objective(point):
            """ вычисляет f с проверкой бюджета и допустимости точки """

            nonlocal calls
            calls += 1
            if calls > BUDGET:
                raise RuntimeError("Превышен бюджет")
            if not np.all(np.isfinite(point)) or np.any(np.abs(point) > 100):
                raise RuntimeError("Недопустимая точка")
            return schaffer2(point)

        started = perf_counter()
        result = method(objective, BOUNDS, BUDGET, seed, **parameters)
        elapsed = perf_counter() - started
        # контрольное вычисление выполняется после остановки таймера
        actual = float(schaffer2(result["x"]))
        if calls != result["nfe"] or not np.isclose(actual, result["f"], rtol=0, atol=1e-14):
            raise RuntimeError("Результат не согласуется с внешней проверкой")
        records.append({"seed": seed, "x1": result["x"][0], "x2": result["x"][1],
                        "f": actual, "nfe": calls, "time_s": elapsed,
                        "success": int(actual <= 1e-4)})
    return records


def write_csv(path, records):
    """ сохраняет измерения в CSV вместе с заголовками столбцов """

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


# ----- сравнение компонентов и чувствительность параметров -----

def main():
    """ сравнивает PSO и SA и исследует три параметра роя """

    results_dir = ROOT / "results"
    image_dir = ROOT / "img"
    results_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    # прогрев не входит в измерения
    particle_swarm(schaffer2, BOUNDS, 100, seed=0)
    simulated_annealing(schaffer2, BOUNDS, 100, seed=0)
    baseline = run_series(particle_swarm)
    annealing = run_series(simulated_annealing)
    summary = []
    for name, records in [("PSO", baseline), ("SA", annealing)]:
        write_csv(results_dir / f"{name.lower()}_runs.csv", records)
        values = np.array([row["f"] for row in records])
        row = {"method": name, "best": values.min(), "median": np.median(values),
               "worst": values.max(), "success_rate": np.mean(values <= 1e-4),
               "mean_time_s": np.mean([r["time_s"] for r in records]),
               "mean_nfe": np.mean([r["nfe"] for r in records])}
        summary.append(row)
        print(row, flush=True)
    write_csv(results_dir / "components_summary.csv", summary)

    settings = {"inertia": ([0.2, 0.4, 0.7, 1.0, 1.2], 0.7),
                "social": ([0.0, 0.5, 1.5, 3.0, 5.0], 1.5),
                "velocity_fraction": ([0.001, 0.01, 0.1, 0.2, 0.5], 0.2)}
    sensitivity = []
    all_runs = []
    labels = {"inertia": "Инерция w", "social": "Социальный коэффициент c₂",
              "velocity_fraction": "Максимальная скорость / ширина области"}
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, (parameter, (grid, default)) in zip(axes, settings.items()):
        rates = []
        for value in grid:
            # при стандартном параметре используем уже выполненную серию PSO
            records = baseline if value == default else run_series(particle_swarm, **{parameter: value})
            rate = np.mean([r["success"] for r in records])
            rates.append(rate)
            sensitivity.append({"parameter": parameter, "value": value,
                                "success_rate": rate,
                                "median_f": np.median([r["f"] for r in records])})
            all_runs.extend({"parameter": parameter, "value": value, **row} for row in records)
            print(parameter, value, rate, flush=True)
        axis.plot(grid, rates, "o-")
        axis.set(xlabel=labels[parameter], ylabel="Доля успехов", ylim=(-0.05, 1.05))
        axis.grid(alpha=0.3)
    figure.savefig(image_dir / "sensitivity.png", dpi=170)
    plt.close(figure)
    write_csv(results_dir / "sensitivity.csv", sensitivity)
    write_csv(results_dir / "sensitivity_runs.csv", all_runs)

    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    errors = [np.log10(np.maximum([r["f"] for r in rows], 1e-16))
              for rows in (baseline, annealing)]
    axis.boxplot(errors)
    axis.set_xticks([1, 2], ["PSO", "SA"])
    axis.axhline(-4, color="tab:red", linestyle="--", label="Порог успеха")
    axis.set_ylabel("lg(max(f − f*, 10⁻¹⁶))")
    axis.legend()
    figure.savefig(image_dir / "components.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
