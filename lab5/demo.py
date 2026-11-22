import argparse
import os

# настройка числа потоков должна предшествовать импорту NumPy и Matplotlib
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib

# рисунки сохраняются в файлы без открытия отдельных окон
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from functions import get_function
from run import METHODS, ROOT, measure, save_rows, series, summarize


# ----- поверхность функции -----

def make_grid(test, size=401):
    """ вычисляет сетку для рисунка отдельно от бюджета оптимизации """

    axis_x = np.linspace(test.lower[0], test.upper[0], size)
    axis_y = np.linspace(test.lower[1], test.upper[1], size)
    x, y = np.meshgrid(axis_x, axis_y)
    z = np.array([test(point) for point in np.column_stack((x.ravel(), y.ravel()))])
    return x, y, z.reshape(x.shape)


def draw_surface(test, grid, image_dir):
    """ сохраняет трехмерную поверхность на всей области поиска """

    x, y, z = grid
    figure = plt.figure(figsize=(9, 6), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    axis.plot_surface(x, y, z, cmap="viridis", rcount=150, ccount=150, linewidth=0)
    axis.set(xlabel="x1", ylabel="x2", zlabel="f(x1, x2)", title=f"Поверхность {test.name}")
    figure.savefig(image_dir / "surface.png", dpi=170)
    plt.close(figure)


# ----- рой и траектория общего рекорда -----

def draw_contour(test, grid, result, image_dir):
    """ показывает четыре состояния роя и найденную к этому моменту траекторию """

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    x, y, z = grid
    for frame, (axis, (nfe, name, positions)) in enumerate(zip(axes.flat, result["snapshots"])):
        contours = axis.contourf(x, y, z, levels=25, cmap="viridis")
        trajectory = np.array([point for moment, point in result["trajectory"] if moment <= nfe])
        axis.plot(trajectory[:, 0], trajectory[:, 1], color="white", linewidth=1,
                  label="Траектория рекорда")
        axis.scatter(positions[:, 0], positions[:, 1], s=18, color="#ffb347",
                     edgecolors="black", linewidths=0.3, label="Частицы", zorder=3)
        axis.scatter(*trajectory[-1], marker="*", s=140, color="red",
                     edgecolors="white", linewidths=0.6, label="Лучшее решение", zorder=4)
        axis.set(xlim=(test.lower[0], test.upper[0]), ylim=(test.lower[1], test.upper[1]),
                 xlabel="x1", ylabel="x2", title=f"{name}, NFE={nfe}")
        axis.legend(loc="upper right", fontsize=8)
        if frame >= 2:
            # отдельный масштаб показывает сблизившиеся частицы на поздних кадрах
            zoom = axis.inset_axes([0.08, 0.09, 0.51, 0.44])
            points = np.vstack((positions, trajectory[-1]))
            padding = np.maximum(np.ptp(points, axis=0) * 0.15,
                                 (test.upper - test.lower) * 1e-12)
            zoom.scatter(positions[:, 0], positions[:, 1], s=18, color="#ffb347",
                         edgecolors="black", linewidths=0.4, zorder=3)
            zoom.scatter(*trajectory[-1], marker="*", s=100, facecolors="none",
                         edgecolors="red", linewidths=1, zorder=4)
            zoom.set(xlim=(points[:, 0].min() - padding[0], points[:, 0].max() + padding[0]),
                     ylim=(points[:, 1].min() - padding[1], points[:, 1].max() + padding[1]))
            zoom.set_title("Увеличение роя", fontsize=8)
            zoom.set_xlabel("x1", fontsize=7, labelpad=1)
            zoom.set_ylabel("x2", fontsize=7, labelpad=1)
            zoom.tick_params(labelsize=7)
            zoom.ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useOffset=False)
            zoom.xaxis.get_offset_text().set_fontsize(7)
            zoom.yaxis.get_offset_text().set_fontsize(7)
            zoom.grid(alpha=0.3)
    figure.colorbar(contours, ax=list(axes.flat), label="f(x1, x2)", shrink=0.8)
    figure.savefig(image_dir / "contour.png", dpi=170)
    plt.close(figure)


# ----- история точности и сравнение методов -----

def draw_convergence(test, result, image_dir):
    """ строит ошибку рекорда после каждого вычисления и отмечает события SA """

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    history = np.asarray(result["history"])
    errors = np.maximum((history - test.f_star) / max(1.0, abs(test.f_star)), 1e-12)
    axis.semilogy(np.arange(1, len(history) + 1), errors, label="Ошибка общего рекорда")
    axis.axhline(test.eps, color="tab:red", linestyle="--", label="Порог успеха")
    first = True
    for nfe, name in result["switches"]:
        if name == "SA принял ухудшение лидера":
            axis.axvline(nfe, color="tab:orange", alpha=0.4, linewidth=1,
                         label="Принято ухудшение лидера" if first else None)
            first = False
    axis.set(xlabel="Число вычислений функции, NFE", ylabel="max(e, 10⁻¹²)",
             title="Сходимость PSO + SA, seed=1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(image_dir / "convergence.png", dpi=170)
    plt.close(figure)


def draw_boxplot(test, measurements, image_dir):
    """ сравнивает ошибки трех методов при одинаковом бюджете """

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data = [np.log10(np.maximum([row["error"] for row in rows], 1e-12))
            for rows in measurements.values()]
    axis.boxplot(data)
    axis.set_xticks([1, 2, 3], ["PSO + SA", "PSO", "SA"])
    axis.axhline(np.log10(test.eps), color="tab:red", linestyle="--", label="Порог успеха")
    runs = len(next(iter(measurements.values())))
    axis.set(ylabel="lg(max(e, 10⁻¹²))", title=f"Сравнение методов, запусков {runs}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(image_dir / "boxplot.png", dpi=170)
    plt.close(figure)


# ----- выполнение эксперимента -----

def main():
    """ запускает три метода и сохраняет таблицы и рисунки основного отчета """

    parser = argparse.ArgumentParser(description="Эксперимент ЛР5, вариант 14")
    parser.add_argument("--func", default="schaffer2")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max-nfe", type=int, default=10_000)
    args = parser.parse_args()
    if args.runs < 30:
        parser.error("Для сравнения требуется минимум 30 запусков")
    try:
        test = get_function(args.func)
        if test.constraints:
            raise ValueError("Этот вариант предназначен для функций без дополнительных ограничений")
    except (ValueError, KeyError) as error:
        parser.error(str(error))

    # отдельная папка сохраняет основной эксперимент при проверке других функций
    image_dir = ROOT / "img"
    if args.func != "schaffer2":
        image_dir = image_dir / args.func
    image_dir.mkdir(parents=True, exist_ok=True)
    measurements = {}
    summaries = []
    for name, method in METHODS.items():
        rows = series(method, test, args.runs, args.max_nfe)
        measurements[name] = rows
        save_rows(ROOT / "results" / f"{name}_{args.func}_runs.csv", rows)
        summary = {"method": name, **summarize(rows)}
        summaries.append(summary)
        print(summary, flush=True)
    save_rows(ROOT / "results" / f"comparison_{args.func}.csv", summaries)

    # повторяем seed=1 только для рисунков, его время в таблицу не добавляется
    _, result = measure(METHODS["hybrid"], test, args.max_nfe, seed=1)
    grid = make_grid(test)
    draw_surface(test, grid, image_dir)
    draw_contour(test, grid, result, image_dir)
    draw_convergence(test, result, image_dir)
    draw_boxplot(test, measurements, image_dir)
    print(f"Рисунки сохранены в {image_dir}")


if __name__ == "__main__":
    main()
