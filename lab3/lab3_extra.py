from pathlib import Path

import matplotlib

# сохраняем графики в папку работы без открытия отдельных окон
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import sympy as sp

from lab3 import f, h, variables


# ----- геометрия варианта 14 -----

ROOT = Path(__file__).resolve().parent


def find_candidates():
    """ заново строит систему Лагранжа по данным основной работы """

    multiplier = sp.Symbol("lam", real=True)
    lagrangian = f + multiplier * h[0]
    equations = [sp.diff(lagrangian, variable) for variable in variables] + list(h)
    solutions = sp.solve(equations, (*variables, multiplier), dict=True)
    candidates = []
    for solution in solutions:
        if not all(solution[v].is_real is True for v in (*variables, multiplier)):
            continue
        point = tuple(solution[v] for v in variables)
        substitutions = dict(zip(variables, point))
        if any(sp.simplify(constraint.subs(substitutions)) != 0 for constraint in h):
            continue
        candidates.append((point, solution[multiplier], sp.simplify(f.subs(substitutions))))
    return sorted(candidates, key=lambda candidate: float(candidate[2]))


# ----- эллипсоид и значения целевой функции -----

def draw_ellipsoid(candidates, image_dir):
    """ показывает допустимое множество и найденные минимум и максимум """

    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, 121), np.linspace(0, np.pi, 81))
    x = 2 * np.sin(theta) * np.cos(phi)
    y = np.sqrt(2) * np.sin(theta) * np.sin(phi)
    z = 2 * np.cos(theta)
    function = sp.lambdify(variables, f, "numpy")
    values = function(x, y, z)
    norm = Normalize(float(candidates[0][2]), float(candidates[-1][2]))
    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(projection="3d", computed_zorder=False)
    axis.plot_surface(x, y, z, facecolors=plt.cm.coolwarm(norm(values)),
                      rcount=81, ccount=121, alpha=0.55, shade=False, linewidth=0, zorder=1)
    for (point, multiplier, value), color in zip(candidates, ["navy", "darkred"]):
        coords = np.array(point, dtype=float)
        axis.scatter(*coords, s=100, marker="*", color=color, depthshade=False, zorder=3,
                     label=f"{tuple(point)}, f = {value}, λ = {multiplier}")
    axis.set(xlabel="x", ylabel="y", zlabel="z", title="Допустимый эллипсоид и экстремумы")
    axis.set_box_aspect((4, 2 * np.sqrt(2), 4))
    axis.view_init(elev=20, azim=-55)
    axis.legend(loc="upper left", fontsize=8)
    colorbar = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    figure.colorbar(colorbar, ax=axis, shrink=0.6, label="f(x, y, z) = x + 2y - z")
    figure.savefig(image_dir / "lagrange_ellipsoid.png", dpi=180)
    plt.close(figure)


# ----- сечение через обе точки экстремума -----

def draw_section(candidates, image_dir):
    """ строит сечение z = -x и линии уровня целевой функции """

    axis_values = np.linspace(-1.8, 1.8, 401)
    x, y = np.meshgrid(axis_values, axis_values)
    section = sp.simplify(f.subs(variables[2], -variables[0]))
    function = sp.lambdify(variables[:2], section, "numpy")
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    contours = axis.contour(x, y, function(x, y), levels=[-4, -2, 0, 2, 4],
                            cmap="coolwarm", linewidths=1.3)
    axis.clabel(contours, inline=True, fontsize=9, fmt="f = %g")
    angle = np.linspace(0, 2 * np.pi, 501)
    axis.plot(np.sqrt(2) * np.cos(angle), np.sqrt(2) * np.sin(angle),
              color="black", linewidth=2, label="Сечение эллипсоида, x² + y² = 2")
    for (point, _, value), color in zip(candidates, ["navy", "darkred"]):
        x_point, y_point = map(float, point[:2])
        axis.scatter(x_point, y_point, s=120, marker="*", color=color, zorder=4)
        axis.annotate(f"({point[0]}, {point[1]}, {point[2]}), f = {value}",
                      (x_point, y_point), xytext=(7, 12), textcoords="offset points", fontsize=9)
    axis.set(xlabel="x", ylabel="y", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8),
             title="Сечение z = -x, целевая функция f = 2x + 2y")
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    figure.savefig(image_dir / "lagrange_section.png", dpi=180)
    plt.close(figure)


def main():
    """ находит кандидатов и сохраняет два рисунка своего варианта """

    candidates = find_candidates()
    if len(candidates) != 2:
        raise RuntimeError("Для этого изображения ожидаются два кандидата варианта 14")
    image_dir = ROOT / "img"
    image_dir.mkdir(exist_ok=True)
    for point, multiplier, value in candidates:
        print(f"Точка {point}, λ = {multiplier}, f = {value}")
    draw_ellipsoid(candidates, image_dir)
    draw_section(candidates, image_dir)
    print(f"Графики сохранены в {image_dir}")


if __name__ == "__main__":
    main()
