"""Lot 2.3 -- trajectoire sur la sphere de Fisher-Rao, sous-domaine a 3 etats
(plan_action_code.md).

Restreint a un domaine minimal a 3 etats (Z={{}, {a}, {a,b}}, 2 concepts,
1 prerequis) et trace la trajectoire p_0 -> p_T sur l'octant positif d'une
sphere de rayon 2, via la carte racine x=2.sqrt(p) -- meme plongement que
kst_engine.fisher_rao_distance. Reprend l'illustration de la semaine 2 du
cours, mais avec une vraie trajectoire simulee plutot qu'un schema.

Usage : python lot2_3_sphere_trajectory.py
Ecrit results/lot2_3_sphere/{trajectoire.csv, figure.pdf, run.log} + une
copie lot2_3_sphere_figure.png a la racine.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- necessaire pour projection="3d"

from kst_engine import Concept, Domain, Question, simulate

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "lot2_3_sphere"
SEED = 0
Z_TRUE_LABEL = "maitrise complete {a,b}"


def make_three_state_domain() -> Domain:
    """Domaine minimal a 3 etats : Z={{}, {a}, {a,b}} (b exige a)."""
    concepts = [Concept("a"), Concept("b")]
    questions = [Question("qa", "a", slip=0.10, guess=0.25),
                Question("qb", "b", slip=0.10, guess=0.25)]
    return Domain(concepts=concepts, prereqs=[("a", "b")], questions=questions)


def fig_sphere(belief_trace: list[np.ndarray], state_labels: list[str], path: Path) -> None:
    x = np.array([2.0 * np.sqrt(p) for p in belief_trace])   # (T+1, 3)

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # octant positif de la sphere de rayon 2 -- surface semi-transparente,
    # sert seulement de repere visuel pour la variete
    u = np.linspace(0, np.pi / 2, 30)
    v = np.linspace(0, np.pi / 2, 30)
    xs = 2 * np.outer(np.cos(u), np.sin(v))
    ys = 2 * np.outer(np.sin(u), np.sin(v))
    zs = 2 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.12, color="#2E86AB", linewidth=0)

    ax.plot(x[:, 0], x[:, 1], x[:, 2], color="#C0504D", linewidth=2.5,
           marker="o", markersize=5, label="trajectoire (π*)")
    ax.scatter(*x[0], color="black", s=90, label="$p_0$ (prior uniforme)", zorder=5)
    ax.scatter(*x[-1], color="#2E86AB", s=90, label="$p_T$ (final)", zorder=5)

    ax.set_xlabel(f"2√p({state_labels[0]})")
    ax.set_ylabel(f"2√p({state_labels[1]})")
    ax.set_zlabel(f"2√p({state_labels[2]})")
    ax.set_title("Lot 2.3 — trajectoire $p_0 \\to p_T$ sur la sphère de Fisher-Rao\n"
                f"(domaine à 3 états, carte racine x=2√p, z_true = {Z_TRUE_LABEL})")
    ax.legend(loc="upper left")
    ax.view_init(elev=22, azim=35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(belief_trace: list[np.ndarray], state_labels: list[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"p({lbl})" for lbl in state_labels]
                  + [f"x{i}=2sqrt(p({lbl}))" for i, lbl in enumerate(state_labels)])
        for t, p in enumerate(belief_trace):
            x = 2.0 * np.sqrt(p)
            w.writerow([t] + [f"{v:.6f}" for v in p] + [f"{v:.6f}" for v in x])


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True, cwd=ROOT)
        return r.stdout.strip()
    except Exception:
        return "inconnu (git indisponible)"


def write_run_log(path: Path, n_questions: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)  : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande    : python lot2_3_sphere_trajectory.py\n")
        f.write(f"commit git  : {_git_commit()}\n")
        f.write(f"domaine     : synthetique, 2 concepts (a,b), b exige a, |Z|=3\n")
        f.write(f"z_true      : {{a,b}} ({Z_TRUE_LABEL})\n")
        f.write(f"graine      : seed={SEED} (politique adaptative pi_star)\n")
        f.write(f"n_questions : {n_questions}\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domain = make_three_state_domain()
    assert domain.n_states == 3, f"attendu 3 etats, obtenu {domain.n_states}"

    state_labels = ["{}" if len(z) == 0 else "{" + ",".join(sorted(z)) + "}" for z in domain.Z]
    print(f"etats : {state_labels}")

    r = simulate(domain, frozenset({"a", "b"}), adaptive=True, seed=SEED)
    print(f"n_questions={r['n_questions']}, belief final={r['belief']}")

    write_csv(r["belief_trace"], state_labels, RESULTS_DIR / "trajectoire.csv")
    fig_sphere(r["belief_trace"], state_labels, RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log", r["n_questions"])
    print(f"\necrit : {RESULTS_DIR}/{{trajectoire.csv, figure.pdf, run.log}}")

    fig_sphere(r["belief_trace"], state_labels, ROOT / "lot2_3_sphere_figure.png")
    print("ecrit (copie racine) : lot2_3_sphere_figure.png")


if __name__ == "__main__":
    main()
