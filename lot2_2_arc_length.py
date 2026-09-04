"""Lot 2.2 -- longueur d'arc cumulee sur Delta(Z) (plan_action_code.md).

Distance parcourue sur la variete (metrique de Fisher-Rao,
kst_engine.cumulative_arc_length) en fonction du numero de question,
adaptatif vs aleatoire, moyennee sur tous les etats de domain.Z (piste B).

Attendu (plan) : l'adaptatif parcourt davantage de distance par question au
debut -- visible comme une pente plus forte en tout debut de trajectoire.

Usage : python lot2_2_arc_length.py
Ecrit results/lot2_2_arc_length/{raw.csv, summary.csv, figure.pdf, run.log}
+ une copie lot2_2_arc_length_figure.png a la racine.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from domains.loader import load_domain_yaml
from kst_engine import Domain, cumulative_arc_length, simulate

ROOT = Path(__file__).resolve().parent
DOMAIN_PATH = ROOT / "domains" / "piste_b.yaml"
RESULTS_DIR = ROOT / "results" / "lot2_2_arc_length"


def run(domain: Domain) -> dict[str, list[list[float]]]:
    """Pour chaque etat, la sequence de longueur d'arc cumulee (adaptatif
    et aleatoire), avec la meme graine pour les deux (comparaison appariee,
    meme convention que benchmark_a3.py)."""
    sequences: dict[str, list[list[float]]] = {"adaptatif": [], "aleatoire": []}
    for seed, z in enumerate(domain.Z):
        for policy, adaptive in (("adaptatif", True), ("aleatoire", False)):
            r = simulate(domain, z, adaptive=adaptive, seed=seed)
            sequences[policy].append(cumulative_arc_length(r["belief_trace"]))
    return sequences


def pad_and_average(sequences: list[list[float]]) -> np.ndarray:
    """Aligne des trajectoires de longueurs differentes en prolongeant
    chacune par sa derniere valeur (le quiz s'est arrete : plus aucune
    distance ne s'accumule ensuite), puis moyenne colonne par colonne."""
    max_len = max(len(s) for s in sequences)
    padded = np.array([s + [s[-1]] * (max_len - len(s)) for s in sequences])
    return padded.mean(axis=0)


def write_raw_csv(sequences: dict[str, list[list[float]]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "z_seed", "question_index", "longueur_arc_cumulee"])
        for policy, seqs in sequences.items():
            for seed, seq in enumerate(seqs):
                for t, val in enumerate(seq):
                    w.writerow([policy, seed, t, f"{val:.6f}"])


def write_summary_csv(means: dict[str, np.ndarray], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "question_index", "longueur_arc_moyenne"])
        for policy, mean in means.items():
            for t, val in enumerate(mean):
                w.writerow([policy, t, f"{val:.6f}"])


def fig_summary(means: dict[str, np.ndarray], path: Path) -> None:
    colors = {"adaptatif": "#2E86AB", "aleatoire": "#C0C0C0"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for policy, mean in means.items():
        ax.plot(range(len(mean)), mean, marker="o", markersize=3,
               color=colors[policy], linewidth=2, label=f"{policy} (π*)" if policy == "adaptatif" else policy)
    ax.set_xlabel("Question posée (t)")
    ax.set_ylabel("Longueur d'arc cumulée (distance de Fisher-Rao)")
    ax.set_title("Lot 2.2 — distance parcourue sur Δ(Z)\n"
                 "(piste B, 7 concepts, moyenne sur les 50 états, |Z|=50)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True, cwd=ROOT)
        return r.stdout.strip()
    except Exception:
        return "inconnu (git indisponible)"


def write_run_log(path: Path, n_states: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC) : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande   : python lot2_2_arc_length.py\n")
        f.write(f"commit git : {_git_commit()}\n")
        f.write(f"domaine    : {DOMAIN_PATH.relative_to(ROOT)}\n")
        f.write(f"graines    : seed = index de l'etat dans domain.Z, "
               f"0..{n_states - 1} (deterministe, {n_states} etats), "
               f"meme graine pour adaptatif et aleatoire (comparaison appariee)\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domain, meta, labels = load_domain_yaml(DOMAIN_PATH)

    sequences = run(domain)
    means = {policy: pad_and_average(seqs) for policy, seqs in sequences.items()}

    for policy, mean in means.items():
        print(f"{policy:10s} : t=0 -> {mean[0]:.3f}, t=5 -> "
             f"{mean[5] if len(mean) > 5 else mean[-1]:.3f}, "
             f"final -> {mean[-1]:.3f} (sur {len(mean)} pas)")

    write_raw_csv(sequences, RESULTS_DIR / "raw.csv")
    write_summary_csv(means, RESULTS_DIR / "summary.csv")
    fig_summary(means, RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log", len(domain.Z))
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure.pdf, run.log}}")

    fig_summary(means, ROOT / "lot2_2_arc_length_figure.png")
    print("ecrit (copie racine) : lot2_2_arc_length_figure.png")


if __name__ == "__main__":
    main()
