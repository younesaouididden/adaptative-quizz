"""Lot 1, E2 -- cout en aval de l'approximation (plan_action_code.md).

Rejoue le benchmark complet (comme benchmark_a3.py) mais avec pi_hat
(politique approximee, kst_engine.simulate_mc) au lieu de pi_star, pour
chaque N in {1,3,5,10,30,100} et chaque mode (sample_y, sample_z), sur
plusieurs replications par etat (le MC introduit du bruit dans la
trajectoire elle-meme, pas seulement dans une decision isolee -- E1
mesurait une decision, E2 mesure l'effet cumule sur toute la trajectoire).

Mesures : nombre de questions, exactitude par concept -- a comparer a la
reference exacte (pi_star, "N infini") deja mesuree dans benchmark_a3.py.

Usage : python lot1_e2_cout_en_aval.py
Ecrit results/lot1_e2/{raw.csv, summary.csv, figure.pdf, run.log} + une
copie lot1_e2_figure.png a la racine (meme pattern que les experiences
precedentes).
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark_a3 import concept_accuracy
from domains.loader import load_domain_yaml
from kst_engine import Domain, simulate, simulate_mc

ROOT = Path(__file__).resolve().parent
DOMAIN_PATH = ROOT / "domains" / "piste_b.yaml"
RESULTS_DIR = ROOT / "results" / "lot1_e2"

N_VALUES = [1, 3, 5, 10, 30, 100]
MODES = ["sample_y", "sample_z"]
MODE_IDX = {"sample_y": 0, "sample_z": 1}
N_REPS = 5   # replications MC independantes par (etat, mode, N)


def run_e2(domain: Domain) -> tuple[list[dict], dict]:
    concepts = [c.name for c in domain.concepts]

    # reference exacte (pi_star, "N infini") -- une seule fois par etat,
    # deterministe (aucune approximation)
    exact_rows = []
    for seed, z in enumerate(domain.Z):
        r = simulate(domain, z, adaptive=True, seed=seed)
        exact_rows.append({
            "n_questions": r["n_questions"],
            "concept_accuracy": concept_accuracy(r["z_hat"], z, concepts),
        })
    exact_summary = {
        "n_questions_mean": float(np.mean([r["n_questions"] for r in exact_rows])),
        "concept_accuracy_mean": float(np.mean([r["concept_accuracy"] for r in exact_rows])),
    }

    records = []
    for mode in MODES:
        for n in N_VALUES:
            for seed, z in enumerate(domain.Z):
                for rep in range(N_REPS):
                    seed_seq = np.random.SeedSequence([seed, MODE_IDX[mode], n, rep])
                    run_seed = int(seed_seq.generate_state(1)[0])
                    r = simulate_mc(domain, z, n_samples=n, mode=mode, seed=run_seed)
                    records.append({
                        "mode": mode,
                        "n_samples": n,
                        "z_seed": seed,
                        "rep": rep,
                        "n_questions": r["n_questions"],
                        "concept_accuracy": concept_accuracy(r["z_hat"], z, concepts),
                    })
    return records, exact_summary


def summarize(records: list[dict]) -> dict[tuple[str, int], dict[str, float]]:
    summary = {}
    for mode in MODES:
        for n in N_VALUES:
            rows = [r for r in records if r["mode"] == mode and r["n_samples"] == n]
            n_q = np.array([r["n_questions"] for r in rows], dtype=float)
            acc = np.array([r["concept_accuracy"] for r in rows], dtype=float)
            summary[(mode, n)] = {
                "n_questions_mean": float(n_q.mean()),
                "n_questions_std": float(n_q.std()),
                "concept_accuracy_mean": float(acc.mean()),
                "n_obs": len(rows),
            }
    return summary


def write_raw_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "n_samples", "z_seed", "rep",
                                          "n_questions", "concept_accuracy"])
        w.writeheader()
        w.writerows(records)


def write_summary_csv(summary: dict[tuple[str, int], dict[str, float]],
                      exact_summary: dict[str, float], n_states: int, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "n_samples", "n_questions_moyen", "n_questions_std",
                   "exactitude_par_concept", "n_observations"])
        w.writerow(["exact (pi_star)", "inf", f"{exact_summary['n_questions_mean']:.2f}",
                   "0.00", f"{exact_summary['concept_accuracy_mean']:.3f}", n_states])
        for (mode, n), s in sorted(summary.items()):
            w.writerow([mode, n, f"{s['n_questions_mean']:.2f}", f"{s['n_questions_std']:.2f}",
                       f"{s['concept_accuracy_mean']:.3f}", s["n_obs"]])


def fig_summary(summary: dict[tuple[str, int], dict[str, float]],
                exact_summary: dict[str, float], path: Path) -> None:
    colors = {"sample_y": "#2E86AB", "sample_z": "#C0504D"}
    markers = {"sample_y": "o", "sample_z": "s"}
    labels = {"sample_y": "sample_y", "sample_z": "sample_z"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for mode in MODES:
        n_q = [summary[(mode, n)]["n_questions_mean"] for n in N_VALUES]
        acc = [summary[(mode, n)]["concept_accuracy_mean"] * 100 for n in N_VALUES]
        ax1.plot(N_VALUES, n_q, marker=markers[mode], color=colors[mode],
                linewidth=2, label=labels[mode])
        ax2.plot(N_VALUES, acc, marker=markers[mode], color=colors[mode],
                linewidth=2, label=labels[mode])

    ax1.axhline(exact_summary["n_questions_mean"], color="black", linewidth=1,
               linestyle="--", label="π* exact (N infini)")
    ax2.axhline(exact_summary["concept_accuracy_mean"] * 100, color="black", linewidth=1,
               linestyle="--", label="π* exact (N infini)")

    ax1.set_xscale("log")
    ax1.set_xlabel("N (échantillons Monte Carlo)")
    ax1.set_ylabel("Nombre moyen de questions")
    ax1.set_title("Coût en questions")

    ax2.set_xscale("log")
    ax2.set_xlabel("N (échantillons Monte Carlo)")
    ax2.set_ylabel("Exactitude par concept (%)")
    ax2.set_title("Précision du diagnostic final")
    ax2.set_ylim(0, 100)

    for ax in (ax1, ax2):
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Lot 1 / E2 — coût en aval de π̂_N sur le benchmark complet (piste B, |Z|=50)")
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


def write_run_log(path: Path, n_states: int, n_records: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)    : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande      : python lot1_e2_cout_en_aval.py\n")
        f.write(f"commit git    : {_git_commit()}\n")
        f.write(f"domaine       : {DOMAIN_PATH.relative_to(ROOT)}\n")
        f.write(f"graines       : reference exacte -- seed = index de l'etat (0..{n_states - 1}). "
               f"MC -- np.random.SeedSequence([z_seed, mode_idx, n, rep]), deterministe.\n")
        f.write(f"replications  : {N_REPS} par (etat, mode, N)\n")
        f.write(f"lignes brutes : {n_records} "
               f"({n_states} etats x {len(MODES)} modes x {len(N_VALUES)} N x {N_REPS} reps)\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domain, meta, labels = load_domain_yaml(DOMAIN_PATH)

    records, exact_summary = run_e2(domain)
    summary = summarize(records)

    print(f"exact (pi_star) : n_questions={exact_summary['n_questions_mean']:.2f}  "
         f"concept_acc={exact_summary['concept_accuracy_mean']:.1%}")
    for mode in MODES:
        print(f"\n{mode} :")
        for n in N_VALUES:
            s = summary[(mode, n)]
            print(f"  N={n:4d}  n_questions={s['n_questions_mean']:.2f}±{s['n_questions_std']:.2f}  "
                 f"concept_acc={s['concept_accuracy_mean']:.1%}")

    write_raw_csv(records, RESULTS_DIR / "raw.csv")
    write_summary_csv(summary, exact_summary, len(domain.Z), RESULTS_DIR / "summary.csv")
    fig_summary(summary, exact_summary, RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log", len(domain.Z), len(records))
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure.pdf, run.log}}")

    fig_summary(summary, exact_summary, ROOT / "lot1_e2_figure.png")
    print("ecrit (copie racine) : lot1_e2_figure.png")


if __name__ == "__main__":
    main()
