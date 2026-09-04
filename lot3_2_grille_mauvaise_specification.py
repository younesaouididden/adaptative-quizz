"""Lot 3.2 + 3.5 -- grille de mauvaise specification et calibration de la
confiance (plan_action_code.md).

Le moteur (KST et IRT) suppose piste B : slip=0.10, guess=0.25 (fige dans
domains/piste_b.yaml, jamais modifie ici). La VERITE qui genere les
reponses varie sur une grille slip x guess, decouplee du moteur via
verite_slip/verite_guess (Lot 3.1). Cas dangereux attendu par le plan :
un monde plus bruite (guess reel plus eleve) que ce que le moteur croit le
rend trop confiant et le fait s'arreter trop tot -- la cellule guess=0,70
correspond au regime effectivement mesure en piste A (calibration EM,
data/domain.yaml).

3.5 (calibration de la confiance) est mesuree sur les memes runs, pour la
politique adaptative KST (confiance = p.max() a l'arret, le seul objet
directement comparable a une probabilite calibree parmi les trois
politiques) : diagramme de fiabilite, confiance annoncee vs proportion
reelle d'etats exactement corrects, pour la cellule bien specifiee
(slip=0.10, guess=0.25) contre la cellule regime piste A (slip=0.10,
guess=0.70).

Protocole commun (plan) : comparaison appariee, memes etats vrais et memes
graines pour les trois politiques dans chaque cellule, 50 etats x 20
repetitions, ecart-type reporte.

Usage : python lot3_2_grille_mauvaise_specification.py
Ecrit results/lot3_2_grille/{raw.csv, summary.csv, figure_degradation.pdf,
figure_calibration.pdf, run.log} + copies PNG a la racine.
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
from irt_baseline import simulate_irt
from kst_engine import Domain, simulate

ROOT = Path(__file__).resolve().parent
DOMAIN_PATH = ROOT / "domains" / "piste_b.yaml"
RESULTS_DIR = ROOT / "results" / "lot3_2_grille"

SLIP_GRID = [0.05, 0.10, 0.20, 0.30]
GUESS_GRID = [0.25, 0.40, 0.55, 0.70]
N_REPS = 20
MOTEUR_SLIP, MOTEUR_GUESS = 0.10, 0.25   # fige dans domains/piste_b.yaml

CALIBRATION_CELLS = [(0.10, 0.25, "bien spécifié (guess=0,25)"),
                     (0.10, 0.70, "régime piste A (guess=0,70)")]


def run_grid(domain: Domain, meta: list[dict]) -> list[dict]:
    concepts = [c.name for c in domain.concepts]
    records = []
    for vs in SLIP_GRID:
        for vg in GUESS_GRID:
            for seed, z in enumerate(domain.Z):
                for rep in range(N_REPS):
                    seed_seq = np.random.SeedSequence([seed, rep, int(vs * 1000), int(vg * 1000)])
                    run_seed = int(seed_seq.generate_state(1)[0])

                    r_kst = simulate(domain, z, adaptive=True, seed=run_seed,
                                    verite_slip=vs, verite_guess=vg)
                    r_rand = simulate(domain, z, adaptive=False, seed=run_seed,
                                     verite_slip=vs, verite_guess=vg)
                    r_irt = simulate_irt(domain, meta, z, seed=run_seed,
                                        verite_slip=vs, verite_guess=vg)

                    for policy, r in (("kst_adaptatif", r_kst), ("aleatoire", r_rand),
                                     ("cat_irt", r_irt)):
                        row = {
                            "verite_slip": vs, "verite_guess": vg,
                            "z_seed": seed, "rep": rep, "policy": policy,
                            "n_questions": r["n_questions"],
                            "correct_diagnosis": r["correct_diagnosis"],
                            "concept_accuracy": concept_accuracy(r["z_hat"], z, concepts),
                            "confidence": r.get("confidence"),   # None pour cat_irt
                        }
                        records.append(row)
    return records


def summarize(records: list[dict]) -> dict[tuple[float, float, str], dict[str, float]]:
    summary = {}
    for vs in SLIP_GRID:
        for vg in GUESS_GRID:
            for policy in ("kst_adaptatif", "aleatoire", "cat_irt"):
                rows = [r for r in records if r["verite_slip"] == vs
                       and r["verite_guess"] == vg and r["policy"] == policy]
                n_q = np.array([r["n_questions"] for r in rows], dtype=float)
                exact = np.array([r["correct_diagnosis"] for r in rows], dtype=float)
                acc = np.array([r["concept_accuracy"] for r in rows], dtype=float)
                summary[(vs, vg, policy)] = {
                    "n_questions_mean": float(n_q.mean()), "n_questions_std": float(n_q.std()),
                    "exact_mean": float(exact.mean()),
                    "concept_acc_mean": float(acc.mean()), "concept_acc_std": float(acc.std()),
                }
    return summary


def write_raw_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["verite_slip", "verite_guess", "z_seed", "rep",
                                          "policy", "n_questions", "correct_diagnosis",
                                          "concept_accuracy", "confidence"])
        w.writeheader()
        w.writerows(records)


def write_summary_csv(summary: dict[tuple[float, float, str], dict[str, float]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["verite_slip", "verite_guess", "policy", "n_questions_moyen",
                   "n_questions_std", "exactitude_etat_exact", "exactitude_par_concept",
                   "exactitude_par_concept_std"])
        for (vs, vg, policy), s in sorted(summary.items()):
            w.writerow([vs, vg, policy, f"{s['n_questions_mean']:.2f}", f"{s['n_questions_std']:.2f}",
                       f"{s['exact_mean']:.3f}", f"{s['concept_acc_mean']:.3f}",
                       f"{s['concept_acc_std']:.3f}"])


def fig_degradation(summary: dict[tuple[float, float, str], dict[str, float]], path: Path,
                    moteur_slip: float = MOTEUR_SLIP, moteur_guess: float = MOTEUR_GUESS,
                    title: str = "Lot 3.2 — dégradation de l'exactitude sous mauvaise spécification"
                    ) -> None:
    """Heatmap slip x guess de l'exactitude par concept, pour les trois
    politiques -- montre l'asymetrie attendue (guess reel > guess suppose
    degrade bien plus que slip reel > slip suppose). moteur_slip/guess
    parametrables pour reutilisation par le Lot 3.6 (moteur prudent)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.subplots_adjust(top=0.72, bottom=0.16, left=0.05, right=0.90, wspace=0.35)
    policies = ["kst_adaptatif", "aleatoire", "cat_irt"]
    titles = {"kst_adaptatif": "KST adaptatif", "aleatoire": "Aléatoire", "cat_irt": "CAT-IRT"}

    for ax, policy in zip(axes, policies):
        grid = np.array([[summary[(vs, vg, policy)]["concept_acc_mean"] * 100
                         for vg in GUESS_GRID] for vs in SLIP_GRID])
        im = ax.imshow(grid, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
        ax.set_xticks(range(len(GUESS_GRID)))
        ax.set_xticklabels([f"{g:.2f}" for g in GUESS_GRID])
        ax.set_yticks(range(len(SLIP_GRID)))
        ax.set_yticklabels([f"{s:.2f}" for s in SLIP_GRID])
        ax.set_xlabel("guess réel (vérité)")
        ax.set_ylabel("slip réel (vérité)")
        ax.set_title(titles[policy])
        for i in range(len(SLIP_GRID)):
            for j in range(len(GUESS_GRID)):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=9)
        # marque la cellule "moteur" (bien specifiee)
        if moteur_slip in SLIP_GRID and moteur_guess in GUESS_GRID:
            mi = SLIP_GRID.index(moteur_slip)
            mj = GUESS_GRID.index(moteur_guess)
            ax.add_patch(plt.Rectangle((mj - 0.5, mi - 0.5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=2.5))

    cbar_ax = fig.add_axes((0.92, 0.16, 0.015, 0.56))
    fig.colorbar(im, cax=cbar_ax, label="Exactitude par concept (%)")
    fig.suptitle(f"{title}\n(cadre noir = cellule bien spécifiée, "
                f"moteur : slip={moteur_slip:.2f}/guess={moteur_guess:.2f})", y=0.98)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_calibration(records: list[dict], path: Path) -> None:
    """Diagramme de fiabilite (3.5) : confiance annoncee vs proportion
    reelle d'etats exactement corrects, politique KST adaptative, pour la
    cellule bien specifiee et la cellule regime piste A."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    colors = {"bien spécifié (guess=0,25)": "#2E86AB", "régime piste A (guess=0,70)": "#C0504D"}

    for vs, vg, label in CALIBRATION_CELLS:
        rows = [r for r in records if r["policy"] == "kst_adaptatif"
               and r["verite_slip"] == vs and r["verite_guess"] == vg]
        conf = np.array([r["confidence"] for r in rows])
        correct = np.array([r["correct_diagnosis"] for r in rows], dtype=float)

        bins = np.linspace(conf.min(), 1.0, 9)
        bin_idx = np.digitize(conf, bins)
        xs, ys, ns = [], [], []
        for b in range(1, len(bins) + 1):
            mask = bin_idx == b
            if mask.sum() < 5:
                continue
            xs.append(conf[mask].mean())
            ys.append(correct[mask].mean())
            ns.append(int(mask.sum()))
        ax.plot(xs, ys, marker="o", color=colors[label], linewidth=2,
               label=f"{label} (n={sum(ns)})")

    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label="calibration parfaite")
    ax.set_xlabel("Confiance annoncée à l'arrêt (max p(z))")
    ax.set_ylabel("Proportion réelle d'états exactement corrects")
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Lot 3.5 — diagramme de fiabilité (politique KST adaptative)\n"
                "sous mauvaise spécification, \"90% sûr\" ne veut plus dire \"9 fois sur 10\"")
    ax.legend(fontsize=9)
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


def write_run_log(path: Path, n_records: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)   : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande     : python lot3_2_grille_mauvaise_specification.py\n")
        f.write(f"commit git   : {_git_commit()}\n")
        f.write(f"domaine      : {DOMAIN_PATH.relative_to(ROOT)} "
               f"(moteur fige : slip={MOTEUR_SLIP}, guess={MOTEUR_GUESS})\n")
        f.write(f"grille verite: slip in {SLIP_GRID} x guess in {GUESS_GRID} "
               f"({len(SLIP_GRID) * len(GUESS_GRID)} cellules)\n")
        f.write(f"graines      : np.random.SeedSequence([z_seed, rep, "
               f"int(verite_slip*1000), int(verite_guess*1000)]), deterministe\n")
        f.write(f"replications : {N_REPS} par (etat, cellule, politique)\n")
        f.write(f"lignes brutes: {n_records}\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    domain, meta, labels = load_domain_yaml(DOMAIN_PATH)

    records = run_grid(domain, meta)
    summary = summarize(records)

    print(f"{len(SLIP_GRID) * len(GUESS_GRID)} cellules, {N_REPS} reps/etat, "
         f"{len(domain.Z)} etats -> {len(records)} lignes")
    print("\nExactitude par concept (KST adaptatif), grille slip x guess :")
    print("slip\\guess  " + "  ".join(f"{g:.2f}" for g in GUESS_GRID))
    for vs in SLIP_GRID:
        row = [f"{summary[(vs, vg, 'kst_adaptatif')]['concept_acc_mean']:.1%}" for vg in GUESS_GRID]
        print(f"{vs:.2f}        " + "  ".join(row))

    write_raw_csv(records, RESULTS_DIR / "raw.csv")
    write_summary_csv(summary, RESULTS_DIR / "summary.csv")
    fig_degradation(summary, RESULTS_DIR / "figure_degradation.pdf")
    fig_calibration(records, RESULTS_DIR / "figure_calibration.pdf")
    write_run_log(RESULTS_DIR / "run.log", len(records))
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure_degradation.pdf, "
         f"figure_calibration.pdf, run.log}}")

    fig_degradation(summary, ROOT / "lot3_2_degradation_figure.png")
    fig_calibration(records, ROOT / "lot3_5_calibration_figure.png")
    print("ecrit (copies racine) : lot3_2_degradation_figure.png, lot3_5_calibration_figure.png")


if __name__ == "__main__":
    main()
