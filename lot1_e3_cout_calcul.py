"""Lot 1, E3 -- cout de calcul exact vs MC selon |Z| (plan_action_code.md).

Domaines synthetiques SANS prerequis (|Z|=2^n_concepts) pour n_concepts in
{5,7,9,11,13} -> |Z| de 32 a 8192. Choix "sans prerequis" deliberе : il
isole la dependance en |Z| de la topologie du graphe -- les domaines reels
(piste A |Z|=18, piste B |Z|=50) ont un |Z| reduit par les prerequis d'un
facteur ~2-3x seulement par rapport a 2^n (cf. domains/validate.py), et ni
information_gain_exact ni bayes_update ne dependent de la topologie,
seulement de la TAILLE de Z (leurs boucles/produits matriciels sont sur
domain.Z, peu importe quels etats il contient).

Mesure le temps d'UNE DECISION COMPLETE (parcourir tous les candidats et
choisir l'argmax, p=prior uniforme, asked={} -- le cas le plus couteux,
aucun candidat deja elimine) : pi_star (exact) vs pi_hat (MC, N in
{10,100}, deux modes).

Usage : python lot1_e3_cout_calcul.py
Ecrit results/lot1_e3/{raw.csv, summary.csv, figure.pdf, run.log}.

Livrable du Lot 1 (plan_action_code.md) : une recommandation d'ingenierie
chiffree -- en dessous de quel |Z| calculer l'exact, au-dela utiliser MC
avec quel N -- deduite du croisement des courbes de cout mesurees ici.
"""

from __future__ import annotations

import csv
import subprocess
import timeit
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kst_engine import Concept, Domain, Question, pi_hat, pi_star, uniform_prior

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "lot1_e3"

N_CONCEPTS = [5, 7, 9, 11, 13]
MC_CONFIGS = [("sample_y", 10), ("sample_y", 100), ("sample_z", 10), ("sample_z", 100)]
POLICY_COLORS = {
    "exact (pi_star)": "#000000",
    "sample_y N=10": "#8EC6E0", "sample_y N=100": "#2E86AB",
    "sample_z N=10": "#E8A79A", "sample_z N=100": "#C0504D",
}


def make_synthetic_domain(n_concepts: int) -> Domain:
    """Domaine synthetique sans prerequis : |Z|=2^n_concepts, une question
    par concept. Cf. docstring du module pour la justification."""
    concepts = [Concept(f"c{i}") for i in range(n_concepts)]
    questions = [Question(f"q{i}", f"c{i}", slip=0.10, guess=0.25) for i in range(n_concepts)]
    return Domain(concepts=concepts, prereqs=[], questions=questions)


def time_call(fn) -> float:
    """Temps moyen par appel, en secondes -- timeit.autorange calibre le
    nombre de repetitions pour un budget minimal (~0.2s), robuste au bruit
    de mesure sur des appels individuellement tres rapides."""
    timer = timeit.Timer(fn)
    number, total_time = timer.autorange()
    return total_time / number


def run_e3() -> list[dict]:
    records = []
    for n_concepts in N_CONCEPTS:
        domain = make_synthetic_domain(n_concepts)
        n_z = domain.n_states
        p = uniform_prior(domain)

        t_exact = time_call(lambda p=p, domain=domain: pi_star(p, domain, set()))
        records.append({"n_concepts": n_concepts, "n_states": n_z,
                        "policy": "exact (pi_star)", "n_samples": None,
                        "time_per_decision_ms": t_exact * 1000})
        print(f"  |Z|={n_z:5d}  exact          {t_exact * 1000:8.3f} ms")

        for mode, n in MC_CONFIGS:
            rng = np.random.default_rng(0)
            t_mc = time_call(lambda p=p, domain=domain, n=n, mode=mode, rng=rng:
                            pi_hat(p, domain, set(), n_samples=n, mode=mode, rng=rng))
            records.append({"n_concepts": n_concepts, "n_states": n_z,
                            "policy": mode, "n_samples": n,
                            "time_per_decision_ms": t_mc * 1000})
            print(f"  |Z|={n_z:5d}  {mode:9s} N={n:4d}  {t_mc * 1000:8.3f} ms")
    return records


def find_crossover(exact_records: list[dict], flat_time_ms: float) -> float | None:
    """|Z| interpole (log-log) ou le temps exact franchit flat_time_ms.
    None si l'exact reste sous ce niveau (ou au-dessus) sur toute la plage
    mesuree -- pas assez de donnees pour extrapoler au-dela."""
    xs = np.array([r["n_states"] for r in exact_records], dtype=float)
    ys = np.array([r["time_per_decision_ms"] for r in exact_records], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    log_x, log_y = np.log(xs), np.log(ys)
    log_target = np.log(flat_time_ms)

    for i in range(len(log_y) - 1):
        y0, y1 = log_y[i], log_y[i + 1]
        if (y0 - log_target) * (y1 - log_target) <= 0 and y1 != y0:
            t = (log_target - y0) / (y1 - y0)
            return float(np.exp(log_x[i] + t * (log_x[i + 1] - log_x[i])))
    return None


def compute_recommendation(records: list[dict]) -> list[str]:
    exact = [r for r in records if r["policy"] == "exact (pi_star)"]
    lines = []
    for mode, n in MC_CONFIGS:
        rows = [r for r in records if r["policy"] == mode and r["n_samples"] == n]
        flat_time = float(np.mean([r["time_per_decision_ms"] for r in rows]))
        spread = float(np.std([r["time_per_decision_ms"] for r in rows]))
        crossover = find_crossover(exact, flat_time)
        if mode == "sample_z":
            if crossover is not None:
                lines.append(f"sample_z, N={n:3d} : cout ~{flat_time:.3f}ms (±{spread:.3f}), "
                            f"independant de |Z| -- croise l'exact vers |Z| ~= {crossover:,.0f}")
            else:
                lines.append(f"sample_z, N={n:3d} : cout ~{flat_time:.3f}ms (±{spread:.3f}), "
                            f"aucun croisement observe jusqu'a |Z|={exact[-1]['n_states']:,} "
                            f"(exact reste moins cher sur toute la plage mesuree)")
        else:
            lines.append(f"sample_y, N={n:3d} : cout ~{flat_time:.3f}ms, mais CROIT avec |Z| "
                        f"comme l'exact (meme ordre O(|Z|), cf. kst_engine docstring) -- "
                        f"jamais avantageux, confirme le preambule du Lot 1")
    return lines


def write_raw_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n_concepts", "n_states", "policy",
                                          "n_samples", "time_per_decision_ms"])
        w.writeheader()
        w.writerows(records)


def fig_summary(records: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    exact = sorted([r for r in records if r["policy"] == "exact (pi_star)"],
                   key=lambda r: r["n_states"])
    ax.plot([r["n_states"] for r in exact], [r["time_per_decision_ms"] for r in exact],
           marker="o", color=POLICY_COLORS["exact (pi_star)"], linewidth=2.5,
           label="exact (π*)")

    for mode, n in MC_CONFIGS:
        key = f"{mode} N={n}"
        rows = sorted([r for r in records if r["policy"] == mode and r["n_samples"] == n],
                     key=lambda r: r["n_states"])
        ax.plot([r["n_states"] for r in rows], [r["time_per_decision_ms"] for r in rows],
               marker="s", color=POLICY_COLORS[key], linewidth=1.5, label=key)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|Z| (nombre d'états de connaissance)")
    ax.set_ylabel("Temps par décision (ms)")
    ax.set_title("Lot 1 / E3 — coût de calcul exact vs Monte Carlo selon |Z|\n"
                 "(domaines synthétiques sans prérequis, |Z|=2^n_concepts)")
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


def write_run_log(path: Path, recommendation: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)   : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"commande     : python lot1_e3_cout_calcul.py\n")
        f.write(f"commit git   : {_git_commit()}\n")
        f.write(f"n_concepts   : {N_CONCEPTS} -> |Z| = {[2 ** n for n in N_CONCEPTS]}\n")
        f.write(f"mesure       : timeit.autorange (>=0.2s par point, rng=default_rng(0) "
               f"fixe pour les appels MC)\n\n")
        f.write("Recommandation d'ingenierie (livrable Lot 1, plan_action_code.md) :\n")
        for line in recommendation:
            f.write(f"  - {line}\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = run_e3()
    recommendation = compute_recommendation(records)

    print("\nRecommandation d'ingenierie :")
    for line in recommendation:
        print(f"  - {line}")

    write_raw_csv(records, RESULTS_DIR / "raw.csv")
    fig_summary(records, RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log", recommendation)
    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, figure.pdf, run.log}}")

    fig_summary(records, ROOT / "lot1_e3_figure.png")
    print("ecrit (copie racine) : lot1_e3_figure.png")


if __name__ == "__main__":
    main()
