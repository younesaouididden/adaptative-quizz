"""
Visualisations matplotlib du refactor concept/question.

Script separe de kst_engine.py pour garder le moteur sans dependance hors
numpy (design deliberement choisi dans le PFA). Produit deux figures :

  1. barres groupees  : nb moyen de questions (adaptatif vs aleatoire),
                        avant (1 question/concept) vs apres (plusieurs) le
                        refactor -- montre que l'algo n'avait rien a
                        optimiser avant, et beaucoup apres.
  2. trace d'entropie : H(p_t) au fil des questions pour un seul etudiant
                        simule, adaptatif vs aleatoire -- montre le
                        mecanisme (chapitre 6-7) plutot que l'agregat.
"""

import numpy as np
import matplotlib.pyplot as plt

from kst_engine import make_demo_domain, simulate, entropy, uniform_prior

OUT_DIR = "."


def benchmark(domain, adaptive):
    n_q, acc = [], []
    for seed, z in enumerate(domain.Z):
        r = simulate(domain, z, adaptive=adaptive, seed=seed)
        n_q.append(r["n_questions"])
        acc.append(r["correct_diagnosis"])
    return np.mean(n_q), np.mean(acc)


def fig_before_after():
    bank_sizes = [1, 3]
    results = {}
    for n in bank_sizes:
        dom = make_demo_domain(questions_per_concept=n)
        results[n] = {
            "adaptatif": benchmark(dom, True),
            "aleatoire": benchmark(dom, False),
        }

    labels = ["avant\n(1 question/concept)", "apres\n(3 questions/concept)"]
    adapt_vals = [results[n]["adaptatif"][0] for n in bank_sizes]
    rand_vals = [results[n]["aleatoire"][0] for n in bank_sizes]

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w/2, adapt_vals, w, label="adaptatif (π*, gain d'information)",
               color="#2E86AB")
    b2 = ax.bar(x + w/2, rand_vals, w, label="aleatoire",
               color="#C0C0C0")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=10)

    ax.set_ylabel("Nombre moyen de questions posees")
    ax.set_title("Effet du refactor concept/question\n"
                 "(moyenne sur tous les etats z in Z du domaine jouet, 8 concepts)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = f"{OUT_DIR}/refactor_avant_apres.png"
    fig.savefig(path, dpi=150)
    print(f"ecrit : {path}")
    return results


def fig_entropy_trace():
    dom = make_demo_domain(questions_per_concept=3)
    # etat "moyen" : ni trivial (vide) ni total, pour illustrer un vrai diagnostic
    z_true = sorted(dom.Z, key=len)[len(dom.Z) // 2]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"adaptatif": "#2E86AB", "aleatoire": "#C0C0C0"}
    plot_labels = {"adaptatif": "adaptatif (π*)", "aleatoire": "aleatoire"}
    for label, adaptive in (("adaptatif", True), ("aleatoire", False)):
        r = simulate(dom, z_true, adaptive=adaptive, seed=0)
        trace = r["entropy_trace"]
        ax.plot(range(len(trace)), trace, marker="o", markersize=3,
               label=f"{plot_labels[label]} ({r['n_questions']} questions)",
               color=colors[label], linewidth=2)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Question posee (t)")
    ax.set_ylabel("Entropie H(p_t) [bits]")
    ax.set_title(f"Reduction de l'incertitude sur Delta(Z)\n"
                 f"z_true = {{{', '.join(sorted(z_true)) or 'etat vide'}}}")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = f"{OUT_DIR}/entropy_trace.png"
    fig.savefig(path, dpi=150)
    print(f"ecrit : {path}")


if __name__ == "__main__":
    results = fig_before_after()
    fig_entropy_trace()

    print("\nResume chiffre :")
    for n, r in results.items():
        gap = r["aleatoire"][0] - r["adaptatif"][0]
        pct = 100 * gap / r["aleatoire"][0]
        print(f"  {n} question(s)/concept : adaptatif={r['adaptatif'][0]:.2f}, "
              f"aleatoire={r['aleatoire'][0]:.2f}  -> gain de {pct:.0f}%")
