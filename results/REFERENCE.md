# Reference des resultats (Lot 0, `plan_action_code.md`)

Associe chaque figure et chaque chiffre du rapport a son script, sa graine,
son commit git et sa date de generation. Regle : un script = une experience
= une commande reproductible. Chaque experience ecrit son propre
`results/<experience>/run.log` (commit, date, graines) au moment ou elle
tourne -- ce tableau en est l'index, a tenir a jour a chaque nouvelle
experience (Lots 1 a 4).

| Experience | Script | Commande | Graines | Sorties |
|---|---|---|---|---|
| Benchmark A3 — adaptatif / aléatoire / CAT-IRT sur piste B | [`benchmark_a3.py`](../benchmark_a3.py) | `python benchmark_a3.py` | `seed` = index de l'état dans `domain.Z` (0–49), déterministe, cf. `run.log` | [`raw.csv`](benchmark_a3/raw.csv) (1 ligne/politique/état), [`summary.csv`](benchmark_a3/summary.csv), [`figure.pdf`](benchmark_a3/figure.pdf), [`run.log`](benchmark_a3/run.log) |

Commit et date exacts de la dernière exécution : voir `results/<experience>/run.log` (généré à chaque run, fait foi sur ce tableau si divergence).

## À venir (plan_action_code.md)

- **Lot 1** — validation de l'approximation Monte Carlo (`information_gain_mc` vs `information_gain_exact`) : E1 fidélité de politique, E2 coût en aval, E3 coût de calcul vs `|Z|`.
- **Lot 2** — ancrage géométrique (distance de Fisher-Rao, trajectoires, information de Fisher vs `(slip, guess)`).
- **Lot 3** — robustesse à la mauvaise spécification (découplage vérité/moteur, grille `slip`×`guess`, graphe de prérequis faux, arène miroir IRT, calibration de la confiance).
- **Lot 4** — A3 sur le domaine calibré piste A (plafond relevé à 150, cf. `note_calibration.md` §6) ; test D1 (hétérogénéité) sur le seul bucket `arithmetic`.

Chacun ajoutera une ligne à ce tableau au moment où il produit son premier résultat.
