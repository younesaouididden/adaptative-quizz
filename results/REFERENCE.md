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
| Lot 1 / E1 — fidélité de π̂_N (`sample_y` vs `sample_z`) sur vraies trajectoires piste B | [`lot1_e1_fidelite_politique.py`](../lot1_e1_fidelite_politique.py) | `python lot1_e1_fidelite_politique.py` | `MASTER_SEED=0` (choix des 20 états + trajectoires) ; `np.random.SeedSequence([point_id, mode_idx, n, rep])` par réplication MC, déterministe, cf. `run.log` | [`raw.csv`](lot1_e1/raw.csv) (1 ligne/point/mode/N/réplication), [`summary.csv`](lot1_e1/summary.csv), [`figure.pdf`](lot1_e1/figure.pdf), [`run.log`](lot1_e1/run.log) |
| Lot 1 / E2 — coût en aval de π̂_N sur le benchmark complet, piste B | [`lot1_e2_cout_en_aval.py`](../lot1_e2_cout_en_aval.py) | `python lot1_e2_cout_en_aval.py` | référence exacte : `seed` = index de l'état (0–49) ; MC : `np.random.SeedSequence([z_seed, mode_idx, n, rep])`, déterministe, cf. `run.log` | [`raw.csv`](lot1_e2/raw.csv) (1 ligne/état/mode/N/réplication), [`summary.csv`](lot1_e2/summary.csv), [`figure.pdf`](lot1_e2/figure.pdf), [`run.log`](lot1_e2/run.log) |
| Lot 1 / E3 — coût de calcul exact vs MC selon la taille de Z, domaines synthétiques | [`lot1_e3_cout_calcul.py`](../lot1_e3_cout_calcul.py) | `python lot1_e3_cout_calcul.py` | `rng=np.random.default_rng(0)` fixe pour les appels MC ; `timeit.autorange` pour le nombre de répétitions (mesure, pas un tirage aléatoire), cf. `run.log` | [`raw.csv`](lot1_e3/raw.csv) (1 ligne par taille de Z et par politique), [`figure.pdf`](lot1_e3/figure.pdf), [`run.log`](lot1_e3/run.log) (contient la recommandation d'ingénierie chiffrée) |

Commit et date exacts de la dernière exécution : voir `results/<experience>/run.log` (généré à chaque run, fait foi sur ce tableau si divergence).

## À venir (plan_action_code.md)

- **Lot 1** — validation de l'approximation Monte Carlo : **fait** (1.1, E1, E2, E3 — voir tableau ci-dessus). Reste seulement 1.5 (figure `|Z|` vs nombre de concepts, cosmétique — E3 couvre déjà l'essentiel du terrain). Recommandation d'ingénierie chiffrée (livrable du lot) : sous `|Z| ≈ 1 400`, calculer l'exact (`pi_star`) ; au-delà, `sample_z` avec `N ≈ 10–30` domine (coût constant, regret déjà < 15 % à N=30 d'après E1). `sample_y` n'est jamais avantageux (même ordre de coût que l'exact, valeur seulement approchée).
- **Lot 2** — ancrage géométrique (distance de Fisher-Rao, trajectoires, information de Fisher vs `(slip, guess)`).
- **Lot 3** — robustesse à la mauvaise spécification (découplage vérité/moteur, grille `slip`×`guess`, graphe de prérequis faux, arène miroir IRT, calibration de la confiance).
- **Lot 4** — A3 sur le domaine calibré piste A (plafond relevé à 150, cf. `note_calibration.md` §6) ; test D1 (hétérogénéité) sur le seul bucket `arithmetic`.

Chacun ajoutera une ligne à ce tableau au moment où il produit son premier résultat.
