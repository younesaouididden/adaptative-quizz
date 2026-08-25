# Diagnostic du `guess` dégénéré (Piste A, tâche A2)

> Suit `DIAGNOSTIC_GUESS_A2.md`, complété par `docs/revue_d0.md` (revue critique de la première version de ce document). État : **D0 complet et tranché comme cause partielle. D1-D3 pas encore exécutés.**

---

## D0 — Définition de `correct`

### Ce que fait le code aujourd'hui

`data/extract_responses.py`, `USE_COLS = ["user_id", "exercise", "correct", "time_done"]` : la colonne `correct` du CSV brut est reprise **telle quelle** (conversion `"true"/"false"` → booléen), **sans aucune déduplication**. Chaque ligne du log — c'est-à-dire chaque instance de pratique (`problem_number`) d'un exercice par un étudiant — devient une observation indépendante dans `responses.parquet`.

Par construction Junyi (README du dataset), le champ `correct` d'une ligne signifie déjà *« première tentative correcte sur cette instance précise, faux si un indice a été demandé »* — donc la contamination par indice est déjà écartée **au niveau de la ligne**. Le vrai problème n'est pas là : c'est que **plusieurs lignes par (étudiant, exercice)** sont conservées, chacune traitée comme une observation indépendante et équivalente du même état de connaissance supposé fixe.

### Colonnes de tentatives/indices présentes dans les logs bruts

Confirmées dans `junyi_ProblemLog_original.csv` (colonnes lues pour ce diagnostic : `user_id`, `exercise`, `problem_number`, `hint_used`, `count_hints`) :

- `problem_number` : rang de la tentative de l'étudiant sur cet exercice (1 = première fois).
- `hint_used`, `count_hints` : indice demandé ou non, et combien.
- (autres colonnes du fichier, non utilisées ici : `count_attempts`, `time_taken_attempts`, `earned_proficiency`, `points_earned`.)

### Distributions mesurées (fichier entier, 25 925 992 lignes, lecture streaming)

| Mesure | Valeur |
|---|---|
| Paires (étudiant, exercice) distinctes | 2 279 215 |
| Tentatives moyennes par paire | **11,37** |
| Part des lignes avec indice utilisé | 6,46 % |
| `problem_number` maximum observé | 5 174 (un seul étudiant, un seul exercice) |

Part des paires atteignant au moins N tentatives :

| ≥ N tentatives | % des paires |
|---|---|
| 1 | 100,00 % |
| 2 | 82,43 % |
| 3 | 73,24 % |
| 5 | 65,65 % |
| 10 | 31,80 % |
| 20 | 13,29 % |
| 50 | 2,52 % |
| 100 | 0,67 % |

**Lecture** : Junyi/Khan Academy fait pratiquer un exercice jusqu'à la « proficiency » — ce n'est pas une exception, c'est la norme. Près des deux tiers des paires (étudiant, exercice) ont au moins 5 tentatives, un tiers en a au moins 10. Le pipeline actuel traite chacune de ces tentatives comme une observation indépendante du même état `z` fixe, alors qu'elles représentent typiquement une trajectoire d'apprentissage **à l'intérieur même de l'exercice** (échecs puis réussite après pratique).

### Validation de la méthode de comptage

Le comptage ci-dessus déduit le nombre de tentatives par paire du nombre de lignes à `problem_number == 1`, sans group-by coûteux (contrainte mémoire : 0,6 Go de RAM libre sur cette machine au moment du diagnostic). Cette déduction suppose que `problem_number` numérote 1..N sans trou pour chaque paire.

**Premier sondage (invalidé)** : lire un seul chunk contigu de 200 000 lignes et vérifier la séquentialité sur les paires y apparaissant plusieurs fois — 97/18 599 seulement semblaient séquentielles. **Mais le fichier est ordonné dans le temps, pas regroupé par étudiant** : un chunk contigu ne contient qu'un fragment tronqué de l'historique de chaque paire, jamais sa totalité. Le résultat ne mesurait rien de valide.

**Sondage corrigé (deux passes)** : sélectionner 30 paires candidates, puis rebalayer tout le fichier pour reconstruire leur historique *complet* (toutes leurs lignes, où qu'elles soient dans le fichier). Résultat : **30/30 séquentielles (1..N sans trou)**. La méthode de comptage est validée.

### Constat initial (avant la revue)

L'hypothèse D0 est soutenue par les chiffres bruts : la répétition massive de tentatives sur un même exercice (11,37 en moyenne, jusqu'à 5174) est exactement le mécanisme décrit dans `DIAGNOSTIC_GUESS_A2.md`. Mais `docs/revue_d0.md` a souligné, à raison, que ce constat mesure *combien* on répète, jamais *ce que ça donne* — voir R1 ci-dessous, qui referme ce trou.

---

## Suite à la revue (`docs/revue_d0.md`) : R1, R3, R4, R5, et le test décisif

### R1 — Taux de réussite par `problem_number` (la mesure décisive)

Mesuré en une passe supplémentaire (`accumulate_correctness_and_review`, même moule mémoire-légère) :

| `problem_number` | % de réussite | n |
|---|---|---|
| 1 | **67,8 %** | 2 279 215 |
| 2 | 78,6 % | 1 878 658 |
| 3 | 81,6 % | 1 669 381 |
| 5 | 85,0 % | 1 496 341 |
| 10 | 85,5 % | 724 799 |
| 20 | 84,9 % | 302 972 |
| 50 | 82,7 % | 57 336 |
| 100 | 85,5 % | 15 299 |

**Le taux grimpe de 67,8 % à ~85 % entre la 1ère et la 5e-10e tentative, puis plafonne.** C'est exactement le profil prédit par la revue : la répétition génère bel et bien des observations contradictoires (échec puis réussite) sous un même `z` supposé fixe. D0 est démontré, pas seulement plausible.

### R4 — Décroissance de `pn_counts` sur les 26M lignes

3 hausses locales détectées, la première à `problem_number = 560` (+1 paire, sur 360 paires à ce point — 0,016 % des paires distinctes). Négligeable et situé loin dans la queue extrême : ne remet pas en cause la méthode de comptage utilisée pour R1/R4/R5 ni pour le sondage de séquentialité (30/30 sur l'historique complet).

### R5 — Part des lignes en `review_mode`

2,14 % des lignes. Contributeur réel mais mineur comparé à l'effet de repratique lui-même (R1).

### R3 — Le codage indice→échec argumente *pour* D0, pas neutre

6,46 % des lignes sont forcées à `correct=false` dès qu'un indice est utilisé, même si l'étudiant savait faire. Ce codage pousse mécaniquement `slip` vers le haut et `guess` vers le bas. Observer malgré ça un `guess` aussi élevé (0,756 sur `arithmetic_base`) est un argument renforçant D0, pas un simple detail neutre.

### R6 — Précision de périmètre

Les mesures ci-dessus portent sur les **25 925 992 lignes brutes** du fichier Junyi, à distinguer des **25 895 300 lignes retenues** par `extract_responses.py` (mappées à un concept) qui alimentent réellement la calibration — écart de 0,1 %, sans conséquence pratique, mais à ne pas confondre.

---

## Le test décisif : EM sur les premières tentatives seulement

`data/diagnostics/d0_first_attempt_variant.py` reconstruit une variante des réponses limitée à `problem_number == 1` (2 277 576 lignes, contre 25 895 300 dans `responses.parquet` — un run ~11× moins cher) et relance l'EM (5 restarts, convergé).

| Concept | slip (actuel → 1ère tentative) | guess (actuel → 1ère tentative) |
|---|---|---|
| algebra_advanced | 0.208 → **0.395** | 0.404 → 0.194 |
| algebra_linear | 0.186 → **0.312** | 0.516 → 0.295 |
| analytic_geometry | 0.123 → **0.289** | 0.626 → 0.403 |
| arithmetic_base | 0.096 → **0.152** | 0.756 → 0.646 |
| calculus | 0.339 → **0.447** | 0.264 → 0.132 |
| fractions_ratios | 0.141 → **0.252** | 0.685 → 0.443 |
| geometry | 0.117 → **0.228** | 0.709 → 0.479 |
| logics | 0.451 → **0.699** | 0.183 → 0.106 |
| probability_statistics | 0.109 → **0.260** | 0.741 → 0.477 |

### Lecture

`guess` **chute nettement pour tous les concepts** (ex. `algebra_advanced` 0,404→0,194, désormais plausible) — D0 est confirmé comme cause réelle et substantielle. Mais le critère de la revue (« `guess[arithmetic_base]` sous ~0,45 ») **n'est pas atteint** : `arithmetic_base` reste à 0,646, `fractions_ratios`/`geometry`/`probability_statistics` restent entre 0,44 et 0,48 — nettement mieux, mais toujours implausible pour du hasard.

**Effet de bord non anticipé** : `slip` **augmente fortement pour tous les concepts**, parfois vers l'implausible dans l'autre sens (`logics` 0,451→0,699 : un étudiant qui maîtrise la logique échouerait alors 70 % du temps dès sa première tentative). Interprétation la plus probable : un biais de « premier contact » — la toute première fois qu'un étudiant voit un exercice, même s'il maîtrise le concept sous-jacent, l'absence de familiarité avec le format précis de CET exercice fait baisser le taux de réussite indépendamment de la maîtrise réelle. Restreindre à `problem_number==1` élimine la contamination par la pratique répétée (corrige `guess`) mais introduit une nouvelle contamination par la nouveauté (gonfle `slip`).

### Verdict D0

**D0 est une cause partielle et substantielle, pas la cause unique.** Conforme au cas « résultat intermédiaire » anticipé par la revue : combiner avec un autre diagnostic est nécessaire. La dérive temporelle (D2, sur 2012-2015) reste un candidat direct pour expliquer le residu — c'est le même mécanisme structurel (plusieurs observations contradictoires sous un `z` fixe), à une échelle de temps différente, et il n'a pas encore été testé.

---

## D1, D2, D3 — pas encore exécutés

D0 tranché (cause partielle). En attente de validation avant de lancer D2 (dérive temporelle, split premiers 20 % / derniers 20 % de chaque étudiant) ou d'explorer le biais de nouveauté mis en évidence ci-dessus.

---

## Scripts

- `data/diagnostics/d0_correct_definition.py` — histogramme streaming de `problem_number` (mémoire légère, pas de group-by sur (étudiant,exercice)), sondage de séquentialité en deux passes, taux de réussite par tentative (R1), part de `review_mode` (R5), vérification de décroissance (R4).
- `data/diagnostics/test_d0_correct_definition.py` — 13 tests sur fixtures synthétiques.
- `data/diagnostics/d0_first_attempt_variant.py` — reconstruit une variante des réponses limitée à la première tentative, relance l'EM dessus (réutilise `calibrate_em.py`, ne duplique pas la logique).
- `data/diagnostics/test_d0_first_attempt_variant.py` — 2 tests sur fixtures synthétiques.

Aucun fichier de production (`domain.yaml`, `responses.parquet`) n'a été modifié. `responses_first_attempt.parquet` (35 Mo, dérivé) est hors git.
