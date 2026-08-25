# Diagnostic du `guess` dégénéré (Piste A, tâche A2)

> Suit `DIAGNOSTIC_GUESS_A2.md`, complété par `docs/revue_d0.md` puis `docs/revue_d0_tour2.md` (deux revues critiques successives de ce document — le verdict a changé entre les deux, voir « Verdict révisé » ci-dessous). État : **D0 complet. Verdict : effet réel mais secondaire ; le run de contrôle pointe vers un problème d'identifiabilité (D3) plus fondamental que prévu. D1/D2/D3 pas encore formellement exécutés.**

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

**Nuance de population (E4, revue tour 2)** : les 67,8 % à `pn=1` portent sur *toutes* les paires (2 279 215) ; les 85,5 % à `pn=10` ne portent que sur les 31,8 % qui ont répété dix fois — ce ne sont pas les mêmes populations. Les deux biais de sélection plausibles jouent tous les deux *en faveur* de la lecture ci-dessus : les gros répéteurs sont plutôt des élèves en difficulté (tire le taux vers le bas aux `pn` élevés), et les paires à tentative unique sont plutôt des exercices faciles réussis du premier coup (tire le taux vers le haut à `pn=1`). La vraie hausse intra-paire est donc probablement **plus forte encore** que ce que montre la table brute.

### R4 — Décroissance de `pn_counts` sur les 26M lignes

3 hausses locales détectées, la première à `problem_number = 560` (+1 paire, sur 360 paires à ce point — 0,016 % des paires distinctes). **Sur toute la plage `problem_number ≤ 100`, qui sert de preuve à R1, la décroissance est stricte** — les seules violations se situent bien après, dans une queue extrême où les effectifs tombent à quelques centaines de paires. Ne remet pas en cause la méthode de comptage utilisée pour R1/R4/R5 ni pour le sondage de séquentialité (30/30 sur l'historique complet).

### R5 — Part des lignes en `review_mode`

2,14 % des lignes. Contributeur réel mais mineur comparé à l'effet de repratique lui-même (R1).

### R3 — Le codage indice→échec argumente *pour* D0, pas neutre

6,46 % des lignes sont forcées à `correct=false` dès qu'un indice est utilisé, même si l'étudiant savait faire. Ce codage pousse mécaniquement `slip` vers le haut et `guess` vers le bas. Observer malgré ça un `guess` aussi élevé (0,756 sur `arithmetic_base`) est un argument renforçant D0, pas un simple detail neutre.

### R6 — Précision de périmètre

Les mesures ci-dessus portent sur les **25 925 992 lignes brutes** du fichier Junyi, à distinguer des **25 895 300 lignes retenues** par `extract_responses.py` (mappées à un concept) qui alimentent réellement la calibration — écart de 0,1 %, sans conséquence pratique, mais à ne pas confondre.

---

## Le test décisif : EM sur les premières tentatives seulement

`data/diagnostics/d0_first_attempt_variant.py` reconstruit une variante des réponses limitée à `problem_number == 1` (2 277 576 lignes, contre 25 895 300 dans `responses.parquet`). **Correction (C1, revue tour 2)** : ce n'est pas « un run ~11× moins cher » — l'EM tourne sur une matrice (étudiant × concept) dont la *forme* ne change pas (246 659 étudiants contre 247 307 dans le run complet, quasi identique), seuls les compteurs à l'intérieur sont ~11× plus petits. C'est **l'information disponible pour séparer `slip` de `guess`** qui est divisée par 11, pas le coût de calcul.

### Le contrôle manquant (C-ctrl, revue tour 2)

La comparaison ci-dessus n'était pas à variables contrôlées : deux choses changent en même temps entre `responses.parquet` et la variante — le **type** d'observation (1ère tentative vs n'importe laquelle) et la **quantité** d'information (~11× moins). `data/diagnostics/d0_control_random_sample.py` isole la quantité seule : pour chaque (étudiant, concept), tirer au hasard **autant** d'observations que dans la variante, mais parmi **toutes** les tentatives (pas seulement `problem_number==1`). Techniquement : le nombre de corrects dans un tirage sans remise de taille K depuis un sac de N dont C corrects suit exactement une loi **hypergéométrique** — pas besoin de relire les 25,9M lignes individuelles, les comptes agrégés déjà calculés par `sufficient_statistics` suffisent (mathématiquement exact, pas une approximation).

`data/diagnostics/d0_full_comparison.py` recalcule les 3 runs (complet, variante, contrôle) avec écarts-types sur 5 restarts (C4) et la marginale `P(concept maîtrisé)` sous le prior calibré (C3, jamais rapportée jusqu'ici alors que le sujet du diagnostic est justement un paramètre dégénéré) :

| Concept | guess complet | guess variante | guess **contrôle** | slip complet | slip variante | slip **contrôle** |
|---|---|---|---|---|---|---|
| algebra_advanced | 0.403 | 0.227 | **0.151** | 0.208 | 0.401 | 0.266 |
| algebra_linear | 0.515 | 0.332 | **0.207** | 0.186 | 0.318 | 0.210 |
| analytic_geometry | 0.624 | 0.416 | **0.415** | 0.123 | 0.294 | 0.164 |
| arithmetic_base | 0.755 | 0.671 | **0.588** | 0.096 | 0.150 | 0.108 |
| calculus | 0.264 | 0.139 | **0.127** | 0.340 | 0.448 | 0.471 |
| fractions_ratios | 0.685 | 0.456 | **0.472** | 0.141 | 0.257 | 0.166 |
| geometry | 0.709 | 0.487 | **0.534** | 0.117 | 0.231 | 0.149 |
| logics | 0.183 | 0.119 | **0.049** | 0.451 | 0.703 | 0.688 |
| probability_statistics | 0.741 | 0.482 | **0.568** | 0.109 | 0.259 | 0.115 |

(écarts-types sur 5 restarts : tous < 0,05, souvent < 0,01 — chaque run pris individuellement est stable ; le tableau `docs/diagnostic_guess.md` original en gardait déjà la preuve pour le run complet, désormais republiée pour les 3 runs dans le script.)

### Lecture — le verdict change

**Pour `guess`, le contrôle est aussi bas ou plus bas que la variante première-tentative, sur 7 des 9 concepts** (`algebra_advanced`, `algebra_linear`, `arithmetic_base`, `calculus`, `logics` nettement ; `analytic_geometry` quasi identique). Si restreindre à la première tentative éliminait spécifiquement la contamination de pratique répétée (l'hypothèse D0), le contrôle — qui garde des répétitions, juste moins nombreuses — ne devrait **pas** reproduire cette baisse. Il la reproduit, et parfois la dépasse. **La baisse de `guess` est donc majoritairement un effet de la quantité d'information, pas du type d'observation** : avec `|Z|=34` et une médiane de 2 observations par (étudiant, concept), l'EM est mal identifié — le postérieur individuel reste proche du prior, et `slip`/`guess` peuvent s'échanger avec peu de coût de vraisemblance.

Pour `slip`, le tableau est plus nuancé : le contrôle reste **entre** le run complet et la variante pour la plupart des concepts (ex. `algebra_advanced` 0,208 → 0,266 → 0,401), ce qui laisse un résidu possible d'effet « première tentative » propre — mais pour `calculus`, `logics` et `probability_statistics`, contrôle et variante sont quasi identiques (0,471≈0,448 ; 0,688≈0,703 ; 0,115 proche de 0,109), signe que pour ces concepts spécifiques (les plus petits en effectif — voir E3 ci-dessous), l'explosion de `slip` est elle aussi essentiellement un artefact de rareté des données, pas un vrai effet de nouveauté.

**Déséquilibre des effectifs (E3, revue tour 2)** : dans la variante, `arithmetic_base` pèse à lui seul 53 % des premières tentatives (1,21M sur 2,28M), tandis que `logics` (8 406) et `calculus` (12 562) reposent sur très peu d'observations. Les deux concepts aux valeurs les plus extrêmes (`slip[logics]=0,699`, `slip[calculus]=0,448`) sont précisément ceux qui ont le moins de données derrière — à mentionner avant de tirer une conclusion ferme sur ces deux concepts spécifiquement.

**La marginale `P(concept maîtrisé)` bouge fortement d'un run à l'autre** (ex. `arithmetic_base` : 0,717 → 0,649 → 0,834), ce qui est cohérent avec un problème d'identifiabilité plutôt qu'avec trois mesures convergentes d'une même quantité stable.

### Verdict D0 — révisé

**Le mécanisme R1 (la pratique répétée génère un vrai signal d'apprentissage intra-exercice) est démontré et réel — mais il n'explique pas majoritairement pourquoi les paramètres calibrés changent entre le run complet et la variante.** Ce changement est dominé par la perte de puissance statistique/identifiabilité quand on réduit à ~1/11e des observations par étudiant, un phénomène de nature différente de D0 (ce n'est pas que les données de la variante soient "plus propres", c'est qu'il y en a trop peu pour contraindre `slip` et `guess` séparément). Autrement dit : **D0 est vrai comme mécanisme, mais le test censé le confirmer était confondu avec un effet d'échantillon, et le contrôle montre que ce second effet domine.**

Ceci relativise aussi la lecture « la variante = la bonne cible de calibration pour un quiz qui pose chaque question à froid » (C2) : c'est un argument de fond légitime pour le mémoire (la condition de déploiement d'un quiz adaptatif *est* la première tentative sur chaque question), mais le run actuel qui porte ce nom n'est pas fiable pour trancher *quelles valeurs* utiliser, puisqu'il souffre du même problème d'identifiabilité que le contrôle. Utiliser la variante telle quelle calibrerait sur du bruit, pas sur un vrai signal « première tentative ».

**Le vrai problème mis au jour n'est plus « D0 vs D1 vs D2 », c'est D3 (identifiabilité)** : avec 9 concepts, `|Z|=34`, et une médiane de ~2 observations par (étudiant, concept) dans les sous-ensembles filtrés, le modèle n'a structurellement pas assez de signal par étudiant pour séparer proprement `slip`, `guess` et la position sur `Z`. Le run complet (25,9M lignes, ~11 observations/paire en moyenne) n'a probablement pas ce problème au même degré — mais rien ne le garantit sans un test dédié.

---

## D1, D2, D3 — pas encore formellement exécutés

D0 exécuté et réinterprété grâce au run de contrôle. Le résultat pointe maintenant vers D3 (identifiabilité) comme candidat principal pour expliquer pourquoi les sous-échantillons donnent des paramètres si différents du run complet — reste à tester directement sur le run complet lui-même (ex. fixer un paramètre à une valeur alternative et mesurer la perte de vraisemblance **par étudiant**, comme le proposait `DIAGNOSTIC_GUESS_A2.md` §6). D2 (dérive temporelle) reste également untested.

---

## Scripts

- `data/diagnostics/d0_correct_definition.py` — histogramme streaming de `problem_number` (mémoire légère, pas de group-by sur (étudiant,exercice)), sondage de séquentialité en deux passes, taux de réussite par tentative (R1), part de `review_mode` (R5), vérification de décroissance (R4).
- `data/diagnostics/test_d0_correct_definition.py` — 13 tests sur fixtures synthétiques.
- `data/diagnostics/d0_first_attempt_variant.py` — reconstruit une variante des réponses limitée à la première tentative, relance l'EM dessus (réutilise `calibrate_em.py`, ne duplique pas la logique).
- `data/diagnostics/test_d0_first_attempt_variant.py` — 3 tests sur fixtures synthétiques, dont un verrouillant par hash que `domain.yaml` n'est jamais écrit (E2).
- `data/diagnostics/d0_control_random_sample.py` — construit le run de contrôle par tirage hypergéométrique (mêmes effectifs que la variante, échantillon aléatoire parmi toutes les tentatives).
- `data/diagnostics/test_d0_control_random_sample.py` — 4 tests sur fixtures synthétiques, dont une vérification de la propriété statistique (moyenne hypergéométrique conforme sur grand échantillon).
- `data/diagnostics/d0_full_comparison.py` — recalcule les 3 runs avec écarts-types et marginales pour comparaison directe.
- `data/diagnostics/test_d0_full_comparison.py` — 2 tests sur fixtures synthétiques.

Aucun fichier de production (`domain.yaml`, `responses.parquet`) n'a été modifié. `responses_first_attempt.parquet` (35 Mo, dérivé) est hors git.
