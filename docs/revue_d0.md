# Revue de D0 — diagnostic du `guess` dégénéré

> Revue critique de `docs/diagnostic_guess.md` et de `data/diagnostics/d0_correct_definition.py`.
> Base de la revue : `git log`, `PROMPT_DEMARRAGE.md`, `PROMPT_PISTE_A.md`, `PROMPT_A2_GRANULARITE.md`,
> `data/extract_responses.py`, `data/calibrate_em.py`, `data/raw/README.txt`,
> le script D0 et ses 6 tests.
> Aucun fichier de code ni de production n'a été modifié pour produire cette revue.

---

## Verdict

**D0 est validé comme constat de mesure, pas comme conclusion.**

La mesure est propre, honnête et correctement auto-corrigée. Mais le rapport conclut
« hypothèse fortement soutenue » alors que rien dans D0 ne relie encore la répétition
des tentatives au `guess` gonflé. Il manque deux mesures très bon marché avant de
passer à D1.

---

## Ce qui tient — vérifié point par point

### 1. Le constat sur le pipeline est exact

`data/extract_responses.py` a bien `USE_COLS = ["user_id", "exercise", "correct", "time_done"]` :
aucune trace de `problem_number`, aucune déduplication. Chaque ligne de log devient bien
une observation indépendante dans `responses.parquet`.

### 2. La citation du README Junyi est exacte, au mot près

`data/raw/README.txt` :

> `correct:` *Whether the student's first attempt is correct, and the field would be false
> if any hint is requested*

### 3. L'astuce de comptage est mathématiquement juste

Si `problem_number` numérote 1..N sans trou pour chaque paire (étudiant, exercice), alors
le nombre de lignes à `problem_number == N` est **exactement** le nombre de paires ayant
au moins N tentatives. C'est élégant et ça évite le group-by qui aurait saturé la machine
(0,6 Go de RAM libre).

### 4. Les chiffres sont cohérents entre eux

Contrôle refait : la moyenne d'une variable entière positive vaut la somme des P(≥ N).
En intégrant les percentiles du tableau (100 %, 82,43 %, 73,24 %, … 0,67 %) plus une queue
au-delà de 100, on retombe sur ≈ 11,4 — soit la moyenne annoncée de **11,37**.
Les deux tableaux du rapport ne se contredisent pas.

### 5. L'auto-invalidation du premier sondage est le meilleur moment du rapport

Le premier sondage (97/18 599 séquentielles) était biaisé : le fichier est trié par temps,
pas regroupé par étudiant, donc un chunk contigu ne voit qu'un **fragment** de chaque
historique. Avoir repéré ça, jeté la mesure, refait en deux passes, puis **verrouillé le
piège dans un test de régression** (`test_reconstruit_historique_disperse_sur_plusieurs_chunks`)
est exactement la bonne discipline.

### 6. Aucun fichier de production touché

`domain.yaml` et `responses.parquet` sont intacts, conformément à la consigne.

---

## Ce qui bloque la validation

### R1 — Le script nommé `d0_correct_definition.py` ne lit jamais la colonne `correct`

C'est le point central. `USE_COLS` du diagnostic vaut
`["user_id", "exercise", "problem_number", "hint_used", "count_hints"]` : on mesure
*combien* on répète, jamais *ce que ça donne*.

La mesure décisive tient en **une passe de streaming**, avec exactement la même structure
de code : **le taux de réussite en fonction de `problem_number`**.

- Si le taux passe de ~0,45 à `problem_number = 1` à ~0,90 à `problem_number = 10`,
  D0 est **démontré**, pas seulement plausible.
- Si le taux est plat, **D0 tombe** : répéter beaucoup ne créerait alors aucune
  observation contradictoire, et il faudrait chercher le coupable ailleurs.

### R2 — L'asymétrie du symptôme n'est pas expliquée

Le rapport dit que la répétition force « `guess` et/ou `slip` » à gonfler. Or la pathologie
observée est franchement asymétrique :

| | valeurs calibrées |
|---|---|
| `guess` | 0,40 à 0,76 sur 7 concepts sur 9 |
| `slip` | 0,10 à 0,21 — plausible partout |

Une explication qui prédit « les deux gonflent » n'explique pas ce qu'on observe.

**Le mécanisme qui, lui, prédit l'asymétrie** : Junyi/Khan fait pratiquer *jusqu'à la
proficiency*. Une paire (étudiant, exercice) répétée se termine donc typiquement par une
série de réussites. Les échecs sont concentrés sur les premières tentatives — peu
nombreuses — et les réussites s'accumulent sur toutes les suivantes. Un élève faible,
classé « non-maîtrise » par le BLIM, apporte ainsi des dizaines de réponses justes sur le
même exercice. Le modèle n'a qu'un seul endroit où ranger « juste alors que non-maître » :
`guess`.

Écrit comme ça, D0 explique **ce symptôme précis**. Et c'est la mesure R1 qui le prouve.

### R3 — Le codage `hint → correct = false` joue dans l'autre sens, et renforce D0

Le rapport dit que la contamination par indice « est déjà écartée au niveau de la ligne ».
C'est plus fort que ça : **6,46 % des lignes sont forcées à `false`** même quand l'élève
savait faire. Ce codage pousse mécaniquement `slip` **vers le haut** et `guess` **vers le
bas**.

Autrement dit : `guess = 0,756` sur `arithmetic_base` est obtenu **malgré** un codage
pessimiste des réponses. C'est un argument *pour* D0, pas un point neutre — à retourner
dans ce sens dans le rapport.

### R4 — Une vérification gratuite manque sur la séquentialité

Le sondage valide 30/30 paires, mais ces 30 sont tirées des 500 000 premières lignes et
prises dans l'ordre du group-by (donc les plus petits `user_id`, les inscrits de 2012).
Le rapport signale honnêtement la limite ; il y a mieux, et c'est gratuit :

> **`pn_counts` doit être décroissant en N.**

Si la numérotation est bien 1..N par paire, il ne peut pas exister plus de paires
atteignant 8 tentatives que 7. L'histogramme est déjà en mémoire — un `assert` d'une ligne.
Ce n'est pas une condition *suffisante*, mais c'est une condition **nécessaire testée sur
les 26 M de lignes** au lieu de 30 paires.

### R5 — Deux colonnes pertinentes absentes de l'inventaire

La liste « autres colonnes non utilisées » oublie :

- **`review_mode`** — README : *exercice refait par l'étudiant après avoir obtenu la
  proficiency*. C'est directement le sujet de D0 : ce sont des lignes de révision
  post-maîtrise, versées telles quelles dans `responses.parquet` comme des observations
  ordinaires. Leur part est à chiffrer dans la même passe.
- **`suggested`** — exercice proposé par le système selon le graphe de prérequis.

### R6 — Détail de rigueur sur le périmètre des chiffres

Le diagnostic tourne sur les **25 925 992 lignes brutes**, alors que l'EM tourne sur les
**25 895 300 lignes retenues** (celles mappées à un concept). L'écart est de 0,1 %, donc
sans conséquence pratique — mais la formulation devrait préciser « fichier brut » plutôt
que de laisser croire qu'il s'agit de l'entrée réelle de la calibration.

---

## Actions recommandées avant D1

1. **Une seule passe de streaming supplémentaire**, même moule que le script actuel,
   produisant trois choses :
   - taux de réussite par `problem_number` (R1) ;
   - part des lignes en `review_mode` (R5) ;
   - assertion de décroissance de l'histogramme `pn_counts` (R4).

2. **Le test qui tranche vraiment : relancer l'EM sur les seules premières tentatives**
   (`problem_number == 1`).
   Cela fait **2,28 M de lignes au lieu de 25,9 M**, soit un run environ **dix fois moins
   cher** que celui qui a produit les paramètres actuels.
   Si `guess` sur `arithmetic_base` descend de 0,756 à une valeur plausible, D0 est tranché
   et **l'option 2 du prompt A2 (revoir la granularité, donc refaire A1) devient
   probablement inutile**.

C'est le vrai enjeu de D0 : bien mené, il peut vous éviter de reconstruire `domain.yaml`.

### Sur la séparation D0 / D2

Le rapport a raison de dire que D0 (répétition intra-exercice) et D2 (dérive 2012-2015)
sont structurellement identiques. Mais le run « premières tentatives seulement » les sépare
déjà en partie : il **supprime D0 et laisse D2 intact**.

- Si `guess` chute nettement → D0 dominait.
- S'il bouge peu → c'est la dérive temporelle qu'il faut regarder.
