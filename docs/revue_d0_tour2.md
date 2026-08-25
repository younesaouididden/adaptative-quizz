# Revue de D0 — 2e tour (commit `c13ef9c`)

> Revue critique des modifications apportées après `docs/revue_d0.md` :
> commit `c13ef9c` « D0 complet : mesure decisive (R1) et test de la variante premiere-tentative ».
>
> Base de la revue : diff complet `bc51dfd..c13ef9c`, état de l'arbre de travail,
> et **mesure directe de `data/diagnostics/responses_first_attempt.parquet`**
> (lecture seule, aucun fichier modifié, aucun run relancé).

---

## Verdict

**Les ajouts R1 / R4 / R5 et la mécanique du test décisif sont validés.**

**L'interprétation du résultat ne l'est pas encore** : la conclusion
« cause partielle + biais de nouveauté » est plausible mais l'évidence produite ne la
soutient pas. Il manque un contrôle qui coûte un seul run (§ « Le contrôle manquant »).

---

## Ce qui est validé

### Les mesures répondent exactement aux remarques du 1er tour

R1, R4 et R5 sont implémentés dans le même moule mémoire-légère que l'existant, sans
duplication de logique. Vérification des effectifs de la table R1 contre les pourcentages
du premier rapport : 1 878 658 / 2 279 215 = 82,43 %, 724 799 → 31,80 %, 15 299 → 0,67 %.
**Tout concorde** — les deux tables du rapport sont cohérentes entre elles.

### Le rapport ne cache pas ce qui dérange

Deux marqueurs d'honnêteté intellectuelle :

- les 3 hausses locales de R4 sont **rapportées**, pas tues ;
- le résultat du test décisif est présenté comme un **demi-succès**, alors qu'il aurait
  été facile de titrer « D0 confirmé » en ne montrant que la colonne `guess`.

### La discipline de test tient

- `test_rapporte_la_vraie_valeur_de_problem_number_en_cas_de_hausse` : régression sur un
  piège qui aurait produit un chiffre **faux et crédible** (position dans la série triée
  au lieu de la vraie valeur de `problem_number`). Bien vu.
- `test_plateau_accepte` : une égalité n'est pas une violation de la décroissance.

### Le périmètre est respecté

`d0_first_attempt_variant.py` réutilise `calibrate_em.calibrate()` sans dupliquer l'EM, et
n'appelle pas `write_calibrated_domain` — il ne touche donc effectivement pas
`domain.yaml`. `.gitignore` couvre bien le parquet dérivé de 35 Mo.

### Précision à ajouter sur R4 (constat plus fort que celui écrit)

Les violations de décroissance commencent à `problem_number = 560`. Donc **toute la plage
qui sert de preuve (pn ≤ 100) est strictement décroissante**. C'est une affirmation plus
solide que « négligeable » — à écrire telle quelle.

---

## Le point qui bloque : l'interprétation de l'explosion de `slip`

### Mesures faites sur `responses_first_attempt.parquet`

| | valeur mesurée |
|---|---|
| étudiants | 246 659 (contre 247 307 dans le run complet — **inchangé**) |
| observations par étudiant | 9,23 |
| concepts couverts par étudiant | **1,88** |
| observations par (étudiant, concept) | médiane **2**, dont **41,8 % à une seule** |
| étudiants avec ≤ 3 observations au total | **57,7 %** |

### C1 — « Un run ~11× moins cher » est faux, et c'est révélateur

Le coût de l'EM est dominé par le E-step sur `n_étudiants × |Z|`. Or
`sufficient_statistics()` réduit les lignes à une matrice (étudiant × concept) **avant**
l'EM, et le nombre d'étudiants n'a pas bougé (246 659 vs 247 307).

**La matrice a exactement la même forme, avec des compteurs 11× plus petits dedans.**
Le coût n'a pas été divisé par 11 : c'est **l'information** qui a été divisée par 11 —
précisément ce dont un EM a besoin pour séparer `slip` de `guess`.

Avec une médiane de 3 observations binaires par étudiant, réparties sur ~2 concepts, pour
inférer un état parmi |Z| = 34 : le postérieur est essentiellement le prior. Dans ce
régime, `slip` et `guess` s'échangent presque librement sans coût de vraisemblance.

> **L'explosion de `slip` pourrait être un artefact d'identifiabilité, pas un
> « biais de nouveauté ».**

### C2 — Une lecture plus simple, qui change la conclusion

Taux de réussite marginaux mesurés sur la variante :

| Concept | taux marginal (1ère tentative) | n |
|---|---|---|
| arithmetic_base | 0,790 | 1 206 043 |
| fractions_ratios | 0,577 | 355 638 |
| geometry | 0,576 | 276 032 |
| algebra_linear | 0,575 | 173 101 |
| analytic_geometry | 0,541 | 70 435 |
| probability_statistics | 0,529 | 57 983 |
| algebra_advanced | 0,457 | 117 376 |
| calculus | 0,248 | 12 562 |
| logics | **0,235** | 8 406 |

`logics` : 23,5 % de réussite en première tentative. Le `slip = 0,699` jugé « implausible »
dit simplement qu'un maître ne réussit que 30 % du temps — sur un concept où *presque
personne* ne réussit du premier coup. Ce n'est pas absurde, c'est **cohérent avec la
donnée**. Idem `calculus` (24,8 % marginal → maître à 55 %).

Surtout : les deux jeux ne mesurent pas la même chose.

| jeu de données | ce qu'il mesure réellement |
|---|---|
| `responses.parquet` | réussir **après s'être entraîné jusqu'à la proficiency** |
| variante 1ère tentative | réussir un exercice **jamais vu** |

Or dans un quiz adaptatif — le but du PFA — chaque question est posée **une fois, à froid**.
La condition de déploiement, c'est la première tentative.

> La variante n'est peut-être pas un diagnostic biaisé : c'est peut-être **la calibration
> correcte pour l'usage visé**, et c'est `responses.parquet` qui mesure une condition qui
> n'arrivera jamais dans le quiz.

À discuter explicitement dans le rapport — c'est un argument de mémoire, pas un détail.

### C3 — Le prior n'est jamais rapporté

Pour un diagnostic dont le sujet **est** un paramètre dégénéré, la première chose à
regarder est la distribution sur `Z`. `calibrate()` la retourne, le script ne l'imprime
pas. La marginale P(concept maîtrisé) par concept, comparée entre les deux runs,
expliquerait directement le basculement des paramètres.

### C4 — Les écarts-types entre restarts ont disparu

Le rapport A2 concluait « ce n'est pas un bug de l'EM » **précisément** grâce aux
écarts-types < 0,001 sur 5 restarts. La nouvelle table ne les donne pas. Sans eux,
impossible de savoir si `slip[logics] = 0,699` est stable ou un optimum local.
C'est une régression de rigueur par rapport à A2, **sur le run qui porte toute la
conclusion**.

---

## Le contrôle manquant (action prioritaire)

Un seul run, et il tranche entre « artefact de finesse des données » et « vrai effet
première-tentative » :

> Tirer, pour chaque (étudiant, concept), **le même nombre d'observations** que dans la
> variante, mais **au hasard parmi toutes les tentatives** (pas seulement
> `problem_number == 1`), puis relancer l'EM.

- Si ce contrôle **reproduit** l'explosion de `slip` → l'effet vient de la **maigreur**
  des données, pas du filtre. Conclusion du rapport à réécrire.
- S'il **garde `guess ≈ 0,7`** → le filtre première-tentative fait bien le travail, et la
  lecture « condition de déploiement » (C2) devient l'axe principal.

**Raison d'être de ce contrôle** : aujourd'hui la comparaison n'est pas à variables
contrôlées — deux choses ont changé en même temps, le **type** d'observation *et* la
**quantité** d'information.

---

## Points d'entretien

### E1 — `data/domain.yaml` est modifié dans l'arbre de travail

Vérifié : le contenu est **md5-identique** une fois normalisé, seules les **fins de ligne**
ont changé (LF → CRLF). Aucune conséquence sémantique, mais :

- ça contredit à première vue « aucun fichier de production n'a été modifié » dès qu'on
  tape `git status` ;
- ça masquera un vrai changement plus tard.

`git checkout data/domain.yaml` suffit. Cause probable : `write_calibrated_domain` ouvre en
mode texte, donc écrit CRLF sous Windows.

### E2 — La promesse « ne touche JAMAIS domain.yaml » n'est couverte par aucun test

Elle tient uniquement parce que `write_calibrated_domain` est appelé dans le `__main__` de
`calibrate_em.py` et non dans `calibrate()`. C'est vrai mais fragile. Un test qui compare
le hash du fichier avant/après exécution de la variante verrouille ça en deux lignes.

### E3 — Déséquilibre des effectifs

`arithmetic_base` pèse **53 %** des premières tentatives (1,21 M sur 2,28 M), tandis que
`logics` (8 406) et `calculus` (12 562) reposent sur très peu d'observations. Les deux
concepts aux valeurs les plus extrêmes sont ceux qui ont le moins de données derrière.
À mentionner avant de conclure quoi que ce soit sur eux.

### E4 — Nuance de population sur R1

Les 67,8 % à `pn = 1` portent sur *toutes* les paires ; les 85,5 % à `pn = 10` sur les
seules 31,8 % qui ont répété dix fois. **Ce ne sont pas les mêmes populations.**

Les deux biais de sélection jouent ici *en faveur* de la conclusion :

- les gros répéteurs sont plutôt les élèves **en difficulté** (ce qui tire le taux vers le
  bas aux `pn` élevés) ;
- les paires à tentative unique sont plutôt des exercices **faciles réussis** (ce qui tire
  le taux vers le haut à `pn = 1`).

Donc la hausse réelle intra-paire est probablement **plus forte encore**. Une phrase
suffit, mais elle doit figurer dans le rapport.

---

## Récapitulatif des actions

| # | Action | Priorité |
|---|---|---|
| C-ctrl | Run de contrôle à information égale (échantillon aléatoire de même taille par (étudiant, concept)) | **haute** |
| C4 | Republier la table avec les écarts-types sur 5 restarts | **haute** |
| C3 | Imprimer et comparer le prior / la marginale P(maîtrise) entre les deux runs | haute |
| C2 | Discuter « 1ère tentative = condition de déploiement du quiz » dans le rapport | haute |
| C1 | Corriger la phrase « ~11× moins cher » (c'est l'information qui est divisée par 11, pas le coût) | moyenne |
| E1 | `git checkout data/domain.yaml` (fins de ligne) | moyenne |
| E4 | Ajouter la nuance de population sur R1 | moyenne |
| E3 | Mentionner le déséquilibre des effectifs avant de conclure sur `logics`/`calculus` | moyenne |
| R4+ | Écrire que la plage pn ≤ 100 est strictement décroissante | basse |
| E2 | Test de non-écriture de `domain.yaml` | basse |
