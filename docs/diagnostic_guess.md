# Diagnostic du `guess` dégénéré (Piste A, tâche A2)

> Suit `DIAGNOSTIC_GUESS_A2.md`. État : **D0 répondu, D1-D3 pas encore exécutés** (consigne : ne rien relancer avant d'avoir présenté ce constat).

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

### Constat

L'hypothèse D0 est **fortement soutenue** par les chiffres : la répétition massive de tentatives sur un même exercice (11,37 en moyenne, jusqu'à 5174) est exactement le mécanisme décrit dans `DIAGNOSTIC_GUESS_A2.md` — un étudiant qui échoue puis réussit un exercice au fil de sa pratique génère des observations contradictoires que le modèle, contraint à un seul `z` par étudiant, ne peut résoudre qu'en gonflant `guess` et/ou en gonflant `slip`.

Ceci **ne tranche pas encore** entre D0 (répétition au sein d'un exercice) et D2 (dérive sur 2012-2015) — les deux mécanismes sont structurellement identiques (plusieurs observations contradictoires d'un même `z` supposé fixe), à des échelles de temps différentes (minutes/heures pour D0, mois/années pour D2). Les deux sont probablement à l'œuvre simultanément.

---

## D1, D2, D3 — pas encore exécutés

Conformément à la consigne (« ne rien relancer avant d'avoir présenté ce constat »), aucun run EM de diagnostic n'a été lancé. En attente de validation de D0 avant de poursuivre.

---

## Scripts

- `data/diagnostics/d0_correct_definition.py` — histogramme streaming de `problem_number` (mémoire légère, pas de group-by sur (étudiant,exercice)), sondage de séquentialité en deux passes.
- `data/diagnostics/test_d0_correct_definition.py` — 6 tests sur fixtures synthétiques, dont un test de régression sur le biais du sondage à une seule passe (historique dispersé sur plusieurs chunks).

Aucun fichier de production (`domain.yaml`, `responses.parquet`) n'a été modifié.
