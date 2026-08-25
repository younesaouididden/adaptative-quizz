# Prompt de reprise — Piste A, décision en attente sur la granularité des concepts

> À coller dans une session qui a accès au repo `adaptative-quizz` (`github.com/younesaouididden/adaptative-quizz`, privé). Le contexte théorique (`PROMPT_DEMARRAGE.md`), les décisions déjà prises (`ADDENDUM_BANQUE_QUESTIONS.md`), et le détail de l'extraction Junyi15 (`PROMPT_PISTE_A.md`) sont dans le repo — ne les redemande pas, ils sont déjà tranchés.

---

## Où on en est

**Sprint 0** (moteur pur) : fait. `kst_engine.py` refactoré concept/question, 47 tests pytest, benchmark adaptatif/aléatoire avant-après commité.

**Piste A** (validation scientifique sur données réelles, dataset Junyi15) :
- **Tâche A1 (extraction)** : fait.
  - `data/extract_junyi.py` agrège 837 exercices Junyi en **9 concepts** (`arithmetic_base`, `fractions_ratios`, `algebra_linear`, `algebra_advanced`, `geometry`, `analytic_geometry`, `probability_statistics`, `calculus`, `logics`), en subdivisant les deux plus grosses areas Junyi (`arithmetic`, `algebra`) par topic, et en gardant les autres areas telles quelles.
  - Prérequis résolus par test binomial bilatéral (α=0,10) sur les conflits de direction, plus un lien éditorial documenté (`calculus -> probability_statistics`, absent des données Junyi qui ne couvrent que la proba discrète, mais mathématiquement justifié).
  - Résultat : `data/domain.yaml`, `|Z|=34`, graphe connexe et acyclique.
  - `data/extract_responses.py` a produit `data/responses.parquet` (25 895 300 lignes retenues sur 25 925 992, soit 99,9%, 247 307 étudiants) à partir des logs bruts (`junyi_ProblemLog_original.csv`, 2,6 Go, hors git).
  - 18 + 5 = 23 tests sur fixtures synthétiques, jamais sur les fichiers réels.
- **Tâche A2 (calibration EM)** : le code est fait et validé, **mais il révèle un problème de fond qui n'est pas encore tranché** (objet de ce prompt).

## Ce qui a été fait en A2

`data/calibrate_em.py` calibre `slip`/`guess` par concept et le prior sur `Z` par espérance-maximisation, sous une hypothèse de modélisation assumée explicitement :

> **Hypothèse (limite documentée, perspective future)** : le BLIM suppose un état de connaissance `z` fixe pendant tout le quiz. Les logs Junyi couvrent 2012-2015 (un étudiant progresse forcément sur une telle période). On traite tout l'historique d'un étudiant comme une seule observation de son état. Modéliser la progression dans le temps (`z_s(t)` évolutif) est une extension naturelle mais **hors périmètre du PFA** — à citer comme perspective future dans le mémoire, pas à implémenter.

Cette hypothèse permet une simplification clé : sous `z` fixe, seul le **nombre** de bonnes/mauvaises réponses par (étudiant, concept) compte, jamais leur ordre — `sufficient_statistics()` réduit donc les 25,9M réponses à une table (étudiant × concept) de compteurs avant l'EM.

Design :
- **E-step** : `gamma[s,z] = P(z | réponses de s)`, en log-espace.
- **M-step** : MLE pondérée par `gamma` pour `slip`/`guess`, moyenne pour le prior. Contrainte `slip+guess<1` imposée par **projection** (réduction proportionnelle des deux valeurs, pas un clip naïf) si violée.
- **Stabilité** : 5 initialisations aléatoires indépendantes.
- 10 tests sur fixtures synthétiques, dont un test de **récupération de paramètres** (génère des données à partir de `slip`/`guess` connus, vérifie que l'EM les retrouve à ±0,03 — le test standard de correction d'un EM).

**Bug trouvé et corrigé en cours de route** : le critère de convergence comparait la log-vraisemblance *totale* (échelle ~11 millions, sur 247k étudiants) à un seuil absolu de `1e-4` — bien trop strict à cette échelle, ce qui faisait échouer la convergence après 100 itérations alors que les paramètres étaient déjà stables. Corrigé pour comparer la log-vraisemblance **moyenne par étudiant**. Après correction : convergence atteinte en 25 itérations, résultats quasi identiques à l'essai précédent.

### Résultat calibré (5 restarts, écart-type entre parenthèses)

| Concept | slip | guess |
|---|---|---|
| algebra_advanced | 0.208 (±0.000) | 0.404 (±0.0002) |
| algebra_linear | 0.186 (±0.000) | 0.516 (±0.0002) |
| analytic_geometry | 0.123 (±0.0003) | 0.625 (±0.0009) |
| arithmetic_base | 0.096 (±0.000) | **0.756** (±0.0001) |
| calculus | 0.339 (±0.0003) | 0.264 (±0.0003) |
| fractions_ratios | 0.141 (±0.0001) | 0.685 (±0.0001) |
| geometry | 0.117 (±0.0001) | 0.709 (±0.0002) |
| logics | 0.451 (±0.0002) | 0.183 (±0.0002) |
| probability_statistics | 0.109 (±0.0002) | 0.741 (±0.0003) |

Log-vraisemblance finale ≈ -11 192 385 (5 restarts entre -11 192 385 et -11 192 398 — quasi identiques, forte présomption d'optimum global plutôt que local).

## Le problème : `guess` anormalement élevé

`guess` est censé représenter la probabilité de répondre juste **sans maîtriser le concept**. Sept concepts sur neuf affichent un `guess` entre 0,40 et 0,76 — implausible pour du hasard (0,756 pour `arithmetic_base` voudrait dire que 76% du temps, un élève qui ne maîtrise pas l'arithmétique de base répond quand même juste).

**Seuls `calculus` (0,264) et `logics` (0,183) restent plausibles** — et ce sont précisément les deux concepts les plus **petits et les plus homogènes** (10 et 5 exercices respectivement, contre 120 à 179 pour les autres).

**Hypothèse retenue** : nos 9 concepts sont des buckets larges et hétérogènes (ex. `arithmetic_base` regroupe addition, soustraction, multiplication, ordre des opérations, heure, dénombrement...). Le BLIM suppose une maîtrise "tout ou rien" par concept ; un étudiant maîtrisant 5 des 6 topics d'un bucket mais classé "non-maîtrise" en binaire continue de réussir la plupart des exercices — ce signal de maîtrise partielle est absorbé dans un `guess` gonflé plutôt que représenté correctement par le modèle.

**Ce n'est pas un bug de l'EM** : les 5 redémarrages convergent vers des valeurs quasi identiques (écarts-type < 0,001 dans la plupart des cas) — le calibrage lui-même est correct et stable. C'est une limite de la granularité du domaine par rapport à ce que le BLIM suppose.

## Décision à prendre (rien coder avant ça)

**Option 1 — Accepter et documenter.** Le calibrage EM est correct ; le domaine à 9 concepts est simplement optimiste sur l'homogénéité interne. À citer explicitement comme limite dans le mémoire, avec `calculus`/`logics` comme contre-exemple qui confirme le diagnostic (buckets homogènes → paramètres plausibles).

**Option 2 — Revoir la granularité.** Subdiviser davantage les buckets larges (se rapprocher de la borne haute 14 concepts, ou repartir des 40 topics Junyi avec la correction de connexité déjà faite lors de A1 — cf. `PROMPT_PISTE_A.md`) pour des concepts plus homogènes où le tout-ou-rien du BLIM tient mieux. Implique de refaire A1 (nouveau `domain.yaml`) puis de relancer A2 dessus.

## Première action attendue

Ne rien coder. Trancher entre les deux options ci-dessus (ou proposer une troisième). Si option 2, préciser le niveau de granularité visé et si une nouvelle décision sur les conflits de prérequis (seuil du test binomial, liens éditoriaux) doit être rouverte ou reste inchangée.
