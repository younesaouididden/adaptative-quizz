# Contexte complet — Plateforme de quiz adaptatif (PFA UM6P)

> Document de passation pour reprendre le travail dans une nouvelle session. Lis-le en entier avant de proposer quoi que ce soit — la plupart des décisions "évidentes" ont déjà été discutées et tranchées, souvent après un détour qui a mal tourné. Ne les rouvre pas sans que l'utilisateur le demande explicitement.

---

## 1. Le projet, en une phrase

PFA (projet de fin d'année) à l'UM6P Vanguard Center/CMSIS : une plateforme de quiz adaptatif fondée sur la **Knowledge Space Theory** (KST), qui infère l'état de connaissance d'un étudiant par sélection de questions maximisant le gain d'information, et met à jour une croyance bayésienne (BLIM) à chaque réponse — pas un score, un **diagnostic par concept**.

- **Encadrant** : Ahmed Ratnani (monographie *Foundations of Learning Systems Theory*), avec S. Ibnjaa, S. Kharou, A. Zahir.
- **Repo** : [github.com/younesaouididden/adaptative-quizz](https://github.com/younesaouididden/adaptative-quizz) (privé). Code local : `C:\Users\HP\PycharmProjects\pfa-quiz-adaptatif\`.
- **Contrainte actuelle** : deadline de stage proche → priorité à une version qui marche, on améliore après.

Fichiers de référence théorique/architecture dans le repo, **déjà lus et intégrés, ne pas redemander** :
- `PROMPT_DEMARRAGE.md` — théorie (4 couches KST), décisions d'architecture (KST discret, FastAPI+Next.js visés, sélection gloutonne).
- `ADDENDUM_BANQUE_QUESTIONS.md` — deux pistes disjointes : **Piste A** (calibration scientifique sur données réelles Junyi15) et **Piste B** (banque de démo avec vraies questions, en pause).
- `PROMPT_PISTE_A.md` — détail de l'extraction Junyi15 (source, pièges).
- `PROMPT_A2_GRANULARITE.md` — historique de la décision de granularité (partiellement dépassé, voir §4).

---

## 2. Ce qui existe et fonctionne (100 tests passent, `python -m pytest -q` à la racine)

### Moteur (`kst_engine.py`)
Implémente les 4 couches KST : `build_knowledge_space` (chapitre 2), `Domain`/`Concept`/`Question` avec BLIM (chapitre 3), `bayes_update`/`entropy`/`information_gain_exact`/`pi_star` (anciennement `select_next`, renommé Lot 5)/`should_stop` (chapitres 6-7). Concept et Question sont **séparés** (refactor fait tôt dans le projet) : plusieurs questions par concept, chacune avec son propre `slip`/`guess`. 47+ tests, cas d'or de la monographie (0,500→0,818) vérifié.

### Piste A — pipeline complet sur données réelles (Junyi Academy, dataset Junyi15)
1. **`data/extract_junyi.py`** — construit `data/domain.yaml` (concepts + prérequis) depuis `junyi_Exercise_table.csv`. Résout les conflits de direction de prérequis par test binomial bilatéral (α=0,10, appliqué uniformément).
2. **`data/extract_responses.py`** — construit `data/responses.parquet` depuis les logs bruts (`junyi_ProblemLog_original.csv`, 26M lignes, lecture par chunks — machine à ~0,6-8 Go de RAM libre, toujours traiter les gros fichiers en streaming).
3. **`data/calibrate_em.py`** — calibre `slip`/`guess` par concept et le prior sur `Z` par espérance-maximisation (chapitre 3), sous l'hypothèse **`z` fixe par étudiant sur toute la période 2012-2015** (limite assumée et documentée — modéliser la progression dans le temps est une perspective future, hors périmètre du PFA).

`data/raw/` (données brutes Junyi15, ~1,2 Go) et `data/responses.parquet` (~300 Mo) sont **hors git** (`.gitignore`) — à re-télécharger/reconstruire si absents (voir `PROMPT_PISTE_A.md` pour la source exacte).

### Le domaine actuel — V1 SIMPLIFIÉE (important, lit avant de proposer une "amélioration")

Après une saga de diagnostic (§4), le domaine a été **volontairement réduit à 5 concepts** pour prioriser une version qui marche avant la deadline :

```yaml
concepts: arithmetic, algebra, geometry, analytic_geometry, probability_statistics
prerequisites: [algebra, analytic_geometry], [arithmetic, geometry]
|Z| = 18
```

slip/guess calibrés (voir `data/domain.yaml` pour les valeurs exactes) — **`guess` reste élevé (0,50-0,75) pour la plupart des concepts**. Ce n'est PAS un bug : diagnostiqué en profondeur (§4), c'est une limite structurelle du jeu de données complet (buckets de concepts hétérogènes), documentée et acceptée pour cette V1 ("Option 1 — accepter et documenter" du tout premier arbitrage de granularité).

### App de démonstration (`app.py`, Streamlit) — **piste B reprise**
Réutilise **directement** `kst_engine.py` (pas de réimplémentation), mais charge désormais `domains/piste_b.yaml`, **pas** `data/domain.yaml` (piste A). Piste B n'est plus en pause : domaine étendu de 5 à 7 concepts en subdivisant `arithmetic` → `{arithmetic_base, fractions_ratios}` et `algebra` → `{algebra_linear, algebra_advanced}` (les deux buckets les plus larges/hétérogènes identifiés en §4), 35 questions écrites à la main (5/concept, difficulté variée, distracteurs plausibles), `guess = 1/nb_options = 0.25` est une **borne combinatoire conservatrice** (pas une valeur choisie : `guess ≤ 1/k` pour un QCM à `k` options, cf. `note_calibration.md`), `slip = 0.10` un **point de référence sur un axe à balayer** (aucun argument de premier principe ne fixe `slip` — voir `note_calibration.md` §3-4) — le problème de calibration EM de la piste A (guess dégénéré) ne s'applique pas ici puisqu'il n'y a pas de calibration empirique. `|Z| = 50`. Écran final = diagnostic par concept (maîtrisé/lacune/incertain), pas un score.

- `domains/loader.py` (chargeur YAML→`Domain`, générique, garde `kst_engine.py` sans dépendance hors numpy), `domains/validate.py` (validateur B2 : cycle, `slip+guess<1`, ≥3 questions/concept, ids uniques, index de réponse valide), `domains/test_loader.py` (8 tests sur fixtures synthétiques).
- L'UI ne prétend plus être calibrée sur Junyi (c'était vrai pour l'ancien domaine à 5 concepts issu de piste A, faux pour piste B) — important pour l'honnêteté du rapport de stage.
- `data/domain.yaml` (piste A, calibré EM, 5 concepts) reste intact et utilisable indépendamment pour le pipeline de calibration/benchmark — seul l'app de démo a changé de domaine.
- Déployée sur Streamlit Community Cloud (`app.py` + `requirements.txt` + `.streamlit/config.toml` à la racine) — **à redéployer/rafraîchir après ce changement**.
- **Bug corrigé précédemment** : `pick_next()` ne faisait jamais passer `stage` à `"quiz"` — l'écran d'intro semblait bloqué au clic. Corrigé (commit `7558349`), vérifié par interaction réelle en local.
- **Accès collègue** : le repo GitHub est privé → l'app Streamlit est privée par défaut. Pour donner accès : Share (en haut à droite de l'app) → ajouter l'e-mail du collègue comme viewer. Alternative : ajouter comme collaborateur GitHub. Rendre le repo public est une option mais expose le code source — pas fait par défaut, à la décision de l'utilisateur.

### Benchmark A3 (`benchmark_a3.py`, `irt_baseline.py`) — fait, sur piste B
Compare trois politiques de sélection sur la banque `domains/piste_b.yaml` (35 questions, 7 concepts, |Z|=50), en rejouant **tous** les états de `domain.Z` avec des étudiants simulés (verité terrain BLIM identique pour les trois — seul l'algorithme diffère, même logique que l'argument "items identiques, deux algorithmes" de l'addendum) :

| Politique | Questions (moy.) | Exactitude état exact | Exactitude par concept |
|---|---|---|---|
| Adaptatif (gain d'info, KST) | 18.6 | 80 % | 95.7 % |
| Aléatoire | 27.7 | 82 % | 96.6 % |
| CAT-IRT (baseline 3PL) | 30.0 (plafond atteint) | 4 % | 66.3 % |

Table brute : `benchmark_a3_table.csv`. Figure prête pour le mémoire : `benchmark_a3.png`.

- **`irt_baseline.py`** : baseline CAT-IRT (3PL, sélection par information de Fisher, estimation de θ par EAP sur grille — pas MLE, qui diverge tant que les réponses sont uniformément justes/fausses, fréquent en début de CAT). Paramètres d'item **déduits** de la banque BLIM existante, pas calibrés indépendamment : `c = question.guess`, `b` depuis la difficulté déclarative YAML (1/2/3 → −1/0/+1), `a = 1.0` fixe pour tous les items. Diagnostic par concept lu via un seuil = `b` moyen des items du concept (mastery testing standard). **Limites assumées, à citer dans le rapport.**
- **Le CAT-IRT est délibérément mal spécifié** par rapport à la vérité terrain BLIM (θ continu unidimensionnel plaqué sur un état de connaissance discret multi-concept) — c'est exactement ce que le benchmark doit montrer : l'écart de performance d'un vrai CAT-IRT appliqué à des données qui suivent en réalité une KST, pas une comparaison entre deux modèles également vrais.
- **Lecture des résultats** : l'adaptatif réduit le nombre de questions de ~33 % par rapport à l'aléatoire (18.6 vs 27.7) pour une exactitude quasi identique (95.7 % vs 96.6 % — écart non significatif vu la taille de l'échantillon, 50 états). Le CAT-IRT n'atteint jamais son critère d'arrêt (SE(θ)≤0.3) en 30 questions et plafonne — sa précision par concept (66.3 %) reste nettement au-dessus du hasard (50 %, un θ unidimensionnel capture un signal d'aptitude globale) mais très en dessous des deux politiques KST, cohérent avec l'incapacité structurelle d'un θ unique à résoudre un état de maîtrise multi-concept partiel.
- Comme piste B n'est pas calibrée empiriquement, ce benchmark mesure le **gain algorithmique** des politiques de sélection sur une banque donnée, pas une validation empirique des paramètres — à formuler ainsi dans le rapport, distinct de la piste A (calibration réelle sur Junyi Academy).
- 10 tests dans `test_irt_baseline.py`, dont un test de récupération de paramètre (θ connu, EAP le retrouve en moyenne sur plusieurs étudiants simulés — même logique que le test EM de piste A) et un test sur la formule d'information de Fisher.

### Lot 1 — validation de l'approximation Monte Carlo (branche `lot5-vocabulaire-theorie`, en cours)
`kst_engine.information_gain_mc` a désormais un paramètre `mode` :
- `mode="sample_y"` (existant) : échantillonne les réponses. Un item étant binaire, il n'existe que **deux** postérieurs possibles quel que soit `N` — optimisé pour ne les calculer qu'une fois chacun (tirage binomial du compte, au lieu d'une boucle Python de `N` `bayes_update`). Coût final `O(|Z|)`, **le même ordre que `information_gain_exact`**, pour une valeur seulement approchée — la démonstration concrète de l'argument du préambule du Lot 1 (échantillonner `y` pour un item binaire est strictement pire que calculer l'exact). N'attaque pas le goulot réel (`|Z|`).
- `mode="sample_z"` (nouveau) : échantillonne les **états** `z ~ p(z)`, via la décomposition duale `I(Z;Y|a) = H(Y|a) − E_z[H(Y|a,z)]`. Coût `O(N)`, **indépendant de `|Z|`** — c'est la variante qui attaque le vrai goulot. Biais de plug-in vers le bas à petit `N` (Jensen, terme `H(Y|a)` non-linéaire en la moyenne empirique), qui se résorbe quand `N` grandit — propriété testée, pas un bug.

`pi_hat(p, domain, asked, n_samples, mode)` = π̂_N (notation du plan) : même structure gloutonne que `pi_star`, mais chaque candidat est évalué par `information_gain_mc` au lieu de `information_gain_exact`.

**E1 — fidélité de la politique**, sur 438 croyances issues de **vraies trajectoires** adaptatives (piste B, 20 états de `Z` rejoués avec `pi_star`, pas des priors uniformes artificiels) :

| N | accord `sample_y` | regret `sample_y` | accord `sample_z` | regret `sample_z` |
|---|---|---|---|---|
| 1 | 20 % | 79 % | 10 % | 44 % |
| 10 | 29 % | 94 % | 20 % | 87 % |
| 100 | 33 % | 99 % | 29 % | 97 % |

Table brute : `results/lot1_e1/raw.csv` (52 560 lignes). Figure : `results/lot1_e1/figure.pdf`.

**Lecture** : le taux d'accord est trompeur, exactement comme le plan l'annonçait — même à `N=100`, π̂_N ne retombe sur l'argmax exact que ~30 % du temps, mais la question choisie reste à ~97-99 % du gain d'information optimal. Le regret est la métrique honnête. `sample_z` a besoin d'environ **3 à 5× plus d'échantillons** que `sample_y` pour un regret équivalent (ex. `sample_y` à N=10 ≈ `sample_z` à N=30) — attendu, puisque `sample_y` profite gratuitement du calcul exact de `p_correct` (coût `O(|Z|)` déjà payé) alors que `sample_z` ne touche jamais `Z` en entier. Sur `|Z|=50` (piste B), calculer l'exact reste trivialement le meilleur choix des deux côtés — `sample_z` ne devient intéressant qu'à partir d'un `|Z|` où `O(|Z|)` dépasse `O(N≈30)`, seuil **pas encore mesuré** (c'est le rôle d'E3, pas fait).

**E2 — coût en aval**, `lot1_e2_cout_en_aval.py` : rejoue le benchmark complet (50 états × 5 réplications) avec `simulate_mc` (π̂_N) au lieu de `simulate` (π*). Résultat rassurant : dès `N=3`, les deux modes retombent quasi sur l'exact (~19-20 questions vs 18,6, ~96 % d'exactitude par concept vs 95,7 %) — le coût en aval de l'approximation est faible, malgré le faible taux d'accord mesuré par E1. **Sauf un piège net à `N=1` pour `sample_z`** : voir ci-dessous.

**Piège découvert par E2, invisible dans E1** — `information_gain_mc(mode="sample_z", n_samples=1)` est **dégénéré**, pas juste bruité : `H(Y|a)` et `E_z[H(Y|a,z)]` sont calculés sur exactement le même unique point échantillonné, leur différence vaut **0.0 exactement**, pour toute question, à tout appel. Conséquence en cascade : `pi_hat` retombe alors sur le premier candidat balayé (argmax sur des ex-aequo à 0), et `should_stop` s'arrête **immédiatement** (`ig=0 < min_ig`) — `simulate_mc(n_samples=1, mode="sample_z")` ne pose **jamais aucune question** (mesuré : 0,00±0,00, exactitude 47 % = à peu près le hasard). E1 ne révèle pas cette pathologie parce qu'il mesure des décisions isolées à mi-trajectoire (croyance déjà informative) ; E2 la révèle parce qu'il rejoue la boucle complète depuis le prior uniforme — **exactement pourquoi les deux expériences sont nécessaires**. `N=1` est proscrit avec `sample_z` (documenté dans `kst_engine.py` + testé) ; `N≥3` suffit à lever la dégénérescence.

**E3 — coût de calcul selon `|Z|`**, `lot1_e3_cout_calcul.py` : domaines synthétiques sans prérequis (`|Z|=2^n_concepts`, 5 à 13 concepts → `|Z|` de 32 à 8192), temps d'une décision complète (`pi_star` vs `pi_hat`). Confirme empiriquement la prédiction du docstring : `sample_y` **croît avec `|Z|` comme l'exact** (même ordre de grandeur, ~13-17ms à `|Z|=8192` pour les deux) — jamais avantageux. `sample_z` reste **quasi constant** (~0,2 à ~2,8ms sur toute la plage) — la variante qui attaque vraiment le goulot.

**Recommandation d'ingénierie chiffrée (livrable du Lot 1)** : croisement des courbes mesurées (interpolation log-log) → `sample_z` devient moins cher que l'exact autour de `|Z| ≈ 1 400` (N=10) à `|Z| ≈ 1 500` (N=100). **Sous `|Z| ≈ 1 400`, calculer l'exact (`pi_star`) ; au-delà, `sample_z` avec `N ≈ 10-30`** (E1 : regret déjà < 15 % à N=30). Piste A (`|Z|=18`) et piste B (`|Z|=50`) sont très en dessous de ce seuil — calculer l'exact y est toujours le bon choix, l'approximation MC n'a d'intérêt que pour des domaines nettement plus riches que ceux actuellement utilisés dans ce PFA. Seuil mesuré sur une machine/version numpy donnée, à prendre comme ordre de grandeur, pas une constante universelle.

**Reste du Lot 1** : 1.5 (figure `|Z|` vs nombre de concepts pour les domaines réels + synthétiques — cosmétique, E3 couvre déjà l'essentiel).

### Lot 2 — ancrage géométrique (branche `lot5-vocabulaire-theorie`, fait)

Les chapitres 4-5 du rapport portent sur Fisher-Rao, les géodésiques, le gradient naturel — les résultats n'avaient jusque-là aucun objet géométrique. Ajouté à `kst_engine.py` :

- **`fisher_rao_distance(p, q)`** (2.1) : `d(p,q) = 2·arccos(Σ_z √(p(z)·q(z)))`, forme fermée via le plongement racine `x=2√p` dans l'octant positif d'une sphère de rayon 2. Clip `[-1,1]` sur l'affinité (piège classique du projet : arccos indéfini si l'affinité dépasse 1 par arrondi flottant).
- **`cumulative_arc_length(belief_trace)`** (2.1) : longueur d'arc cumulée le long d'une trajectoire de croyances.
- **`expected_fisher_rao_step(slip, guess, prior)`** (2.4) : déplacement géométrique moyen sur `Δ({non-maîtrise, maîtrise})` après une réponse — distinct d'`item_information` (Lot 1, mesure KL/Wald), mais **classe les mêmes configurations dans le même ordre** (testé, `test_coherent_avec_item_information_sur_le_classement`) : deux angles indépendants sur le même phénomène.
- `simulate()` renvoie désormais aussi `belief_trace` (séquence complète des croyances, pas seulement l'entropie) — extension additive, aucune rupture de compatibilité.

**2.2 — longueur d'arc cumulée** (`lot2_2_arc_length.py`), piste B, 50 états, adaptatif vs aléatoire (même graine, comparaison appariée) : l'adaptatif parcourt davantage de distance par question en début de trajectoire (pente plus forte jusqu'à t≈15), exactement la prédiction du plan — puis plafonne plus tôt (il s'arrête après moins de questions), pendant que l'aléatoire continue d'accumuler et le dépasse en distance totale.

**2.3 — trajectoire sur la sphère** (`lot2_3_sphere_trajectory.py`) : domaine minimal à 3 états (`Z={∅,{a},{a,b}}`), trajectoire `p₀→p_T` tracée sur l'octant positif via `x=2√p` — l'illustration théorique de la semaine 2, avec une vraie trajectoire simulée (π*, z_true={a,b}) plutôt qu'un schéma.

**2.4 — effondrement géométrique vs `(slip,guess)`** (`lot2_4_fisher_info_slip_guess.py`) : courbe à `slip=0,10` fixe, avec les 5 concepts piste A (calibrés EM, `guess` 0,50–0,75) et piste B (`guess=0,25`) marqués avec leurs vraies valeurs. **Résultat net** : les concepts piste A se classent `probability_statistics > algebra > analytic_geometry > geometry > arithmetic` en déplacement géométrique — **exactement le même ordre** qu'en `item_information` (Lot 1). Piste B (déplacement ≈0,72) domine largement tous les concepts piste A (déplacement 0,15–0,36). **La saga du `guess` dégénéré (§4) cesse d'être une limite documentée après coup et devient une prédiction quantitative de la théorie, vérifiée sur données réelles par deux mesures indépendantes qui s'accordent** — l'effet narratif exact que visait le Lot 2.

Table brute et figures : `results/lot2_2_arc_length/`, `results/lot2_3_sphere/`, `results/lot2_4_fisher_info/`. 12 nouveaux tests (`TestFisherRaoDistance`, `TestCumulativeArcLength`, `TestExpectedFisherRaoStep`, + 3 sur `belief_trace`), 157 tests passent au total.

### Lot 3 — robustesse à la mauvaise spécification (branche `lot5-vocabulaire-theorie`, fait)

Aujourd'hui un seul jeu de paramètres génère les réponses **et** est supposé par le moteur — l'objection la plus facile à formuler pour un jury. Les 6 points du plan sont traités.

**3.1 — Découplage.** `simulate()` (`kst_engine.py`) et `simulate_irt()` (`irt_baseline.py`) gagnent chacun deux paramètres optionnels `verite_slip`/`verite_guess` : si fournis, la réponse simulée est générée avec **ces** valeurs, alors que la mise à jour bayésienne continue d'utiliser `domain.L` (ce que le moteur croit, figé à la construction du `Domain`). `None` (défaut) = comportement inchangé. Exactement le changement de signature minimal demandé par le plan — pas une réécriture. 4 tests.

**3.2 — Grille de bruit** (`lot3_2_grille_mauvaise_specification.py`) : moteur figé à piste B (`slip=0,10`/`guess=0,25`), vérité balayée sur `slip∈{0,05·0,10·0,20·0,30}×guess∈{0,25·0,40·0,55·0,70}` (16 cellules), 3 politiques (KST adaptatif, aléatoire, CAT-IRT), 50 états × 20 répétitions = 48 000 lignes. Exactitude par concept (KST adaptatif) :

| slip\guess | 0,25 | 0,40 | 0,55 | 0,70 |
|---|---|---|---|---|
| 0,05 | 98,1 % | 94,0 % | 85,6 % | 71,5 % |
| **0,10** | **95,8 %** | 91,8 % | 83,0 % | 70,7 % |
| 0,20 | 88,6 % | 85,5 % | 77,2 % | 63,9 % |
| 0,30 | 78,6 % | 75,1 % | 67,6 % | 55,9 % |

(cellule en gras = bien spécifiée, correspond au 95,7 % déjà mesuré par le benchmark A3 — cohérence croisée confirmée). Dégradation nette et attendue quand `guess` réel dépasse ce que le moteur croit.

**3.3 — Graphe de prérequis faux** (`lot3_3_prereq_faux.py`) : 400 étudiants simulés, 15 % (60) avec un état vrai **hors** `Z` (viole la structure de prérequis, ex. `algebra_advanced` maîtrisé sans `algebra_linear` — 78 états invalides possibles sur 128 sous-ensembles). `correct_diagnosis` est faux par construction pour ce groupe (le moteur ne peut représenter l'état) — la distance de Hamming est la bonne métrique : **0,315** concept mal diagnostiqué en moyenne pour les états valides, **1,467** pour les états invalides (~4,7× pire, mais dégradation **gracieuse**, pas un effondrement — le moteur reste majoritairement correct, 5,5/7 concepts en moyenne, même sur un état qu'il ne peut structurellement pas représenter).

**3.4 — Arène miroir** (`lot3_4_arene_miroir.py`) : vérité générée par le 3PL continu (le modèle DE l'IRT) au lieu du BLIM, 200 étudiants (`θ~N(0,1)`). Résultat voulu et confirmé : **l'IRT gagne sur sa propre vérité** (85,0 % vs 73,1 % d'exactitude par concept pour KST) — l'exact miroir du benchmark A3 (vérité BLIM : KST 95,7 % vs IRT 66,3 %). Le message n'est pas « KST bat IRT » ni l'inverse : **chaque modèle domine sur sa propre vérité générative**, et la vraie question — laquelle décrit les données Junyi — renvoie à la piste A, désamorçant l'objection de l'arène truquée.

**3.5 — Calibration de la confiance.** Diagramme de fiabilité (confiance annoncée à l'arrêt vs proportion réelle d'états exactement corrects), politique KST adaptative, sur les mêmes runs que 3.2 : à la cellule bien spécifiée, la courbe suit la diagonale (erreur de calibration `ECE=0,018`). Au régime piste A (`guess=0,70`), la confiance annoncée atteint ~90 % alors que l'exactitude réelle plafonne à ~10-38 % (`ECE=0,467`) — **« 90 % sûr » ne veut plus dire « 9 fois sur 10 »**, mesuré et chiffré, pas seulement affirmé.

**3.6 — Correction** (`lot3_6_correction.py`) : même grille rejouée avec un moteur volontairement prudent (`slip=0,20`/`guess=0,40` supposés). **Résultat plus nuancé que ne le laissait supposer le plan** — ce n'est pas une simple restauration :
- Régime piste A : `ECE` passe de 0,467 à **0,271** (amélioration réelle, ~42 %), au prix de plus de questions (23,5→27,7).
- Cellule bien spécifiée : `ECE` passe de 0,018 à **0,195** (nettement **dégradée** — le moteur devient sous-confiant là où il n'avait pas besoin de l'être), et coûte aussi plus cher (19,3→27,3 questions).

**La prudence n'est pas un correctif gratuit : c'est un compromis robustesse/précision explicite**, qui améliore le pire cas au prix du cas normal — un résultat plus intéressant et plus honnête pour le rapport qu'un simple « ça marche », cohérent avec l'esprit `note_calibration.md` (jamais forcer un résultat à correspondre à l'attente).

Résultats bruts et figures : `results/lot3_2_grille/`, `results/lot3_3_prereq_faux/`, `results/lot3_4_arene_miroir/`, `results/lot3_6_correction/`. 161 tests passent au total (4 nouveaux pour 3.1).

### Lot 4 — Boucler la piste A (branche `lot5-vocabulaire-theorie`, fait)

**A3 sur le domaine calibré** (`lot4_a3_piste_a.py`) — fait. Rejoue le benchmark A3 sur `data/domain.yaml` (5 concepts, `|Z|=18`, `guess` calibré EM 0,50–0,75) au lieu de piste B, plafond de questions relevé à 150 (`note_calibration.md` §6). Banque synthétique : 40 questions/concept (200 au total), toutes partageant le `(slip, guess)` réellement calibré du concept — aucune variation par item inventée, cette donnée n'existe pas pour piste A.

| Politique | Questions (moy.) | Exactitude par concept |
|---|---|---|
| **Adaptatif (π*)** | 74,8 ± 23,6 | 92,2 % |
| **Aléatoire** | 114,8 ± 35,1 | 91,1 % |
| **CAT-IRT** | 150,0 (plafond) | 60,0 % |

**Dégradation forte confirmée**, exactement la prédiction du Lot 2.4 : ~4× plus de questions que piste B (18,6) pour une précision comparable (92,2 % vs 95,7 %). Prédiction fermée (`questions_needed` par concept, somme sur les 5 valeurs réelles) : 108,5 questions — contre 74,8 mesurées, écart de ~45 % (bien moins précis que l'accord à 3 % obtenu sur piste B). Écart honnête à noter : l'approximation de Wald ignore le partage d'information entre concepts via les prérequis, et cet effet pèse proportionnellement plus lourd sur seulement 5 concepts fortement contraints que sur 7.

**Test D1 (hétérogénéité)** (`lot4_d1_heterogeneite.py`) — fait, résultat **négatif et net**. Isole le bucket `arithmetic` (301 exercices, area Junyi brute, 25 925 992 lignes lues → 17 679 833 retenues via réextraction complète du log brut), le subdivise comme dans la première tentative à 9 concepts (`arithmetic_base`/`fractions_ratios`, même mapping topic, git commit `568dc1d`), et relance l'EM **isolément** sur ce sous-domaine à 2 concepts seul (pas joint avec 7 ou 9 autres concepts comme précédemment). Convergence propre (22 itérations, écart-type < 0,0002 sur 5 redémarrages) :

| Concept | slip | guess |
|---|---|---|
| `arithmetic` combiné (piste A, 5 concepts) | 0,1064 | 0,7462 |
| `arithmetic_base` (ancien, 9 concepts, **fit joint**) | 0,096 | 0,756 |
| `fractions_ratios` (ancien, 9 concepts, **fit joint**) | 0,141 | 0,685 |
| **`arithmetic_base` (D1, fit ISOLÉ)** | **0,0949** | **0,7588** |
| **`fractions_ratios` (D1, fit ISOLÉ)** | **0,1367** | **0,6845** |

**Deux résultats en un.** (1) Le fit isolé retombe presque exactement sur l'ancien fit joint (écarts en 3ᵉ décimale) — élimine une explication alternative possible (interférence de l'estimation conjointe avec 7-9 concepts) : ce n'était pas un artefact de l'ajustement joint. (2) **La subdivision ne fait PAS baisser `guess`** — `arithmetic_base` (0,759) est même *plus élevé* que le bucket combiné (0,746), `fractions_ratios` (0,685) descend un peu mais reste très implausible. **L'hypothèse d'hétérogénéité par granularité n'est pas confirmée** pour ce bucket, avec une preuve chiffrée à l'appui (pas seulement l'absence de preuve du contraire). Cohérent avec §4 point 4 (le jeu complet reste bien identifié malgré un `guess` élevé — la piste probable est ailleurs, non résolue, à documenter comme limite ouverte plutôt qu'à forcer).

`responses_d1.parquet` (193 Mo, régénérable via le script) exclu de git comme les autres artefacts de données bruts (`.gitignore` mis à jour).

---

## 3. Artifacts annexes (hors pipeline de production, mais dans le repo)
- `sprint0_suivi.pptx` + `build_deck.js` — deck de suivi Sprint 0 (théorie, refactor, benchmark). Généré via pptxgenjs.
- `refactor_avant_apres.png`, `entropy_trace.png` — figures du benchmark Sprint 0.
- Un artifact HTML autonome (port JS fidèle du moteur, publié via l'outil Artifact de Claude) existe aussi comme démo alternative — mentionné pour mémoire, mais **l'app Streamlit est maintenant la version de référence pour présenter**.

---

## 4. La saga du diagnostic `guess` dégénéré — pourquoi le domaine est passé de 9 à 5 concepts

**Ne pas relancer cette investigation sans raison nouvelle — elle a déjà consommé beaucoup de temps et a une conclusion claire.**

1. Domaine initial à 9 concepts (subdivision d'`arithmetic`/`algebra` par topic) → `guess` calibré anormalement élevé (0,40-0,76 sur 7/9 concepts). Documenté dans `PROMPT_A2_GRANULARITE.md`.
2. Diagnostic D0 (`docs/diagnostic_guess.md`, `docs/revue_d0.md`) : hypothèse que la répétition de pratique (un étudiant réessaie un exercice jusqu'à la maîtrise) contamine le signal. **R1 confirme le mécanisme** (taux de réussite 67,8%→85% entre 1ère et 10e tentative) mais...
3. **Revue tour 2** (`docs/revue_d0_tour2.md`) : un run de contrôle (échantillon aléatoire de même taille que la variante "1ère tentative", tiré par loi hypergéométrique) reproduit la même baisse de `guess` que le filtre "1ère tentative". **Conclusion : c'est un problème d'IDENTIFIABILITÉ (trop peu d'observations par (étudiant, concept) une fois sous-échantillonné), pas la contamination par pratique répétée en soi.**
4. **Mais** : le jeu de données COMPLET (25,9M réponses, pas sous-échantillonné) reste lui-même bien identifié (faibles écarts-types sur 5 redémarrages EM indépendants) et affiche quand même un `guess` élevé. Donc le problème du jeu complet n'est PAS l'identifiabilité — c'est probablement l'hypothèse originale (hétérogénéité des concepts trop larges pour le tout-ou-rien du BLIM), qui n'a jamais été formellement invalidée, juste mise de côté.
5. **Décision finale (celle qui compte)** : face à la deadline, l'utilisateur a tranché **"on simplifie, une V1 qui marche, on améliore après"**. Domaine réduit à 5 concepts (retour aux `area` Junyi brutes, sans subdivision par topic ; `calculus`/`logics` exclus — trop peu d'exercices). `\|Z\|` est descendu à 18 (sous la fourchette [30,500] visée initialement pour 8-14 concepts — bornes assouplies consciemment, documenté dans `data/extract_junyi.py`).

**Si l'utilisateur revient sur la qualité de la calibration** : le prochain pas naturel serait de tester l'hétérogénéité directement (D1 du plan original, jamais exécuté formellement) plutôt que de relancer un énième round d'exploration sur l'identifiabilité (déjà tranché, cf. point 3).

---

## 5. Pièges déjà rencontrés — pour ne pas les refaire

- **La colonne `prerequisites` de `junyi_Exercise_table.csv` est une liste séparée par virgules**, pas une valeur unique. Bug trouvé et corrigé une fois (avait rendu `calculus` isolé à tort).
- **Le fichier de logs est ordonné dans le temps, pas regroupé par étudiant.** Un sondage sur un chunk contigu ne voit qu'un fragment tronqué de l'historique d'un étudiant — invalidé une fois, corrigé en deux passes (candidats puis reconstruction complète).
- **Toujours comparer les log-vraisemblances EM en moyenne PAR ÉTUDIANT, jamais en total** — un seuil de convergence absolu sur le total (~11M) est soit inatteignable soit trivial selon la taille de l'échantillon. Bug réel rencontré et corrigé.
- **RAM très limitée sur cette machine** (a varié entre 0,6 et 8 Go libres selon les moments) — toujours privilégier lecture par chunks / statistiques agrégées plutôt que charger les fichiers bruts (26M lignes) en mémoire d'un coup. Le tirage hypergéométrique sur comptes agrégés (au lieu de relire les lignes individuelles) est un pattern réutilisable si besoin de ré-échantillonner.
- **Sur Windows, `git push`/`git commit` peuvent dépasser un timeout de 2 min** sans avoir échoué — toujours vérifier `git status`/`git log` avant de réessayer ou de conclure à un échec.
- **`app.py` (Streamlit) a eu un vrai bug de state machine** (oubli de faire avancer `stage`) — si un écran semble "figé" après une interaction, vérifier D'ABORD la logique de transition d'état avant de soupçonner l'outil de test/le navigateur.

---

## 6. Style de travail attendu (déjà établi, à maintenir)

- Petits incréments testables, un test pytest par fonction, fixtures synthétiques pour les tests (jamais les fichiers réels de plusieurs millions de lignes).
- Commenter les fonctions mathématiques avec le numéro de chapitre de la monographie correspondant.
- Ne pas trancher seul les choix "élégance d'ingénierie vs fidélité théorique" — les surfacer explicitement. (Ceci dit, sous contrainte de deadline explicite comme actuellement, l'utilisateur a demandé des décisions rapides — équilibrer les deux consignes selon ce que l'utilisateur redemande.)
- Committer et pousser après chaque étape significative, avec des messages de commit détaillés (contexte + résultat chiffré quand pertinent).
- Documents de passation (`PROMPT_*.md`, `docs/*.md`) écrits à chaque point de décision important — ce fichier en est un.

---

## 7. Où on en est précisément, là maintenant

- **Décision actée** : le rendu est un **rapport scientifique** (stage recherche), Streamlit suffit pour la démo/soutenance → **Sprint 1 (API FastAPI) explicitement écarté pour l'instant**, pas de valeur sans frontend Next.js prévu.
- **Piste B reprise et étendue** (5 → 7 concepts, commit `d7b74a6`) : `domains/piste_b.yaml` + `domains/loader.py` + `domains/validate.py` + `domains/test_loader.py`, 35 questions, `app.py` bascule dessus. 108 tests passent, app vérifiée en navigateur (intro → quiz → résultats, diagnostic correct sur les 7 concepts). Poussé sur GitHub.
- **Piste A intacte**, non touchée par ce travail — `data/domain.yaml` (5 concepts calibrés EM) reste le domaine de référence pour la calibration/le futur benchmark A3.
- **Benchmark A3 fait** (`benchmark_a3.py` + `irt_baseline.py`, voir §2) : adaptatif vs aléatoire vs CAT-IRT (baseline 3PL) sur `domains/piste_b.yaml`, exhaustif sur les 50 états de `Z`. Résultat net : adaptatif réduit ~33 % des questions vs aléatoire à précision quasi égale (95.7 % vs 96.6 %) ; le CAT-IRT plafonne à 30 questions et reste nettement moins précis (66.3 %). Table (`benchmark_a3_table.csv`) et figure (`benchmark_a3.png`) prêtes pour le mémoire. 118 tests passent (108 + 10 nouveaux sur `irt_baseline.py`). Poussé sur GitHub.
- Question restée ouverte (non résolue, action à l'utilisateur) : accès du collègue à l'app déployée sur Streamlit Community Cloud — **à redéployer** après le changement de domaine piste B (Share → ajouter son e-mail, ou collaborateur GitHub).
- **Pas de prochaine tâche explicitement actée au-delà de ça** — demander à l'utilisateur ce qu'il veut faire ensuite (intégrer les résultats A3 dans une rédaction de rapport ? refaire tourner le benchmark sur le domaine calibré piste A pour comparer ? présentation/soutenance ?) plutôt que de supposer.
