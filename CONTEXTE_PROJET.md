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
Implémente les 4 couches KST : `build_knowledge_space` (chapitre 2), `Domain`/`Concept`/`Question` avec BLIM (chapitre 3), `bayes_update`/`entropy`/`information_gain_exact`/`select_next`/`should_stop` (chapitres 6-7). Concept et Question sont **séparés** (refactor fait tôt dans le projet) : plusieurs questions par concept, chacune avec son propre `slip`/`guess`. 47+ tests, cas d'or de la monographie (0,500→0,818) vérifié.

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
Réutilise **directement** `kst_engine.py` (pas de réimplémentation), mais charge désormais `domains/piste_b.yaml`, **pas** `data/domain.yaml` (piste A). Piste B n'est plus en pause : domaine étendu de 5 à 7 concepts en subdivisant `arithmetic` → `{arithmetic_base, fractions_ratios}` et `algebra` → `{algebra_linear, algebra_advanced}` (les deux buckets les plus larges/hétérogènes identifiés en §4), 35 questions écrites à la main (5/concept, difficulté variée, distracteurs plausibles), `slip`/`guess` = valeurs **expertes non calibrées** (`guess = 1/nb_options = 0.25`, `slip = 0.10`) — le problème de calibration EM de la piste A (guess dégénéré) ne s'applique pas ici puisqu'il n'y a pas de calibration empirique. `|Z| = 50`. Écran final = diagnostic par concept (maîtrisé/lacune/incertain), pas un score.

- `domains/loader.py` (chargeur YAML→`Domain`, générique, garde `kst_engine.py` sans dépendance hors numpy), `domains/validate.py` (validateur B2 : cycle, `slip+guess<1`, ≥3 questions/concept, ids uniques, index de réponse valide), `domains/test_loader.py` (8 tests sur fixtures synthétiques).
- L'UI ne prétend plus être calibrée sur Junyi (c'était vrai pour l'ancien domaine à 5 concepts issu de piste A, faux pour piste B) — important pour l'honnêteté du rapport de stage.
- `data/domain.yaml` (piste A, calibré EM, 5 concepts) reste intact et utilisable indépendamment pour le pipeline de calibration/benchmark — seul l'app de démo a changé de domaine.
- Déployée sur Streamlit Community Cloud (`app.py` + `requirements.txt` + `.streamlit/config.toml` à la racine) — **à redéployer/rafraîchir après ce changement**.
- **Bug corrigé précédemment** : `pick_next()` ne faisait jamais passer `stage` à `"quiz"` — l'écran d'intro semblait bloqué au clic. Corrigé (commit `7558349`), vérifié par interaction réelle en local.
- **Accès collègue** : le repo GitHub est privé → l'app Streamlit est privée par défaut. Pour donner accès : Share (en haut à droite de l'app) → ajouter l'e-mail du collègue comme viewer. Alternative : ajouter comme collaborateur GitHub. Rendre le repo public est une option mais expose le code source — pas fait par défaut, à la décision de l'utilisateur.

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
- **Prochaine étape actée mais pas commencée** : le benchmark (adaptatif / aléatoire / IRT) reste envisagé après la piste B — `kst_engine.simulate()` existe déjà et fonctionne sur n'importe quel domaine (testé manuellement sur `domains/piste_b.yaml` : 11 à 30 questions selon l'état simulé, moyenne ≈18,6). Ce qui manque : la baseline CAT-IRT (3PL, critère de Fisher), à ajouter pour un vrai A3. Cadrage discuté : sur piste B ça mesure le **gain algorithmique** (comparaison de politiques de sélection), pas une validation empirique — à formuler clairement ainsi dans le rapport, distinct de la calibration réelle de piste A.
- Question restée ouverte (non résolue, action à l'utilisateur) : accès du collègue à l'app déployée sur Streamlit Community Cloud — **à redéployer** après ce changement de domaine (Share → ajouter son e-mail, ou collaborateur GitHub).
- **Pas de prochaine tâche explicitement actée au-delà de ça** — demander à l'utilisateur s'il veut enchaîner sur le benchmark CAT-IRT maintenant, ou autre chose, plutôt que de supposer.
