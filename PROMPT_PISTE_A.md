# Prompt de reprise — Piste A, tâche A1 (extraction et nettoyage Junyi15)

> À coller dans la session qui a produit `kst_engine.py`, `test_kst_engine.py`, `kst_benchmark_plots.py` et le deck `sprint0_suivi.pptx`. Le contexte théorique (PROMPT_DEMARRAGE.md) et les décisions déjà prises (ADDENDUM_BANQUE_QUESTIONS.md) sont dans le repo — ne les redemande pas, ils sont déjà tranchés.

---

## Où on en est

Sprint 0 (moteur pur) est fait : refactor concept/question, 47 tests pytest passent (1 xfail documenté), benchmark avant/après commité. Piste B (banque de démo) a un schéma YAML proposé mais pas encore commencée en code. **On met piste B en pause et on démarre piste A** : collecte et nettoyage des données Junyi15, pas d'estimation de calendrier supplémentaire — le travail commence maintenant.

Objectif de piste A pour mémoire : calibrer `slip`/`guess` par EM sur des vraies réponses d'élèves et produire le benchmark adaptatif / aléatoire / CAT-IRT sur données réelles (voir ADDENDUM_BANQUE_QUESTIONS.md, tâches A2/A3). **Ce prompt ne couvre que A1** — l'extraction. Ne pas enchaîner sur A2/EM ou A3/benchmark sans validation intermédiaire.

---

## Ce qui a déjà été vérifié (ne pas re-chercher)

Recherche faite en amont pour identifier la source exacte de Junyi15 :

- Le portail officiel est PSLC DataShop, dataset 1198 (`https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198`), **gated derrière un compte** — pas une option pour un agent automatisé.
- Le paquet pip `EduData` (`pip install EduData`) expose la même donnée brute via un miroir **sans authentification**, à l'URL exacte trouvée en inspectant le code source du paquet (`EduData/DataSet/download_data/download_data.py`, dict `URL_DICT`) :

  ```
  http://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/junyi.rar
  ```

  Confirmé par une requête `HEAD` : `Content-Length: 1295551222` (~1,21 Go), `Content-Type: application/x-rar-compressed`, réponse `200 OK` sans redirection vers une page de login.

- **Ne pas confondre avec `ktbd-junyi`** (autre entrée de `URL_DICT`, dataset déjà pré-agrégé aux 1000 étudiants les plus actifs, format JSON de séquences) — c'est un sous-produit pour le knowledge tracing, pas la donnée brute dont on a besoin ici (table d'exercices complète + annotations de prérequis + logs bruts).
- Ne pas utiliser `edudata download junyi` en boîte noire : il télécharge la bonne archive mais applique aussi une décompression automatique dont le comportement n'a pas été vérifié. Préférer télécharger `junyi.rar` directement à l'URL ci-dessus et décompresser explicitement, pour garder la main sur ce qui atterrit dans `data/raw/`.
- D'après le notebook de référence de l'addendum (`docs/analysis/junyi/junyi.ipynb` du repo EduData), l'archive doit contenir au moins :
  - `junyi_Exercise_table.csv` — colonnes : `name`, `live`, `prerequisites`, `h_position`, `v_position`, `creation_date`, `seconds_per_fast_problem`, `pretty_display_name`, `short_display_name`, `topic`, `area`. Timestamps à résolution complète (microseconde), confirmant qu'il s'agit bien de Junyi15 et pas de la coupe Kaggle 2020.
  - `relationship_annotation_training.csv` et `relationship_annotation_testing.csv` — annotations de relations entre exercices (papier EDM 2015 de Chang, Hsu, Chen).
  - une table de logs d'interaction élève × exercice (nom exact à confirmer une fois l'archive ouverte).

  **Ces noms de fichiers et colonnes viennent de la doc, pas d'une inspection directe de l'archive : à vérifier et corriger dans le script dès qu'elle est décompressée.**

---

## Ce qu'il reste à décider une fois les fichiers ouverts (à documenter, pas à trancher en silence)

1. **Quelle source fait foi pour la relation de prérequis** : la colonne `prerequisites` de `junyi_Exercise_table.csv` (un lien direct par exercice, façon arbre), ou `relationship_annotation_{training,testing}.csv` (annotations pairées, probablement plus riches mais à un autre niveau de confiance) ? Les deux ne se contredisent pas forcément mais peuvent différer. Documenter le choix et pourquoi.
2. **La stratégie d'agrégation exercices → concepts.** Le graphe Junyi est entre ~1330 exercices, il faut descendre à 8–14 nœuds. `topic` et/ou `area` sont les candidats naturels pour grouper — mais vérifier que le résultat de ce groupement reste un graphe de prérequis cohérent (pas de cycle introduit par l'agrégation, pas de nœud isolé). Si `topic` donne un nombre de groupes hors de la fourchette 8–14, le dire explicitement plutôt que de forcer un découpage arbitraire.

---

## Tâche A1 — Script d'extraction (`data/extract_junyi.py`)

**Entrée** : les fichiers bruts Junyi15 dans `data/raw/`.
**Sortie** :
- `data/domain.yaml` — concepts (8 à 14) + prérequis (paires `[a, b]`, même format que `prereqs` dans `kst_engine.build_knowledge_space`). Pas de `slip`/`guess`/questions ici : piste A calibre ces paramètres par EM en A2, ce YAML n'a pas besoin des champs de banque de questions de piste B.
- `data/responses.parquet` — colonnes `student_id`, `concept_id`, `correct`, `timestamp`. `concept_id` doit être le nœud agrégé (post-groupement), pas l'exercice brut.

**Contraintes non négociables** (rappel addendum) :
- Sous-graphe **connexe**. Un ensemble de nœuds sans arêtes fait exploser `Z = 2^n` et annule l'intérêt de la KST — ne jamais livrer ça, même temporairement.
- 8 à 14 concepts, `|Z|` entre 30 et 500 une fois passé dans `build_knowledge_space`.
- **Si `|Z| > 2000`, arrêter le script et signaler** plutôt que de continuer avec un espace ingérable.

**Étapes suggérées, à petits incréments testables (même discipline que le moteur — un test par transformation, pas juste un script monolithique qui tourne une fois)** :
1. Téléchargement + décompression dans `data/raw/` — **demander confirmation avant de télécharger** (fichier de ~1,21 Go, pas anodin). Ne pas committer `data/raw/` dans git (à ajouter au `.gitignore`).
2. Chargement et inspection des tables réelles — comparer aux colonnes attendues ci-dessus, corriger ce prompt/le code si l'archive diffère.
3. Construction du graphe de prérequis **au niveau exercice**, en explicitant la source retenue (`prerequisites` vs `relationship_annotation_*`).
4. Agrégation exercice → concept (par `topic`/`area` ou équivalent), avec un test qui vérifie que l'agrégation ne casse pas l'acyclicité.
5. Extraction du plus grand sous-graphe connexe si le graphe complet ne l'est pas déjà, avec le nombre de concepts qui en résulte affiché clairement.
6. Écriture de `domain.yaml`, validation immédiate via `build_knowledge_space` de `kst_engine.py` (réutiliser la fonction existante, ne pas la dupliquer) pour vérifier `30 ≤ |Z| ≤ 500`.
7. Mapping des logs bruts vers `responses.parquet`, avec le même mapping exercice→concept qu'à l'étape 4.

**Un test pytest par étape**, comme pour le moteur (`test_extract_junyi.py`), avec des fixtures sur un petit extrait synthétique des tables plutôt que de dépendre du fichier de 1,2 Go pour la CI.

---

## Comment on continue de travailler

Mêmes règles que section 7 de `PROMPT_DEMARRAGE.md` : petits incréments testables, pas de dépendance lourde sans demander (au-delà d'`EduData`, déjà installé pour la reconnaissance de la source — `pandas`/`pyarrow` seront probablement nécessaires pour le parquet, à confirmer avant d'ajouter). Documenter dans le code *pourquoi* un choix d'agrégation a été fait, pas seulement quoi — ça sert directement à justifier le sous-graphe retenu dans le mémoire.

**Première action attendue** : demander la confirmation de téléchargement de `junyi.rar` (source et taille ci-dessus), puis, une fois les fichiers ouverts, rapporter la structure réelle des tables avant d'écrire le script d'agrégation — ne pas supposer que la doc EduData est exacte.
