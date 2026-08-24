# Addendum au prompt de démarrage — La banque de questions (pistes A et B)

> À coller à la suite de `PROMPT_DEMARRAGE.md`, ou seul si le contexte du projet est déjà chargé.
> Rappel du contexte : plateforme de quiz adaptatif fondée sur la Knowledge Space Theory, moteur `kst_engine.py`, backend FastAPI, front Next.js, PFA UM6P.

---

## Le principe : deux banques, pas une

Il ne faut **pas** chercher une banque unique qui serve à tout. Le projet a deux besoins disjoints, et les confondre fait perdre des semaines.

| | **Piste A — validation scientifique** | **Piste B — démo jouable** |
|---|---|---|
| Objectif | calibrer `slip`/`guess`, produire le benchmark du mémoire | avoir un quiz utilisable dans le navigateur |
| Besoin réel | des **logs de réponses réelles** + un graphe de prérequis | des **énoncés propres** rattachés à des concepts |
| Sans importance | le texte des questions (on rejoue des logs, on n'affiche rien) | le volume de réponses (personne ne passera le quiz 500 fois) |
| Livrable | tableau comparatif adaptatif / aléatoire / IRT | ~60 questions + le domaine en YAML |

Les deux pistes ne partagent aucun fichier de données. Elles partagent uniquement le moteur.

---

## PISTE A — Calibration et benchmark sur données réelles

### Dataset retenu : Junyi Academy (version Junyi15)

Raison du choix : c'est le seul dataset public largement utilisé qui fournisse un **graphe de prérequis entre exercices annoté par des experts**. C'est littéralement la relation de surmise `⪯` dont la KST a besoin, obtenue gratuitement au lieu d'être construite à la main. ASSISTments2009 n'a aucun graphe de structure ; MOOCCubeX n'en a qu'un incomplet.

Colonne clé des métadonnées d'exercices : `prerequisite` (le parent dans la carte de connaissances). Colonnes utiles associées : `name`, `topic`, `area`, `h_position`, `v_position`.

Documentation :
- https://edudata.readthedocs.io/en/doc/build/blitz/junyi/junyi.html
- https://github.com/bigdata-ustc/EduData/blob/master/docs/analysis/junyi/junyi.ipynb

### Trois pièges à traiter explicitement

1. **Prendre Junyi15, pas la version Kaggle 2020.** Dans le dataset partagé sur Kaggle en 2020, les timestamps sont arrondis au quart d'heure, ce qui empêche de reconstruire l'ordre exact des réponses. Comme l'ordre des questions est précisément l'objet du projet, cette version est inutilisable ici.
2. **Le graphe est entre exercices, pas entre concepts abstraits.** Il faut agréger vers 8–14 nœuds.
3. **Extraire un sous-graphe connexe**, jamais 10 exercices tirés au hasard. Un ensemble de nœuds sans arêtes donne `Z = 2^n` : l'espace de connaissance explose et perd tout son intérêt.

### Tâches

**A1 — Script d'extraction** (`data/extract_junyi.py`)
Entrée : les fichiers bruts Junyi15. Sortie : un `domain.yaml` (concepts + prérequis) et un `responses.parquet` (`student_id`, `concept_id`, `correct`, `timestamp`).
Contraintes : sous-graphe connexe, 8 à 14 concepts, `|Z|` entre 30 et 500. Si `|Z| > 2000`, s'arrêter et signaler.

**A2 — Calibration EM des paramètres BLIM**
Estimer `slip_q` et `guess_q` par concept à partir des logs, par espérance-maximisation sur le modèle
`P(correct | z, q) = 1 − slip_q si q ∈ z, sinon guess_q`.
Prior sur `Z` estimé conjointement. **Imposer `slip_q + guess_q < 1` à chaque itération** : sans cette contrainte, l'EM peut converger vers une solution où répondre juste est une preuve de non-maîtrise, et l'inférence s'inverse silencieusement.
Vérifier la convergence de la log-vraisemblance, et tester la stabilité sur plusieurs initialisations.

**A3 — Benchmark**
Rejouer les séquences de réponses réelles avec trois politiques : sélection par gain d'information, sélection aléatoire, et un CAT-IRT (θ unidimensionnel, 3PL, critère d'information de Fisher) comme baseline.
Métriques : nombre de questions pour atteindre un seuil de confiance donné, précision du diagnostic final, courbe `H(p_t)` moyenne.
Livrable : un tableau et une figure exploitables tels quels dans le mémoire.

### Alternative si Junyi bloque

Eedi / NeurIPS 2020 Education Challenge (https://arxiv.org/abs/2007.12061). QCM diagnostiques à 4 options, une seule correcte — ce qui donne `guess = 0.25` directement. Les tâches 3 et 4 incluent le texte des questions, les tâches 1 et 2 non.
Inconvénient : la taxonomie Eedi est une **hiérarchie de sujets**, pas un graphe de prérequis. Il faudrait le construire à la main, ce qui annule l'avantage principal de la piste A.

---

## PISTE B — La banque de la démo

### Protocole de constitution

1. **Générer 8 questions par concept, n'en garder que 5** après relecture humaine. Le surplus permet d'écarter les ratées sans se retrouver à court.
2. **Une question = un seul concept.** Le BLIM repose sur l'hypothèse d'indépendance locale ; une question mobilisant deux concepts casse le modèle de vraisemblance. Si le rattachement à un concept unique est ambigu, écarter la question.
3. **QCM à 4 options, distracteurs incarnant une erreur de raisonnement identifiable.** Ce n'est pas cosmétique : un distracteur manifestement absurde est éliminé d'office par l'étudiant, ce qui fait passer le `guess` réel de 0.25 à 0.33 ou plus. Le paramètre devient faux et le diagnostic dérive.
4. **Varier la difficulté à l'intérieur de chaque concept.** Cinq questions quasi identiques sur un même concept ne laissent aucun choix au sélecteur — même pathologie que le couplage concept/question, un cran plus bas.
5. Valeurs initiales en l'absence de calibration : `guess = 1/nb_options`, `slip = 0.10`. Les marquer explicitement comme non calibrées dans le YAML.

### Amorce disponible

Les 30 questions du quiz existant d'un encadrant (https://quiz-adaptative.vercel.app — algèbre 6, analyse 12, géométrie 3, probabilités 3, général 6). Elles sont taguées par domaine et non par concept avec prérequis, et 30 est insuffisant, mais les réutiliser donne un argument fort pour le benchmark : **items identiques, deux algorithmes**. Personne ne peut alors objecter que l'écart vient de la banque plutôt que de la méthode.

### Tâches

**B1 — Schéma de configuration** (`domains/<nom>.yaml`)
Le domaine et la banque doivent être **entièrement en configuration, jamais en dur dans le code**. Structure attendue :

```yaml
domain: <nom>
concepts:
  - id: fractions
    label: "Fractions"
    slip: 0.10
    guess: 0.25
    calibrated: false
prerequisites:
  - [fractions, equations1]
questions:
  - id: q001
    concept: fractions
    difficulty: 2
    stem: "..."
    options: ["...", "...", "...", "..."]
    answer: 0
    distractor_rationale: ["...", "...", "..."]
```

**B2 — Validateur** (`domains/validate.py`)
Vérifier : absence de cycle dans les prérequis, `slip + guess < 1` pour chaque concept, au moins 3 questions par concept, `|Z|` dans la fourchette, index de réponse valide, unicité des identifiants.
Afficher `|Z|`, `2^n`, `H(p₀)` et la distribution des questions par concept.

**B3 — Chargement dans le moteur**
`Domain.from_yaml(path)`. Aucune régression sur les tests existants.

---

## Ordre d'exécution recommandé

**Piste B d'abord.** Extraire et nettoyer Junyi représente facilement deux semaines. La piste B donne une démo qui tourne rapidement. La piste A n'est lancée que si le calendrier le permet.

Un moteur correct avec des paramètres assumés comme non calibrés vaut mieux qu'un pipeline de données à moitié terminé. Si la piste A n'aboutit pas, le mémoire doit dire clairement que `slip`/`guess` sont des valeurs expertes non calibrées — c'est une limite honnête, pas un échec.

---

## Première action attendue

Ne rien coder tout de suite. Proposer d'abord :
1. le schéma YAML définitif (B1), en signalant ce qui manque dans ma proposition ci-dessus ;
2. une estimation du travail réel de la piste A, pour décider si elle tient dans le calendrier.

Le domaine concret n'est pas encore fixé. Travailler sur le domaine jouet de `kst_engine.py` en attendant, et faire en sorte que le changement de domaine soit un simple changement de fichier YAML.
