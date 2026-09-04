# Passation — D2 : tester l'hypothèse `z` fixe comme cause du `guess` élevé

> Document de passation, 4 septembre 2026. À coller dans une session ayant accès au dépôt. Suite logique de D0 (contamination par pratique répétée → écartée) et D1 (hétérogénéité par granularité → **écartée**, Lot 4). D2 est la troisième hypothèse, et la dernière encore debout.

---

## 1. À lire d'abord (ne pas redemander, ne pas rouvrir)

- `CONTEXTE_PROJET.md` — état général, et §4 pour l'historique complet de la saga du `guess`.
- `BILAN_CRITIQUE.md` — forces et faiblesses ; ce document-ci développe le §3.2.
- `note_calibration.md` — règle de provenance des paramètres (dérivé / mesuré / balayé, jamais choisi). **S'applique à tout nombre produit ici.**
- `RAPPORT_LOT4.md` — le test D1 et son résultat négatif.

**Priorité relative, à ne pas perdre de vue :** le livrable du stage est le rapport, qui n'est **pas** commencé. D2 est un bonus scientifique, pas un prérequis pour rendre. Si la rédaction n'est pas bien avancée, ne pas lancer D2 — les deux faiblesses concernées se documentent très bien comme limites assumées.

---

## 2. L'énigme, en une page

La calibration EM sur 25,9 M réponses Junyi15 donne des `guess` de 0,50 à 0,75 — implausibles pour du hasard. Deux explications ont été testées et écartées :

- **D0** : contamination par la pratique répétée → mécanisme confirmé mais insuffisant ; un contrôle par échantillon aléatoire de même taille reproduit l'effet (c'était de l'identifiabilité, pas de la contamination).
- **D1** (Lot 4) : hétérogénéité par granularité → **écartée**. Subdiviser `arithmetic` ne fait pas baisser `guess` (`arithmetic_base` 0,759 > `arithmetic` combiné 0,746), et le fit isolé retombe sur l'ancien fit joint.

Reste **D2 : l'hypothèse `z` fixe elle-même.** Le BLIM suppose un état de connaissance constant ; les logs couvrent 2012-10 à 2015-01. Un étudiant qui apprend pendant la fenêtre d'observation a un taux de réussite intermédiaire qu'aucun `z` binaire ne peut expliquer — l'EM doit alors rapprocher `guess` et `1-slip` pour couvrir ces cas intermédiaires.

### Ce que les données disent déjà (mesuré le 4 septembre, à réutiliser)

Taux de réussite bruts par concept sur `responses.parquet`, et mastery marginale **impliquée** par les paramètres calibrés (via `taux = p·(1−slip) + (1−p)·guess`) :

| Concept | taux de réussite | slip | guess | P(maîtrise) impliquée |
|---|---|---|---|---|
| arithmetic | 0,855 | 0,106 | 0,746 | 0,736 |
| probability_statistics | 0,808 | 0,152 | 0,497 | 0,887 |
| analytic_geometry | 0,799 | 0,109 | 0,654 | 0,611 |
| geometry | 0,798 | 0,120 | 0,703 | 0,541 |
| algebra | 0,753 | 0,186 | 0,517 | 0,795 |

**Deux enseignements immédiats :**

1. **Le taux de réussite brut de la plateforme est très élevé (0,75–0,86).** Junyi est une plateforme d'entraînement : on s'exerce jusqu'à réussir. `guess` ne peut pas être bas si les non-maîtrisants réussissent souvent — la question devient : *pourquoi le modèle classe-t-il autant de réussites comme du non-maîtrisé ?*
2. **La piste « le prior s'est effondré sur les états vides » est morte** : la mastery impliquée vaut 0,54 à 0,89 selon le concept. Le prior n'est pas dégénéré. Inutile d'y consacrer du temps.

---

## 3. Le mécanisme précis à tester

Sous BLIM, un étudiant est soit maîtrisant (réussit à `1−slip` ≈ 0,89), soit non-maîtrisant (réussit à `guess`). Un étudiant dont le taux **observé** est intermédiaire (disons 0,60 sur 100 réponses) n'est bien expliqué par aucun des deux — sauf si `guess` remonte vers le milieu. Plus la population contient d'étudiants « intermédiaires », plus `guess` est tiré vers le haut.

**L'hypothèse D2 dit que ces taux intermédiaires sont d'origine temporelle** : l'étudiant échoue tôt, réussit tard, et sa moyenne sur 2 ans tombe au milieu. Si c'est vrai, l'intermédiarité doit être **ordonnée dans le temps**, pas dispersée au hasard.

C'est ça, la prédiction falsifiable — et elle se teste sans lancer un seul EM.

---

## 4. Protocole, du moins cher au plus cher

Les timestamps sont exploitables directement dans `data/responses.parquet` (colonne `timestamp`, microsecondes depuis epoch, 2012-10-12 → 2015-01-11). **Aucune réextraction du log brut n'est nécessaire** — contrairement à D1.

### D2.1 — Signature temporelle de l'intermédiarité *(décisif, ~1 journée, aucun EM)*

Pour chaque paire (étudiant, concept) ayant assez d'observations (seuil à fixer **avant** de regarder, ex. ≥ 20) :

1. couper l'historique **en deux moitiés par le temps** (pas par le nombre d'essais) ;
2. comparer le taux de réussite première moitié vs seconde moitié ;
3. agréger, et surtout : **restreindre aux étudiants dont le taux global est intermédiaire** (ex. entre 0,4 et 0,8) — ce sont eux qui forcent `guess` vers le haut. La question est : *ces étudiants-là progressent-ils dans le temps, ou leurs réussites sont-elles dispersées ?*

**Prédiction D2** : chez les intermédiaires, hausse systématique et substantielle de la première à la seconde moitié.
**Prédiction concurrente (maîtrise partielle stable)** : pas de tendance temporelle, les réussites sont dispersées.

**Contrôle indispensable** : comparer à une permutation aléatoire de l'ordre temporel au sein de chaque étudiant. Si la hausse mesurée survit à la permutation, c'est un artefact de méthode, pas un signal.

### D2.2 — Calibration par fenêtre temporelle *(confirme la conséquence, ~1-2 jours)*

Si D2.1 confirme, vérifier que ça se traduit bien sur `guess` :

1. recalibrer l'EM en ne gardant que les **N premiers jours d'activité de chaque étudiant** (N ∈ {7, 30, 90, ∞}) ;
2. **prédiction** : `guess` décroît quand la fenêtre se resserre (moins d'apprentissage intra-fenêtre).

**Piège connu, déjà tombé dedans une fois** (`docs/revue_d0_tour2.md`) : une fenêtre courte = moins d'observations par étudiant = identifiabilité dégradée, ce qui fait baisser `guess` **tout seul**, sans rien prouver. Le tour 2 de D0 s'est fait avoir exactement là.

**Contrôle obligatoire** : pour chaque fenêtre, un run apparié sur un **sous-échantillon aléatoire de l'historique complet, de même taille par étudiant**. Si `guess` baisse dans la fenêtre temporelle **mais pas** dans le contrôle de même volume, la baisse est attribuable au temps. Sinon, ce n'est encore que de l'identifiabilité. Le tirage sur comptes agrégés (loi hypergéométrique) utilisé au tour 2 est réutilisable tel quel.

Techniquement : `calibrate_em.calibrate()` prend déjà `domain_path` et `responses_path` en paramètres, et `sufficient_statistics()` agrège après coup — il suffit de **filtrer le DataFrame sur `timestamp` avant** de le passer. Pas de réécriture.

### D2.3 — Borne sur le gain potentiel *(optionnel, pour le mémoire)*

Si D2.1 et D2.2 confirment : estimer ce que vaudrait `guess` sous un modèle autorisant **une seule transition** de `z` par (étudiant, concept). Même un calcul approché donnerait un chiffre à opposer aux 0,746 actuels, et transformerait « limite assumée » en « limite quantifiée ». Implémenter le modèle dynamique complet reste **hors périmètre** du PFA (c'est la perspective future déjà annoncée dans `calibrate_em.py`).

---

## 5. Ce qui tuerait l'hypothèse

À écrire noir sur blanc **avant** de lancer quoi que ce soit, pour ne pas réinterpréter après coup :

- **D2.1 ne montre aucune tendance temporelle chez les intermédiaires** (ou une tendance qui survit à la permutation) → l'hypothèse temporelle est morte. La cause serait alors une maîtrise partielle *stable* — que le BLIM binaire ne peut pas représenter, ce qui est une conclusion différente et tout aussi publiable.
- **D2.2 : `guess` baisse autant dans le contrôle de même volume que dans la fenêtre temporelle** → c'est de l'identifiabilité, pas du temps. Même verdict qu'au tour 2 de D0.
- **`guess` reste ≥ 0,70 même sur une fenêtre de 7 jours avec contrôle propre** → aucune des deux lectures ne tient, et il faut chercher ailleurs (format des exercices Junyi ? indices/aides intégrés à la plateforme ? réponses à choix très restreint ?).

Le résultat négatif est un résultat. D1 en est déjà un, il est rapporté tel quel, et ça n'a rien coûté à la crédibilité du travail — au contraire.

---

## 6. Conventions à respecter (déjà établies)

- Sorties dans `results/<experience>/` avec `raw.csv`, `summary.csv`, `figure.pdf` (vectoriel), `run.log` (graines + commit git). Ajouter une ligne à `results/REFERENCE.md`.
- Graines déterministes via `np.random.SeedSequence([...])`, jamais `hash()` (non stable entre processus pour les chaînes).
- Un test pytest par fonction, sur fixtures synthétiques — jamais sur les fichiers réels de plusieurs millions de lignes.
- Seuils et critères fixés **avant** de regarder les résultats, et appliqués uniformément.
- RAM limitée sur cette machine : lecture par chunks / statistiques agrégées, jamais le log brut en mémoire d'un coup.
- Sur Windows, `git push`/`commit` peuvent dépasser un timeout de 2 min sans avoir échoué : vérifier `git status` avant de conclure à un échec.
- Les caractères hors cp1252 (`≈`, `∅`, …) dans un `print()` font planter le script sur cette console — les garder pour les figures et les fichiers, pas pour la sortie standard.

---

## 7. Première action attendue

**Ne pas lancer d'EM.** Commencer par **D2.1**, qui est cheap, ne dépend d'aucun recalcul lourd, et discrimine directement entre les deux lectures (apprentissage temporel vs maîtrise partielle stable). Fixer le seuil d'observations minimum et la bande « intermédiaire » avant de regarder les données, puis mesurer, avec le contrôle par permutation.

Si D2.1 est négatif, s'arrêter là et documenter : l'énigme du `guess` aura été réduite à une seule explication restante (maîtrise partielle non représentable par un `z` binaire), ce qui est un bon résultat de mémoire et referme proprement la saga.
