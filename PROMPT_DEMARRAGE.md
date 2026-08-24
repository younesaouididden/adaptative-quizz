# Prompt de démarrage — Plateforme de quiz adaptatif (PFA UM6P)

> À coller tel quel dans Claude Code / Sonnet 5, **avec le fichier `kst_engine.py` en pièce jointe ou dans le repo**.

---

## Ton rôle

Tu es mon binôme de développement sur un projet de fin d'année (PFA) à l'UM6P Vanguard Center / CMSIS. On construit une **plateforme de quiz adaptatif** qui implémente la théorie développée dans la monographie *Foundations of Learning Systems Theory* d'Ahmed Ratnani (encadrant du projet, avec S. Ibnjaa, S. Kharou et A. Zahir).

Ce n'est pas un projet web générique : le moteur doit être **fidèle à la théorie**, parce que le mémoire devra justifier chaque choix algorithmique par le chapitre correspondant. Quand tu hésites entre une solution élégante en ingénierie et une solution fidèle au formalisme, dis-le-moi explicitement au lieu de trancher seul.

---

## 1. La théorie, en assez de détail pour coder juste

Le système ne sait pas ce que l'étudiant sait. Il maintient une **croyance** — une distribution de probabilité sur les états de connaissance possibles — pose la question qui **réduit le plus son incertitude**, met à jour par Bayes, et recommence.

Quatre couches, dans cet ordre :

### Couche 1 — Combinatoire : l'espace des connaissances `Z` (chapitre 2)

Un domaine est un ensemble fini d'items `P = {p₁, …, pₙ}`. A priori `2ⁿ` états possibles, mais les **prérequis** en éliminent la plupart.

Exemple : `P = {a, b, c}` où *b* exige *a*, et *c* est indépendant →
`Z = {∅, {a}, {c}, {a,c}, {a,b}, {a,b,c}}`. L'état `{b}` seul est infaisable.

- **Relation de surmise** `⪯` : `a ⪯ b` signifie « *a* est prérequis de *b* ». Réflexive et transitive.
- `Z` = les sous-ensembles **clos vers le bas** pour `⪯`. Cette construction garantit automatiquement l'**union-closure** (`z₁, z₂ ∈ Z ⟹ z₁ ∪ z₂ ∈ Z`), donc `Z` est bien un *espace de connaissance* au sens de Doignon–Falmagne.
- Un **learning space** ajoute l'accessibilité et la bonne graduation (deux états reliés par une chaîne différant d'un seul item). Utile pour recommander « le prochain concept à apprendre ».

C'est cette réduction de `2ⁿ` à `|Z|` qui rend l'inférence temps-réel possible.

### Couche 2 — Probabilité : la croyance et le modèle de réponse (chapitre 3)

La croyance est un point `p ∈ Δ(Z)`, le simplexe des distributions sur `Z`.

Modèle d'émission : le **BLIM** (Basic Local Independence Model) —

```
P(correct | z, q) = 1 − slip_q   si q ∈ z
                  = guess_q      sinon
```

C'est l'analogue discret du modèle IRT 3PL : `guess_q` joue le rôle du paramètre *c*, `slip_q` celui du plafond. **Contrainte de validité : `slip_q + guess_q < 1`**, sinon répondre juste devient une preuve de non-maîtrise et l'inférence s'inverse silencieusement.

Mise à jour bayésienne :

```
p_{t+1}(z) = P(y|q,z) · p_t(z) / Σ_{z'} P(y|q,z') · p_t(z')
```

*Exemple de référence à garder comme test unitaire* : prior 0.50 sur « maîtrise les intégrales », `slip = 0.10`, `guess = 0.20`, réponse correcte → posterior **0.818**.

### Couche 3 — Géométrie : pourquoi ce n'est pas de l'arithmétique naïve (chapitres 4–5)

Ce n'est pas directement du code, mais ça impose deux contraintes non négociables.

Le simplexe n'est **pas** euclidien. Passer de `(0.99, 0.01)` à `(0.98, 0.02)` et passer de `(0.51, 0.49)` à `(0.50, 0.50)` ont exactement la même distance euclidienne (0.01414) mais un sens statistique opposé. La bonne structure locale est la **métrique de Fisher** `gᵢⱼ`, unique métrique invariante par statistiques suffisantes (théorème de Chentsov, 1972). Elle apparaît comme la hessienne du KL : `D_KL(p_θ ‖ p_{θ+dθ}) = ½ Σ gᵢⱼ dθⁱ dθʲ + O(‖dθ‖³)`.

Conséquences pour le code :

1. **La mesure de progrès affichée doit être entropique (KL, entropie), jamais une différence de pourcentages.**
2. **Le bord du simplexe est à distance infinie.** Une probabilité qui atteint exactement 0 est *irréversiblement* éliminée. Il faut de l'ε-smoothing partout. Attention : lisser uniquement le dénominateur du KL (`log(P/(Q+ε))`) **biaise l'estimateur et peut le rendre négatif** — lisser les deux distributions puis renormaliser (`p ← (p+ε)/(1+|Z|ε)`).

### Couche 4 — Contrôle : la sélection de question (chapitres 6–7)

C'est un **POMDP** : état caché `z ∈ Z`, action `a ∈ A` (la question), observation `y` (juste/faux), état d'information `p ∈ Δ(Z)`.

Critère glouton de sélection :

```
IG(a; p) = E_{y ~ P(·|a,p)} [ D_KL(p_a^y ‖ p) ]      π*(p) = argmax_a IG(a; p)
```

**Important** : `IG` s'écrit de trois façons équivalentes — l'espérance de KL ci-dessus, la réduction d'entropie `H(p) − E_y[H(p_a^y)]`, et l'information mutuelle `I(Z;Y|a)`. **Implémente la version entropie**, elle est nettement plus stable numériquement.

Coût exact : `O(|Z| · |Y|)` par question candidate. Avec des réponses binaires et `|Z| < 1000`, **le calcul exact est instantané — pas besoin de Monte Carlo**. L'estimateur Monte Carlo (échantillonner N réponses de la prédictive et moyenner les KL) reste dans le code comme test de cohérence et pour le jour où les réponses seront à choix multiples.

---

## 2. Décisions d'architecture déjà prises — ne pas les rouvrir sans me demander

| Décision | Choix | Raison |
|---|---|---|
| Modèle d'apprenant | **KST discret**, `p ∈ Δ(Z)` | c'est le sujet du PFA ; démarre sans données de calibration ; diagnostic par concept et non score global |
| IRT | **baseline de comparaison uniquement** | un encadrant a déjà fait la version IRT pure (`quiz-adaptative.vercel.app`) ; on ne la double pas, on se compare à elle |
| Backend | **Python + FastAPI** | le moteur est numérique, numpy est non négociable |
| Frontend | **Next.js** | |
| Taille du domaine V1 | **8 à 14 concepts** | garde `|Z|` entre ~30 et ~500 |
| Sélection | **gloutonne un pas** | le POMDP multi-pas est intraitable ; la politique myope est le cas `T=1` de l'équation de Bellman, c'est théoriquement justifié |

**Résultat expérimental visé pour le mémoire** : *« à précision diagnostique égale, la sélection par gain d'information sur Δ(Z) demande N questions de moins que le CAT-IRT. »* Le benchmark n'est donc pas un bonus, c'est un livrable.

---

## 3. Ce qui existe déjà : `kst_engine.py`

Environ 250 lignes, sans dépendance hors numpy. Contenu :

- `build_knowledge_space(items, prereqs)` — clôture transitive puis sous-ensembles clos vers le bas, avec détection de cycles
- `Item` (slip/guess, validation `slip+guess<1`), `Domain` (précalcule la matrice de vraisemblance `L[z,q]`)
- `bayes_update` — **en log-probabilités**, avec stabilisation par le max
- `item_marginals` — `P(item i maîtrisé) = Σ_{z ∋ i} p(z)`
- `entropy`, `information_gain_exact`, `information_gain_mc`
- `select_next` (argmax glouton), `should_stop` (3 critères : budget, confiance, IG résiduel)
- `simulate` — étudiant synthétique dont on connaît l'état réel

**Tests qui passent déjà** — ne les casse pas :

```
exemple de la monographie   : 0.500 → 0.818          ✓
Monte Carlo (5000) vs exact : 0.3177 vs 0.3162       ✓
domaine jouet 8 items       : 2^8 = 256 → |Z| = 17
```

**Limite connue et assumée** : sur le domaine jouet, adaptatif = 7.06 questions vs aléatoire = 7.71. L'écart est faible **par construction**, pas par bug — voir le point suivant.

---

## 4. Première tâche : le refactor concept / question

Le moteur actuel fait `1 item = 1 question`. Avec 8 concepts et 8 questions disponibles, un quiz qui pose tout dans le désordre finit au même endroit qu'un quiz intelligent : l'algorithme n'a rien à optimiser. **C'est le défaut de design principal à corriger avant tout le reste.**

Ce qu'il faut :

- séparer `Concept` (nœud du graphe de prérequis, élément de `P`) et `Question` (item de la banque, rattaché à un concept, avec ses propres `slip`/`guess` et son énoncé)
- `Z` se construit sur les **concepts** ; la vraisemblance `L` s'indexe sur les **questions** via leur concept
- viser une banque de 5 à 10 questions par concept → `|A| ≈ 60` pour `|Z| ≈ 17`

Une fois ce découplage fait, la sélection a un vrai choix (quelle *question* sur quel *concept*), et l'écart avec le baseline aléatoire devient significatif. Refais tourner le benchmark après le refactor et montre-moi les chiffres avant/après.

---

## 5. Feuille de route

**Sprint 0 (en cours)** — moteur pur, hors web.
Livrable : refactor concept/question + suite de tests pytest + benchmark adaptatif vs aléatoire vs IRT.
Critère d'acceptation : le diagnostic exact dépasse 85 % et l'écart avec l'aléatoire est net.

**Sprint 1** — API FastAPI.
`POST /sessions` (crée une session, renvoie la première question) · `POST /sessions/{id}/answers` (met à jour `p`, renvoie la suivante ou l'état final) · `GET /sessions/{id}/diagnosis`.
L'état de session est **le vecteur `p` de taille `|Z|`** — c'est tout. Stockage : SQLite au début, Redis ensuite.
Sérialise `p` en float64 ; ne le renvoie jamais au client.

**Sprint 2** — Front Next.js.
Deux règles d'affichage :
- **ne jamais montrer `p` en entier** — afficher les **marginales par concept**
- barre de progression = `1 − H(p_t)/H(p₀)`, c'est-à-dire la part d'incertitude levée. C'est directement l'objet du chapitre 6, pas une métrique inventée.
Écran final : les concepts maîtrisés, ceux qui ne le sont pas, et **le prochain concept accessible** à travailler (un item `q ∉ ẑ` tel que `ẑ ∪ {q} ∈ Z`).

**Sprint 3** — calibration (EM sur les `slip`/`guess` à partir des logs) et évaluation comparative complète.

---

## 6. Pièges à éviter

- Ne jamais faire d'arithmétique de probabilités hors log-space dans la boucle de mise à jour.
- Ne jamais laisser `p(z)` atteindre exactement 0 — c'est irréversible.
- Ne pas reposer deux fois la même question, mais **autoriser** plusieurs questions sur le même concept (c'est tout l'intérêt du refactor).
- Un graphe de prérequis trop dense écrase `|Z|` et rend le diagnostic trivial ; trop clairsemé et `|Z|` explose. Si `|Z| > 2000`, alerte-moi.
- Ne pas implémenter de lookahead multi-pas « pour améliorer » : c'est intraitable et hors périmètre.

---

## 7. Comment je veux qu'on travaille

- Petits incréments testables. Un test pytest pour chaque fonction du moteur, avec les valeurs de référence de la monographie comme cas d'or.
- Commente les fonctions mathématiques avec **le numéro de chapitre correspondant** — ça servira directement à la rédaction du mémoire.
- Quand tu ajoutes une approximation numérique, écris en commentaire ce qu'elle biaise.
- Pas de dépendance lourde sans me demander.

---

## 8. Ce dont j'ai encore besoin de te donner

Le domaine réel n'est pas encore défini. Il me faut :

1. 8 à 14 concepts
2. les paires de prérequis
3. les `slip`/`guess` initiaux (défaut raisonnable : `guess = 1/nb_choix`, `slip = 0.10`)

**Commence par le refactor concept/question sur le domaine jouet existant**, et je te fournirai le domaine réel ensuite — il devra être un simple fichier de configuration (YAML ou JSON), jamais du code en dur.

---

**Première action attendue** : lis `kst_engine.py`, dis-moi ce que tu en penses (y compris ce qui est mal fait), puis propose-moi le plan de refactor concept/question avant d'écrire quoi que ce soit.
