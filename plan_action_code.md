# Plan d'action — axe code & résultats numériques (PFA KST)

> Établi le 2 septembre 2026. Complète `RAPPORT_AVANCEMENT.md`.
> Répartition : Yassine = théorie et rédaction ; Younes = code, données, résultats numériques.
> Un seul rapport commun. Pas de contrainte de deadline forte.

## Vue d'ensemble

Le socle technique est fait (moteur stable, calibration EM sur Junyi, banque de démo, benchmark à trois politiques, 118 tests). Ce qui reste n'est plus de la construction mais de l'**alignement avec le rapport** : produire les résultats que la partie théorique annonce sans les fournir, et exprimer les résultats existants dans le vocabulaire de la théorie.

Ordre d'exécution recommandé : **Lot 0 → Lot 5 → Lot 1 → Lot 2 → Lot 3 → Lot 4**.
Le Lot 5 (vocabulaire) passe tôt parce qu'il fixe les noms de colonnes et de fonctions ; le faire après aurait produit une pile de CSV à renommer.

---

## Lot 0 — Infrastructure de livraison

**Pourquoi.** Le rapport est écrit par quelqu'un d'autre. Sans traçabilité, un chiffre recopié à la main dérive d'une version à l'autre et la dernière semaine part en réconciliation de tableaux.

- `results/<experience>/` : un dossier par expérience, contenant le CSV brut, la ou les figures, et le log d'exécution.
- Figures en **PDF vectoriel** (le rapport est en LaTeX), pas en PNG.
- Graines aléatoires fixées et écrites dans le CSV.
- `results/REFERENCE.md` : un tableau associant chaque figure et chaque chiffre du rapport à son script, sa graine, son commit git et sa date.
- Un script = une expérience = une commande reproductible.

---

## Lot 5 — Vocabulaire et traduction (à faire tôt)

**Pourquoi.** Les deux moitiés du PFA décrivent les mêmes objets avec des mots différents. Sans harmonisation, le rapport se lit comme deux documents agrafés.

Correspondances à acter :

| Théorie (Yassine / monographie) | Code (Younes) |
|---|---|
| `p_t ∈ ∆(Z)`, état de croyance | vecteur de probabilités sur `Z` |
| Transition bayésienne, éq. (1) | `bayes_update` |
| `π*(p) = argmax_a IG(a; p)`, politique optimale exacte | `select_next` |
| `IG(a;p) = E_y[D_KL(p^y_a ‖ p)]` | gain d'information exact |
| ε-lissage | lissage symétrique en log-espace |
| Action pédagogique `a ∈ A` | question de la banque |
| Probabilité d'émission `P(y|a,z)` | modèle BLIM (`slip`, `guess`) |

- Renommer dans le code, ou au minimum publier cette table dans le README et en annexe du rapport.
- Relégender **toutes** les figures existantes (Fig. 1, 2, 3) avec la notation de la monographie.
- Point à affirmer dans le rapport : l'implémentation calcule `π*` **exactement**, pas une version simplifiée. C'est la politique que la théorie déclare intractable — d'où la valeur du Lot 1.

---

## Lot 1 — Validation de l'approximation Monte Carlo (chantier phare)

**Pourquoi.** La présentation d'août pose l'estimateur Monte Carlo comme la solution d'ingénierie et annonce que le choix de `N` est un compromis biais-variance-temps — sans aucun chiffre. Mesurer l'erreur d'approximation exige la vérité exacte comme référence : seul l'axe code peut le faire.

**Préalable — question à trancher avec Yassine avant de coder.**
L'estimateur de la diapo 7 échantillonne les *réponses* `y ~ P(·|a,p)`. Pour un item binaire, `|Y| = 2` : la somme exacte a deux termes, et l'échantillonner est strictement pire que la calculer. Par ailleurs chaque `y` tiré exige quand même une mise à jour bayésienne complète sur tout `Z` — l'estimateur n'attaque donc jamais le terme qui explose réellement, qui est `|Z|`. Deux résolutions possibles :
- soit une « action » désigne un **bloc** de `k` questions, et alors `|Y| = 2^k` et l'échantillonnage sur `y` se défend ;
- soit il faut échantillonner les **états** `z ~ p(z)`, ce qui attaque le vrai goulot.

Le plan ci-dessous implémente **les deux** et les compare, ce qui répond à la question au lieu de la contourner.

### 1.1 Implémentation
`information_gain_mc(a, p, N, mode)` avec `mode ∈ {"sample_y", "sample_z"}`, en gardant `information_gain_exact` comme référence.

### 1.2 E1 — fidélité de la politique
Sur un échantillon de croyances `p` **issues de vraies trajectoires** (pas de priors uniformes artificiels), pour `N ∈ {1, 3, 5, 10, 30, 100}` :
- taux d'accord `argmax_MC == argmax_exact` ;
- **regret en IG** : `IG_exact(a_MC) / IG_exact(a*)`.

Le taux d'accord seul est trompeur — choisir la deuxième meilleure question quand elle est presque équivalente ne coûte rien. Le regret est la métrique honnête.

### 1.3 E2 — coût en aval
Rejouer le benchmark complet avec `π̂_N` au lieu de `π*`, pour chaque `N`. Mesures : nombre de questions, exactitude par concept.

### 1.4 E3 — coût de calcul en fonction de `|Z|`
Temps par décision, exact vs MC(`N`), sur des domaines synthétiques de 5, 7, 9, 11, 13 concepts pour balayer `|Z|` de 18 à quelques milliers. C'est ce qui prouve que l'approximation sert à quelque chose : à `|Z| = 50` elle ne sert à rien.

### 1.5 Figure de justification
`|Z|` en fonction du nombre de concepts (domaines réels : 5 → 18, 7 → 50 ; plus les synthétiques). Cette seule courbe justifie l'existence du chapitre d'approximation.

**Livrable attendu :** une recommandation d'ingénierie chiffrée — « en dessous de tel `|Z|`, calculer l'exact ; au-delà, MC sur `z` avec `N ≈ …` », et pas seulement une mesure.

---

## Lot 2 — Ancrage géométrique des résultats

**Pourquoi.** Les chapitres 4-5 du rapport portent sur Fisher-Rao, les géodésiques et le gradient naturel. Les résultats actuels ne contiennent aucun objet géométrique.

### 2.1 Distance de Fisher-Rao
Forme fermée sur le simplexe : `d(p,q) = 2·arccos(Σ_z √(p(z)·q(z)))`. Une ligne de numpy. Plus la longueur d'arc cumulée le long d'une trajectoire.

### 2.2 Figure — longueur d'arc cumulée
Distance parcourue sur la variété en fonction du numéro de question, adaptatif vs aléatoire, moyennée sur les états. Attendu : l'adaptatif parcourt davantage de distance par question au début.

### 2.3 Figure — trajectoire sur la sphère de Fisher-Rao
Restreindre à un sous-domaine à 3 états et tracer la trajectoire `p_0 → p_T` sur l'octant positif via la carte racine `x = 2√p`. C'est exactement l'illustration de la diapo 5 de la semaine 2, mais avec de vraies trajectoires. Candidate naturelle pour la figure d'ouverture du rapport.

### 2.4 Information de Fisher d'un item en fonction de (`slip`, `guess`)
Courbe, avec les valeurs piste A (`guess` 0,50–0,75) et piste B (`guess` 0,25) marquées dessus.

**Effet narratif.** Quand `guess` monte, les probabilités d'émission `P(y|a,z)` pour des états `z` différents se rapprochent, l'item cesse de séparer les états, son information de Fisher s'effondre, et le déplacement géométrique par question tend vers zéro. La « saga du `guess` » cesse d'être une limite gênante et devient une **prédiction quantitative de la théorie vérifiée sur données réelles** — le seul résultat empirique de tout le PFA.

---

## Lot 3 — Robustesse du benchmark à une mauvaise spécification

**Pourquoi.** Aujourd'hui, un seul jeu de paramètres génère les réponses **et** est supposé par le moteur. C'est l'objection la plus facile à formuler pour un jury.

### 3.1 Découplage
Deux jeux distincts : `params_verite` (génère les réponses) et `params_moteur` (ce que le moteur croit, figé à `slip=0,10` / `guess=0,25`). Modification de signature dans le simulateur, pas une réécriture.

### 3.2 Bruit mal estimé
Grille sur la vérité : `slip ∈ {0,05 · 0,10 · 0,20 · 0,30}` × `guess ∈ {0,25 · 0,40 · 0,55 · 0,70}`. Le cas dangereux est asymétrique : un monde plus bruité que ce que le moteur croit le rend trop confiant et le fait s'arrêter trop tôt. La cellule `guess = 0,70` correspond au régime mesuré en piste A.

### 3.3 Graphe de prérequis faux
Générer 15 % d'étudiants dont l'état vrai `z ∉ Z` (par ex. `algebra_advanced` maîtrisé sans `algebra_linear`). Le moteur ne peut pas représenter cet état. Métrique : distance de Hamming entre état inféré et état vrai, pas seulement exact/pas exact.

### 3.4 Arène miroir
Générer les réponses depuis un 3PL à `θ` continu et faire tourner les deux politiques dessus. IRT gagnera probablement — et c'est le résultat voulu. Le message devient « chaque modèle domine sur sa propre vérité générative ; la vraie question est laquelle décrit les données Junyi », ce qui renvoie proprement vers la piste A et désamorce l'objection de l'arène truquée.

### 3.5 Calibration de la confiance
Diagramme de fiabilité : confiance annoncée à l'arrêt en abscisse, proportion réelle d'états exactement corrects en ordonnée, avec la diagonale. Sous mauvaise spécification, « 90 % sûr » ne veut plus dire « juste 9 fois sur 10 ».

### 3.6 Correction
Rejouer la grille avec un moteur volontairement prudent (`slip=0,20` / `guess=0,40` supposés) et montrer que ça restaure la calibration au prix de quelques questions. Un mode de défaillance **et** son remède vaut mieux qu'un constat.

### Protocole commun
Comparaison appariée : mêmes états vrais et mêmes graines pour les trois politiques dans chaque cellule. 50 états × ~20 répétitions par état, avec écart-type reporté.

---

## Lot 4 — Boucler la piste A

- **A3 sur le domaine calibré** (5 concepts, `|Z| = 18`, `guess` 0,50–0,75). Attendu : dégradation forte du gain d'efficacité, ce qui se relie directement au Lot 2.4.
- **Test D1 (hétérogénéité).** Ne pas re-tester tout le domaine : prendre le seul bucket `arithmetic`, le subdiviser comme en piste B, relancer l'EM sur ce sous-domaine et regarder si `guess` baisse. Si oui, l'hypothèse passe d'une explication verbale à une preuve chiffrée, et la subdivision à 7 concepts de la piste B se trouve justifiée a posteriori.

---

## Lot 6 — Reste administratif

- Redéployer l'app Streamlit après la bascule piste A → piste B.
- Donner l'accès au collègue (Share → e-mail, ou collaborateur GitHub).

---

## Deux limites à encadrer explicitement dans le rapport

1. **Dynamique de la croyance ≠ dynamique de la connaissance.** Le titre parle de contrôle optimal des dynamiques d'apprentissage, mais `z` est **fixe** dans toute l'implémentation : c'est la croyance qui évolue sur un état statique. Ce sont deux flots différents sur `∆(Z)`. À poser en introduction comme limitation assumée, pas en note de bas de page.
2. **Le benchmark A3 mesure un gain algorithmique, pas une validation empirique des paramètres.** Les paramètres IRT sont dérivés de la banque BLIM (discrimination fixée à 1,0), pas calibrés indépendamment. La validation empirique reste du ressort exclusif de la piste A.
