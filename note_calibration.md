# Note méthodologique — d'où viennent les nombres (`slip`, `guess`)

> 2 septembre 2026. Annexe au plan d'action, destinée à être reprise dans le rapport.
> Lecture de la consigne : **aucun nombre choisi à la main, aucun nombre ajusté pour que le résultat sorte bien.** La calibration EM sur Junyi n'entre pas dans cette catégorie — c'est une mesure, avec son incertitude. Les valeurs « expertes » de la piste B, si.

## 1. La règle

Tout paramètre du modèle doit tomber dans exactement une de ces trois cases, et la case doit être écrite noir sur blanc à côté du nombre :

| Provenance | Statut | Exemple dans le projet |
|---|---|---|
| **Dérivée** — conséquence de la structure du problème | Défendable telle quelle | `guess ≤ 1/k` pour un QCM à `k` options |
| **Mesurée** — estimée sur données, avec incertitude reportée | Défendable, à condition de publier l'incertitude | `guess` piste A, EM sur 25,9 M réponses, écart-type < 0,001 sur 5 redémarrages |
| **Balayée** — aucune valeur privilégiée, on montre l'invariance | Défendable, et c'est le seul traitement honnête du reste | `slip` |

Une quatrième case existe et est interdite : **choisie**. Un nombre posé « parce que ça marche mieux comme ça » invalide tout résultat qui en dépend. C'est exactement le statut actuel de `slip = 0,10` en piste B, et il faut le corriger.

## 2. `guess` n'est pas un paramètre libre

Pour un QCM à `k` options avec une seule bonne réponse, un étudiant qui ne maîtrise pas le concept et répond au hasard uniforme a une probabilité `1/k` de tomber juste. Ce n'est pas une estimation, c'est de la combinatoire.

Mais l'hypothèse de hasard uniforme est fausse **dans le bon sens** : les distracteurs de la banque piste B ont été écrits pour incarner une erreur de raisonnement plausible. Un étudiant qui ne maîtrise pas n'est donc pas indifférent entre les options — il est *attiré* par le distracteur. La vraie probabilité de réussite par hasard est donc inférieure à `1/k`.

**Énoncé rigoureux à mettre dans le rapport :**

> `guess ≤ 1/k`, avec égalité si et seulement si les distracteurs sont statistiquement indiscernables de la bonne réponse pour un non-maîtrisant.

Pour `k = 4` : `guess ≤ 0,25`. Utiliser 0,25 est donc un choix **conservateur et borné**, pas une valeur inventée — et c'est exactement comme ça qu'il faut le formuler. Le moteur qui suppose 0,25 alors que la vérité est plus basse est prudent, jamais optimiste.

## 3. `slip` n'a aucune valeur de premier principe

Il n'existe aucun argument combinatoire ou théorique qui fixe `slip`. 0,10 est une convention de la littérature, rien de plus. Deux options honnêtes, une seule bonne :

- ~~Choisir 0,10 et l'annoncer comme « valeur experte »~~ → c'est la case interdite.
- **Balayer `slip` et montrer que la conclusion ne dépend pas de la valeur.** C'est ce que fait déjà le Lot 3 du plan. Il suffit de le présenter comme le traitement principal du paramètre, et non comme un test de robustesse annexe.

Chiffres du balayage (voir §4 pour la formule) :

| `slip` | Information par question | Questions par concept |
|---|---|---|
| 0,02 | 1,822 nats | 1,6 |
| 0,05 | 1,415 nats | 2,1 |
| 0,10 | 1,071 nats | 2,7 |
| 0,15 | 0,850 nats | 3,5 |
| 0,20 | 0,683 nats | 4,3 |
| 0,30 | 0,438 nats | 6,7 |

Le nombre de questions varie d'un facteur 4 sur cette plage, mais **le classement des politiques ne change pas** — c'est ça, la conclusion à défendre, et elle ne demande aucun choix de valeur.

## 4. Combien d'information porte une question — formule et chiffres

C'est le cœur de la note : un nombre entièrement dérivé, qui remplace l'argument verbal sur l'hétérogénéité des concepts.

Une question binaire portant sur un concept a deux lois d'émission :
- étudiant qui maîtrise : réponse juste avec probabilité `1 − slip`
- étudiant qui ne maîtrise pas : réponse juste avec probabilité `guess`

Le log-rapport de vraisemblance dérive en moyenne, par question observée, de la divergence KL entre ces deux lois (argument de Wald / SPRT). En moyennant les deux directions :

```
KL(p‖q) = p·log(p/q) + (1−p)·log((1−p)/(1−q))
Ī(slip, guess) = ½ · [ KL(1−slip ‖ guess) + KL(guess ‖ 1−slip) ]
```

Pour passer d'une croyance de 0,5 à 0,95 sur un concept, il faut déplacer le log-odds de `log(19) ≈ 2,944` nats, soit environ `2,944 / Ī` questions.

| Régime | `slip` | `guess` | `1−slip−guess` | `Ī` (nats/question) | Questions/concept |
|---|---|---|---|---|---|
| **Piste B** (borne QCM 4 options) | 0,10 | 0,25 | 0,65 | **1,071** | **2,7** |
| Piste A, bas de la plage mesurée | 0,10 | 0,50 | 0,40 | 0,439 | 6,7 |
| Piste A, milieu | 0,10 | 0,70 | 0,20 | 0,135 | 21,8 |
| Piste A, haut de la plage mesurée | 0,10 | 0,75 | 0,15 | 0,082 | 35,7 |
| Ancien domaine 9 concepts, pire cas | 0,10 | 0,76 | 0,14 | 0,073 | 40,3 |

**Une question en régime piste A porte environ 8 fois moins d'information qu'en régime piste B.**

La contrainte `slip + guess < 1` du BLIM cesse au passage d'être une contrainte technique arbitraire : c'est exactement la condition `Ī > 0`, c'est-à-dire « l'item apporte de l'information ». Quand `slip + guess → 1`, l'item devient un pur bruit.

## 5. La formule prédit le benchmark déjà mesuré

Piste B, 7 concepts, `slip = 0,10`, `guess = 0,25` :

```
7 × 2,944 / 1,071 = 19,2 questions
```

**Mesure du benchmark A3 : 18,6 questions.**

C'est une prédiction en forme fermée, faite sans regarder le résultat, qui tombe à 3 % de la mesure. À présenter avec ses réserves — le calcul suppose des concepts indépendants et une question par concept, alors que la structure de prérequis permet à la politique adaptative de partager de l'information entre concepts, et l'approximation de Wald ignore le dépassement au franchissement du seuil. L'accord à 3 % tient donc en partie de la chance. Mais l'ordre de grandeur, lui, est acquis, et c'est le premier endroit du projet où la théorie prédit un chiffre expérimental au lieu de le commenter après coup.

## 6. La formule prédit aussi le résultat du benchmark piste A, avant de le lancer

Piste A, 5 concepts :

| `guess` mesuré | Questions nécessaires | Plafond actuel |
|---|---|---|
| 0,50 | 34 | 30 |
| 0,60 | 55 | 30 |
| 0,70 | 109 | 30 |
| 0,75 | 179 | 30 |

Le benchmark A3 sur le domaine calibré **ne peut pas converger** sous un plafond de 30 questions. Ce n'est pas un échec du code ni un mauvais réglage : c'est arithmétique. Deux conséquences :

- Il faut lancer ce benchmark **avec un plafond relevé** (150 questions au moins), sinon on mesure le plafond et non la méthode.
- Le résultat attendu est connu d'avance et reste publiable : il quantifie le coût réel de l'hétérogénéité des concepts, en questions.

## 7. Seuil opérationnel

À quelle valeur de `guess` un quiz devient-il infaisable ? Avec `slip = 0,10`, en résolvant `2,944 / Ī(0,10, g) = budget` :

- **3 questions par concept** → `guess < 0,278`
- **5 questions par concept** → `guess < 0,427`

La banque piste B a 5 questions par concept, donc son budget tient tant que `guess < 0,43`. La calibration piste A donne 0,50 à 0,75. **C'est le chiffre qui explique tout le projet en une ligne : les concepts de la piste A sont au-delà du seuil de faisabilité, ceux de la piste B sont en deçà.**

## 8. Ce que ça change

1. La saga du `guess` n'est plus une limite embarrassante : c'est un **seuil de faisabilité franchi**, mesuré sur données réelles, avec un nombre à côté.
2. `guess = 0,25` cesse d'être une « valeur experte » et devient une **borne combinatoire conservatrice**, reformulée comme telle partout dans le code, l'app et le rapport.
3. `slip = 0,10` cesse d'être une valeur et devient un **balayage** ; aucune conclusion du rapport ne doit dépendre du choix d'un point sur cet axe.
4. Le benchmark A3 sur piste A se lance avec un plafond de 150, pas de 30.
5. Ces formules sont à implémenter comme fonctions testées du dépôt (`item_information(slip, guess)`, `questions_needed(slip, guess, n_concepts, target)`), pas comme un calcul jeté dans un coin de note — elles servent trois fois dans le plan (Lots 2.4, 3.2 et 4).
