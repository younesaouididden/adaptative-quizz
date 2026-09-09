# Le code du projet, expliqué

> Ce document explique **ce que fait le code, pourquoi il le fait, et ce qu'on
> a mesuré avec**. Il ne suppose aucune familiarité avec Python : tout est
> expliqué en français, les rares formules sont accompagnées de leur lecture
> en mots. Il ne remplace pas le rapport — c'est le document qui permet de
> lire le rapport en sachant ce qu'il y a derrière.
>
> État au moment de la rédaction : ~5 200 lignes de code, ~2 200 lignes de
> tests, **179 tests automatiques** dont 178 passent et 1 est un échec attendu
> et documenté. Tout est dans un dépôt git, chaque résultat chiffré est
> rattaché au script et à la graine aléatoire qui l'a produit.

---

## 1. L'idée, en une page

Un quiz classique pose les mêmes questions à tout le monde, dans le même
ordre. Un quiz **adaptatif** choisit la question suivante en fonction des
réponses déjà données. Le but : arriver au même diagnostic en posant moins de
questions.

Notre quiz repose sur la **Knowledge Space Theory (KST)**. L'idée centrale :
un élève n'est pas résumé par une note unique, mais par l'**ensemble des
concepts qu'il maîtrise**. Et tous les ensembles ne sont pas possibles : si
« fractions » est un prérequis d'« algèbre », personne ne maîtrise l'algèbre
sans les fractions. Les ensembles réellement possibles forment un catalogue
qu'on appelle **Z**, l'espace des connaissances.

Le programme fait alors trois choses, en boucle :

1. **Il doute de façon chiffrée.** Il ne dit pas « l'élève est ici », il dit
   « il y a 12 % de chances que ce soit l'état n°7, 8 % que ce soit le n°12,
   … » — une probabilité sur chacun des états du catalogue. C'est la
   **croyance**.
2. **Il choisit la question la plus informative.** Pour chaque question encore
   disponible, il calcule à l'avance de combien la réponse (quelle qu'elle
   soit) va réduire son incertitude, et il pose celle qui réduit le plus.
3. **Il met à jour sa croyance** avec la réponse reçue, via la règle de
   Bayes — puis il recommence, jusqu'à être assez sûr.

Tout le reste du code est soit ce qui alimente cette boucle (les domaines, les
questions, les paramètres), soit ce qui la mesure (les expériences), soit ce
qui la vérifie (les tests).

**Résultat principal, en un chiffre** : sur notre domaine de 7 concepts et 35
questions, le quiz adaptatif atteint son diagnostic en **18,6 questions en
moyenne**, contre **27,7** pour un tirage aléatoire des mêmes questions — soit
**33 % de questions en moins, à qualité de diagnostic équivalente**.

---

## 2. Le vocabulaire minimum

Sept termes reviennent partout. Les voici une fois pour toutes.

| Terme | Ce que c'est, en français |
|---|---|
| **Concept** | Une compétence du domaine : « fractions », « géométrie »… Il y en a 7 dans notre domaine principal. |
| **Prérequis** | « A doit être maîtrisé avant B ». Notre domaine en déclare 4. |
| **État de connaissance** | L'ensemble exact des concepts qu'un élève maîtrise. Par exemple {arithmétique, fractions} et rien d'autre. |
| **Z (espace des connaissances)** | Le catalogue de tous les états **possibles**, c'est-à-dire compatibles avec les prérequis. |
| **Croyance** | Une probabilité attribuée à chaque état de Z. Elles font 1 au total. C'est l'état interne du programme. |
| **slip** | Probabilité de rater une question **alors qu'on maîtrise** le concept (inattention). |
| **guess** | Probabilité de réussir une question **sans maîtriser** le concept (chance, QCM). |

Le couple (slip, guess) définit le **modèle de réponse**, appelé **BLIM**
(*Basic Local Independence Model*) : une question portant sur un concept
maîtrisé est réussie avec probabilité « 1 − slip », une question portant sur un
concept non maîtrisé est réussie avec probabilité « guess ». C'est tout le
modèle.

**Une contrainte qui a l'air technique mais qui est le cœur du sujet** :
il faut slip + guess < 1. Sinon, réussir une question deviendrait une preuve
de **non**-maîtrise et tout le raisonnement s'inverserait. Le code refuse de
construire une question qui violerait cette contrainte — ce n'est pas un
garde-fou cosmétique : c'est exactement la condition « cette question apporte
de l'information » (voir §4.7).

---

## 3. Un chiffre à retenir sur la taille de Z

Avec 7 concepts, il existe 2⁷ = **128** ensembles de concepts imaginables.
Mais une fois les 4 relations de prérequis appliquées, il n'en reste que
**50** qui sont réellement possibles.

C'est le premier bénéfice concret de la théorie : **la structure de prérequis
divise par 2,5 l'espace à explorer, avant même d'avoir posé une seule
question.** Sur le domaine issu des données réelles (5 concepts, 2
prérequis), on passe de 32 à 18 états.

Cette réduction est aussi la principale difficulté de calcul du projet : le
nombre d'états croît exponentiellement avec le nombre de concepts, et chaque
décision du programme parcourt tout Z. C'est ce qui motive une bonne partie
des expériences (§7).

---

## 4. Le moteur : `kst_engine.py`

C'est le fichier central, ~730 lignes, et il ne dépend que de `numpy` (la
bibliothèque de calcul numérique standard de Python). Ce choix est délibéré :
le moteur reste lisible, portable, et testable sans installer quoi que ce soit
d'exotique. Tout le reste du projet l'utilise sans jamais le réimplémenter —
l'application web, les expériences, les benchmarks appellent tous ces mêmes
fonctions.

### 4.1 Construire le catalogue Z

**Fonction : `build_knowledge_space`.**

On lui donne la liste des concepts et la liste des prérequis. Elle répond par
la liste de tous les états possibles.

Elle procède en deux temps :

1. **Elle complète les prérequis implicites.** Si on déclare « A avant B » et
   « B avant C », elle en déduit toute seule « A avant C ». (C'est ce qu'on
   appelle la *clôture transitive*.)
2. **Elle filtre.** Elle passe en revue les 2ⁿ ensembles imaginables et ne
   garde que ceux où, pour chaque concept présent, tous ses prérequis sont
   présents aussi.

Au passage, elle **refuse de fonctionner si les prérequis contiennent un
cycle** (« A avant B, B avant A ») : ce serait une incohérence du domaine, et
il vaut mieux une erreur bruyante qu'un catalogue silencieusement faux.

### 4.2 Le modèle de réponse

**Objets : `Concept`, `Question`, `Domain`.**

Un point de conception important : **les concepts et les questions sont deux
choses distinctes**. Z se construit sur les concepts. Les questions, elles,
sont rattachées à un concept — et **plusieurs questions peuvent porter sur le
même concept**.

Pourquoi ça compte : sans cette séparation, une fois chaque concept interrogé
une fois, le programme n'aurait plus rien à choisir. C'est la banque de
plusieurs questions par concept qui donne un vrai sens à la sélection.

Quand on construit un domaine, le code calcule une fois pour toutes un tableau
appelé `L` : pour chaque état possible et chaque question, la probabilité de
réussir cette question dans cet état. C'est « 1 − slip » si le concept de la
question est dans l'état, « guess » sinon. Tout le reste du moteur lit dans ce
tableau au lieu de recalculer.

### 4.3 Mettre à jour la croyance : `bayes_update`

C'est la règle de Bayes, appliquée après chaque réponse : la nouvelle
probabilité d'un état est proportionnelle à son ancienne probabilité
multipliée par la vraisemblance de la réponse observée dans cet état.

Deux précautions, qui ont l'air d'être des détails et n'en sont pas :

- **Le calcul se fait en logarithmes.** Multiplier beaucoup de petits nombres
  entre eux fait sortir un ordinateur de sa plage de précision. Passer par les
  logarithmes transforme les multiplications en additions et supprime le
  problème.
- **On ajoute une quantité minuscule (ε = 10⁻¹²) partout avant de combiner.**
  Raison : si la probabilité d'un état tombe à **exactement** zéro, elle y
  reste pour toujours — aucune réponse future ne peut la faire remonter.
  L'état est éliminé définitivement, par un arrondi de calcul plutôt que par
  une preuve. Le lissage l'empêche.

  **Le point délicat** : on lisse **les deux** quantités (la croyance *et* la
  vraisemblance), jamais une seule. Ne lisser qu'un côté introduit un biais
  systématique — et, dans le calcul du gain d'information, peut produire des
  valeurs négatives, ce qui n'a aucun sens pour une quantité d'information.
  C'est une erreur classique ; elle est documentée dans le code à l'endroit où
  elle serait tentante.

### 4.4 Ce qu'on montre à l'utilisateur : `concept_marginals`

La croyance porte sur 50 états — illisible pour un humain. Cette fonction la
traduit en une phrase par concept : « P(fractions maîtrisées) = 0,87 », en
additionnant les probabilités de tous les états qui contiennent ce concept.

C'est **cela** que l'application affiche, jamais la croyance brute.

### 4.5 Choisir la question : le gain d'information

**Fonctions : `entropy`, `information_gain_exact`, `pi_star`.**

L'**entropie** mesure l'incertitude en bits. Croyance uniforme sur 50 états :
entropie maximale. Croyance concentrée sur un seul état : entropie nulle. Le
quiz est donc une machine à faire baisser l'entropie.

Le **gain d'information** d'une question, c'est la réduction d'entropie qu'on
peut *espérer* en la posant. On ne connaît pas la réponse à l'avance, alors on
fait la moyenne sur les deux réponses possibles, pondérée par leur
probabilité :

> gain = (incertitude actuelle) − (incertitude moyenne attendue après réponse)

La politique **`pi_star`** (notée π\* dans la théorie) pose simplement la
question de gain maximal parmi celles non encore posées.

**Un point qui mérite d'être souligné dans le rapport** : la littérature
présente cette politique optimale comme *intraitable* en général, parce qu'elle
demanderait d'explorer tous les futurs possibles. Ici elle est calculée
**exactement**, sans approximation. La raison est simple et vaut la peine
d'être dite : les réponses sont binaires (juste/faux), donc il n'existe que
**deux** avenirs possibles par question, pas une infinité. Le coût est
proportionnel au nombre d'états — c'est tout.

### 4.6 Quand s'arrêter : `should_stop`

Trois critères, le premier qui se déclenche gagne :

1. **Budget épuisé** — 30 questions maximum par défaut.
2. **Confiance atteinte** — un état concentre au moins 85 % de la croyance.
3. **Plus rien à apprendre** — la meilleure question restante rapporterait
   moins de 0,01 bit. Continuer serait faire perdre du temps à l'élève pour
   rien.

Le troisième critère est le plus intéressant scientifiquement : c'est le
programme qui reconnaît lui-même avoir atteint la limite de ce que sa banque
de questions permet de distinguer.

### 4.7 Deux formules fermées : prédire sans simuler

**Fonctions : `item_information`, `questions_needed`.**

Celles-ci répondent à une question naturelle : *combien de questions faut-il, à
peu près ?* — sans lancer la moindre simulation.

`item_information` mesure, en une seule formule, l'information moyenne
qu'apporte **une** question caractérisée par son (slip, guess). Elle vaut zéro
quand slip + guess = 1 : c'est bien la même contrainte qu'au §2, vue sous un
autre angle. **Une question dont le taux de réussite est le même que l'on
maîtrise ou non le concept n'apprend rien** — c'est évident dit comme ça, et
la formule le retrouve.

`questions_needed` en déduit un ordre de grandeur du nombre de questions
nécessaires. **Vérification** : sur notre domaine, la formule prédit **19,2
questions**, la simulation complète en mesure **18,6**. Un accord à 3 % entre
un calcul de trois lignes et un benchmark de plusieurs minutes — c'est le
genre de recoupement qui donne confiance dans les deux.

### 4.8 La géométrie : mesurer le chemin parcouru

**Fonctions : `fisher_rao_distance`, `cumulative_arc_length`,
`expected_fisher_rao_step`.**

L'ensemble des croyances possibles forme un espace courbe, et il existe une
notion naturelle de **distance** entre deux croyances : la distance de
**Fisher-Rao**. Intuitivement : de combien la croyance a-t-elle réellement
bougé ?

Cela permet de mesurer une trajectoire de quiz comme on mesurerait un trajet
sur une carte, et de comparer : le quiz adaptatif parcourt-il plus de chemin
par question que le quiz aléatoire ? (Réponse mesurée : oui — voir §7.)

Détail de mise en œuvre qui illustre l'état d'esprit du projet : la formule
passe par un arccosinus, qui n'est pas défini au-delà de 1. Quand deux
croyances sont presque identiques, l'arrondi machine peut produire
1,0000000001 et faire échouer le calcul. Le code borne explicitement la
valeur, avec en commentaire la justification que c'est exact et non une
rustine.

### 4.9 Le simulateur

**Fonction : `simulate`.**

On lui donne un état de connaissance « vrai », elle fait passer tout le quiz à
un élève fictif qui répond selon le modèle BLIM, et elle rend : le nombre de
questions posées, le diagnostic final, la confiance atteinte, et l'historique
complet de la croyance.

C'est l'outil de mesure de tout le projet — il n'y a pas eu d'expérimentation
sur des élèves réels ; les données réelles servent à la calibration (§5), pas à
l'évaluation des politiques.

**Un raffinement essentiel**, introduit pour tester la robustesse : on peut
donner au simulateur des paramètres (slip, guess) **différents** de ceux que le
moteur utilise pour raisonner. Autrement dit : *l'élève se comporte d'une
certaine façon, le programme en croit une autre.* Sans cette séparation, on ne
peut mesurer que des situations où le modèle est parfaitement juste —
c'est-à-dire jamais la réalité. C'est ce qui rend tout le Lot 3 possible.

---

## 5. Les deux terrains : données réelles et banque écrite

Le projet travaille sur **deux domaines distincts**, et les confondre serait la
principale source de malentendu à la lecture.

### Piste A — le domaine issu de données réelles (`data/`)

Construit à partir de **Junyi Academy**, une plateforme éducative taïwanaise
dont les journaux sont publics : environ **26 millions de réponses** d'élèves
entre 2012 et 2015.

La chaîne de traitement, en quatre étapes :

1. **`extract_junyi.py`** — lit le catalogue d'exercices, regroupe les
   exercices en 5 concepts, et **déduit les prérequis des données** par un test
   statistique (test binomial) : si presque personne ne réussit B sans réussir
   A, on déclare A prérequis de B.
2. **`extract_responses.py`** — transforme les 2,6 Go de journaux bruts en un
   fichier exploitable. Lecture **par morceaux** : le fichier ne tient pas en
   mémoire.
3. **`calibrate_em.py`** — **estime slip et guess à partir des données**, par
   l'algorithme EM (*Expectation-Maximization*). On ne les choisit pas : on
   cherche les valeurs qui rendent les réponses observées les plus
   vraisemblables.
4. Le résultat est écrit dans `data/domain.yaml`.

**Et c'est là qu'un problème apparaît**, qui est devenu une partie importante
du travail : les valeurs de `guess` estimées valent **0,50 à 0,75** selon le
concept. Or `guess` est censé être la probabilité de réussir **sans**
maîtriser. 75 % par pure chance, ce n'est pas crédible.

Trois hypothèses ont été formulées et **testées, chacune écartée par une
mesure** :

- **La pratique répétée** (un élève retente jusqu'à réussir, ce qui gonfle
  artificiellement les taux) : le mécanisme existe mais ne suffit pas — un
  échantillon aléatoire témoin de même taille reproduit le même effet. Le
  problème est d'**identifiabilité statistique**, pas de contamination.
- **Des concepts trop larges** (« arithmétique » mélange des choses trop
  différentes) : on a subdivisé et recalibré. `guess` n'a **pas** baissé
  (0,759 pour le sous-concept isolé, contre 0,746 pour le concept large). Le
  résultat isolé retombe presque exactement sur le résultat d'origine.
- **L'hypothèse d'un état figé** (le modèle suppose que l'élève ne progresse
  pas pendant l'observation, alors que les données couvrent trois ans) : si des
  élèves apprenaient en cours de route, leur intermédiarité serait **ordonnée
  dans le temps** — échecs au début, réussites à la fin. Test direct sur 41 246
  couples (élève, concept) : le taux de réussite ne monte pas entre la première
  et la seconde moitié de leur période d'activité (variation moyenne
  **−0,020**, seulement 47 % en hausse), et le résultat ne survit à aucun degré
  au contrôle statistique.

**Conclusion honnête, et c'est un résultat en soi** : la cause du `guess` élevé
reste inconnue. Mais ce n'est pas une piste laissée de côté faute de temps —
c'est une énigme cernée par trois tests directs, dont les protocoles avaient
été fixés **avant** de regarder les résultats. C'est scientifiquement plus
solide qu'une limite simplement mentionnée.

### Piste B — la banque écrite à la main (`domains/piste_b.yaml`)

7 concepts, 4 prérequis, **35 questions à choix multiples rédigées avec leur
énoncé, leurs options et leur bonne réponse**. C'est le domaine de la démo
jouable et de la majorité des expériences.

Ici, slip et guess ne viennent pas de données. **Mais ce ne sont pas des
valeurs choisies arbitrairement**, et la distinction est importante :

- guess = 0,25 est une **borne combinatoire** : un QCM à 4 options ne peut pas
  donner plus de 1 chance sur 4 à qui répond au hasard. C'est une valeur
  **dérivée**, et conservatrice (les mauvaises réponses proposées incarnent des
  erreurs de raisonnement plausibles, donc le vrai guess est probablement plus
  faible).
- slip = 0,10 n'a **aucune** justification de premier principe. C'est assumé
  comme tel : c'est un point de référence sur un axe qu'on **balaye** (Lot 3),
  pas une constante à défendre.

Cette discipline — chaque paramètre est **dérivé**, **mesuré** ou **balayé**,
jamais « choisi » — est appliquée dans tout le code et dans les commentaires.
Elle vaut d'être reprise telle quelle dans le rapport : c'est ce qui distingue
un paramètre justifiable d'un paramètre posé.

---

## 6. Le concurrent : CAT-IRT (`irt_baseline.py`)

Pour montrer qu'une méthode est bonne, il faut la comparer à ce qui se fait
déjà. Le standard des tests adaptatifs (GMAT, GRE, évaluations à grande
échelle) est le **CAT-IRT** : *Computerized Adaptive Testing* fondé sur
l'*Item Response Theory*.

Différence de fond avec notre approche :

| | KST (ce projet) | IRT (le standard) |
|---|---|---|
| Représentation de l'élève | un **ensemble de concepts** maîtrisés | un **nombre unique** θ, le « niveau » |
| Diagnostic produit | concept par concept | une note sur une échelle |
| Choix de la question | celle qui informe le plus sur l'état | celle dont la difficulté colle au niveau estimé |

Le fichier implémente un vrai CAT-IRT : modèle logistique à 3 paramètres,
sélection par **information de Fisher**, estimation du niveau par **EAP**
(espérance a posteriori).

Un choix technique qui mérite mention : l'estimation se fait par EAP et non par
maximum de vraisemblance, parce que cette dernière **diverge vers l'infini**
tant que l'élève a tout juste ou tout faux — ce qui est le cas au début de
n'importe quel test. C'est la pratique standard des vrais systèmes CAT, pas un
contournement.

**Honnêteté sur cette comparaison** : les paramètres des items IRT sont dérivés
de notre banque plutôt que calibrés indépendamment, et le modèle IRT est
**délibérément mal spécifié** par rapport à la vérité utilisée pour simuler les
élèves. La comparaison ne dit donc pas « KST est meilleur qu'IRT dans
l'absolu ». Elle dit : « quand les élèves se comportent selon un modèle à
concepts multiples, un modèle à note unique perd beaucoup ». Le miroir de cette
expérience a d'ailleurs été fait — et il donne le résultat inverse (§7, ligne
« arène miroir »), ce qui est la meilleure preuve que ni l'un ni l'autre
résultat n'est truqué.

---

## 7. Ce qu'on a mesuré

Chaque expérience est **un script, une commande, un dossier de résultats**
contenant les données brutes, le résumé, la figure, et un journal indiquant la
date, la version exacte du code et les graines aléatoires utilisées. Tout est
rejouable à l'identique.

### 7.1 Le benchmark principal

Trois politiques, **exactement la même banque de 35 questions**, les mêmes
élèves simulés — seul l'algorithme change :

| Politique | Questions posées (moyenne) | Diagnostic exact | Exactitude par concept |
|---|---|---|---|
| **Adaptative (KST)** | **18,6** | 80 % | 95,7 % |
| Aléatoire | 27,7 | 82 % | 96,6 % |
| CAT-IRT | 30,0 (plafond, jamais d'arrêt spontané) | 4 % | 66,3 % |

**Lecture honnête de ce tableau** : la politique adaptative ne diagnostique pas
*mieux* que le tirage aléatoire — à un point près, les deux font aussi bien.
**Son gain est ailleurs : elle y arrive en 33 % de questions en moins.** C'est
exactement la promesse d'un test adaptatif, et il faut la présenter ainsi
plutôt que de suggérer un gain de précision qui n'existe pas.

Le CAT-IRT, lui, ne parvient jamais à s'arrêter et se trompe presque toujours
sur l'état exact — attendu, puisqu'un nombre unique ne peut pas représenter
« maîtrise A et C mais pas B ».

### 7.2 Toutes les expériences, en une table

| Question posée | Ce qu'on a trouvé |
|---|---|
| **Le calcul approché (Monte-Carlo) vaut-il le coup ?** | Non, à notre échelle. En dessous d'environ 1 400 états, le calcul exact est plus rapide **et** juste. Au-delà seulement, une variante spécifique devient intéressante. **Résultat négatif utile** : il ferme une piste d'optimisation qu'on aurait pu croire nécessaire. |
| **Une des deux approximations testées était-elle défaillante ?** | Oui, et sévèrement : à très faible nombre de tirages elle vaut **exactement zéro** pour toutes les questions, ce qui fait arrêter le quiz avant la première question. Défaut de fond découvert par une expérience et non par un test — documenté dans le code à l'endroit exact où il piège. |
| **Le quiz adaptatif parcourt-il plus de « chemin » ?** | Oui, à nombre de questions égal : après 10 questions, la croyance a parcouru une longueur de 6,58 en adaptatif contre 5,52 en aléatoire (+19 %), mesurée en géométrie de Fisher-Rao. Deux mesures indépendantes (information et déplacement géométrique) classent en outre les concepts dans le **même ordre** — recoupement non trivial. |
| **Que se passe-t-il si slip et guess sont faux ?** | Dégradation nette et **asymétrique** (se tromper dans un sens coûte plus que dans l'autre). Surtout : **la confiance affichée devient mensongère** — l'erreur de calibration passe de 0,018 (paramètres justes) à 0,467 (régime des données réelles). C'est la limite la plus importante à énoncer dans le rapport. |
| **Et si le graphe de prérequis est faux ?** | Dégradation **gracieuse**, pas d'effondrement. Face à des élèves dans des états que le modèle juge impossibles, il ne peut par construction jamais tomber juste sur l'état exact (0 %) — mais il se trompe en moyenne sur **1,5 concept sur 7** seulement, contre 0,3 sur les élèves représentables. Le modèle faux donne le concept le plus proche, pas n'importe quoi. |
| **Un moteur « prudent » corrige-t-il la mauvaise calibration ?** | **Non — c'est un compromis, pas un correctif gratuit.** Il améliore le pire cas (erreur de calibration 0,467 → 0,271) mais dégrade le cas normal (0,018 → 0,195) et pose plus de questions. Résultat contraire à l'intuition de départ, rapporté tel quel. |
| **Arène miroir : et si les élèves suivaient le modèle du concurrent ?** | L'IRT gagne (85,0 % contre 73,1 %). **Chaque modèle domine sur sa propre vérité générative.** La vraie question n'est donc pas « quel algorithme est meilleur » mais « lequel des deux décrit le mieux de vrais élèves » — ce qui renvoie aux données réelles. |
| **Le benchmark tient-il sur le domaine réel calibré ?** | La hiérarchie tient (74,8 questions contre 114,8 et 150) mais tout se dégrade fortement : 74,8 questions au lieu de 18,6. **Conséquence directe du `guess` dégénéré** — ce qui relie le problème de calibration à un coût mesurable, et pas seulement théorique. |
| **La cause du `guess` élevé ?** | Trois hypothèses testées, trois écartées (§5). Ouvert, mais cerné. |

### 7.3 Ce qui rend ces résultats défendables

Plusieurs des résultats ci-dessus **contredisent l'intuition de départ** : le
Monte-Carlo inutile, le moteur prudent qui n'est pas un correctif, la
subdivision qui ne réduit pas `guess`, l'IRT qui gagne sur son propre terrain.
Ils sont rapportés tels quels.

Pour les tests statistiques, les seuils de décision ont été **fixés par écrit
avant** de regarder les résultats. C'est ce qui permet d'écrire « l'hypothèse
est écartée » plutôt que « les données ne montrent rien de clair ».

---

## 8. Comment on sait que ça marche

C'est la partie invisible dans un rapport, et pourtant c'est ce qui donne du
poids aux chiffres.

**179 tests automatiques**, exécutés en 10 secondes, qui vérifient chaque
brique séparément :

| Fichier de tests | Nombre | Ce qu'il vérifie |
|---|---|---|
| `test_kst_engine.py` | 90 | Le moteur : catalogue Z, Bayes, gain d'information, arrêt, géométrie |
| `data/test_extract_junyi.py` | 18 | La construction du domaine réel |
| `test_d2_1_signature_temporelle.py` | 17 | Le test de l'hypothèse temporelle |
| `test_irt_baseline.py` | 11 | La baseline CAT-IRT |
| `data/diagnostics/` (4 fichiers) | 20 | Les diagnostics sur le `guess` |
| `data/test_calibrate_em.py` | 10 | L'algorithme EM |
| `domains/test_loader.py` | 8 | Le chargement des domaines |
| `data/test_extract_responses.py` | 5 | L'extraction des réponses |

Trois principes appliqués systématiquement :

1. **Les tests utilisent des données fabriquées, jamais les vraies.** Un test
   doit vérifier la correction du calcul, pas dépendre d'un fichier de 2,6 Go.
2. **Tests de récupération de paramètres.** Le plus convaincant : on fabrique
   des données à partir de valeurs **connues**, on lance l'estimation, et on
   vérifie qu'elle retrouve ces valeurs. Si l'algorithme est faux, il échoue.
3. **Tout le hasard est reproductible.** Chaque simulation part d'une graine
   fixée et enregistrée : relancer une expérience dans deux ans doit redonner
   exactement les mêmes chiffres.

Il reste **un test en échec attendu** : il documente une limite connue plutôt
que de la cacher. Un test qu'on désactive silencieusement est une dette ; un
test marqué comme échec attendu est une note de bas de page exécutable.

---

## 9. La démonstration : `app.py`

Une petite application web (Streamlit, ~160 lignes) qui fait passer le quiz
pour de vrai : elle affiche l'énoncé, propose les options, et après chaque
réponse met à jour la probabilité de maîtrise de chaque concept.

Le point important : **elle ne réimplémente rien**. Elle appelle exactement les
mêmes fonctions que celles vérifiées par les 90 tests du moteur. Ce que le
visiteur voit tourner est le moteur du rapport, pas une maquette qui lui
ressemble.

---

## 10. Ce que le code ne fait pas

À énoncer clairement, plutôt que de le laisser découvrir :

- **Aucune expérimentation sur des élèves réels.** Toutes les comparaisons de
  politiques sont faites sur des élèves simulés. Les données réelles servent à
  la calibration, pas à l'évaluation.
- **Le modèle suppose un niveau figé.** L'élève ne progresse pas pendant le
  quiz. Sur la durée d'un quiz c'est raisonnable ; sur trois ans de journaux,
  c'est une approximation (et elle a été testée, §5).
- **La calibration sur données réelles n'est pas concluante.** Les valeurs de
  `guess` obtenues ne sont pas crédibles, et la cause reste inconnue après
  trois tests.
- **Les paramètres de la banque écrite ne sont pas mesurés.** Ils sont dérivés
  ou balayés — jamais présentés comme mesurés.
- **La taille du domaine reste modeste** (7 concepts, 50 états). Le passage à
  l'échelle a été étudié en coût de calcul (§7) mais pas éprouvé sur un vrai
  grand domaine.

Aucun de ces points n'est un défaut caché : chacun est écrit dans le code, à
l'endroit où il s'applique.

---

## 11. Pour lancer les choses

Lancer tous les tests :

```bash
python -m pytest -q
```

Rejouer le benchmark principal :

```bash
python benchmark_a3.py
```

Ouvrir la démo jouable :

```bash
streamlit run app.py
```

Chaque expérience se lance de la même façon (`python <nom_du_script>.py`) et
écrit ses résultats dans son propre dossier sous `results/`. Le fichier
`results/REFERENCE.md` recense **toutes** les expériences avec, pour chacune,
le script, la commande, les graines et les fichiers produits : c'est l'index
qui permet de remonter de n'importe quel chiffre du rapport à ce qui l'a
produit.
