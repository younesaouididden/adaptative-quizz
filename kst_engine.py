"""
Moteur de quiz adaptatif fonde sur la Knowledge Space Theory.

Implemente les couches 1, 2 et 4 du PFA :
  1. Combinatoire : construction de Z a partir d'un graphe de prerequis
  2. Probabilite  : croyance p sur Delta(Z), mise a jour bayesienne (BLIM)
  4. Controle     : selection par maximisation du gain d'information

Aucune dependance hors numpy.

Correspondance theorie (monographie) <-> code (Lot 5, plan_action_code.md) --
a tenir a jour si l'un des deux cote change :

  Theorie                                          Code
  ------------------------------------------------  -----------------------------
  p_t in Delta(Z), etat de croyance                 p : np.ndarray (vecteur sur Z)
  Transition bayesienne, eq. (1)                    bayes_update
  pi*(p) = argmax_a IG(a;p), politique optimale      pi_star
  IG(a;p) = E_y[D_KL(p^y_a || p)]                    information_gain_exact
  epsilon-lissage                                    lissage symetrique en log-espace
                                                      (bayes_update, EPS)
  Action pedagogique a in A                          question de la banque (Question)
  Probabilite d'emission P(y|a,z)                    modele BLIM (slip, guess)

Les deux dernieres lignes ne sont PAS renommees dans le code : "action a" et
"question" designent le meme objet, mais "question" reste plus lisible en
Python qu'une lettre seule -- cf. Question/Concept, deja separes (voir
docstrings des deux classes). Idem pour p_t : le parametre s'appelle deja `p`
partout, ce qui correspond directement a la notation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain, combinations

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# COUCHE 1 : L'espace des connaissances Z (chapitre 2)
# ---------------------------------------------------------------------------

def build_knowledge_space(items: list[str],
                          prereqs: list[tuple[str, str]]) -> list[frozenset]:
    """Construit Z = { z subset P : z clos vers le bas pour la relation de surmise }.

    prereqs est une liste de paires (a, b) signifiant "a est prerequis de b",
    c'est-a-dire a <= b. On calcule d'abord la cloture transitive, puis on
    retient les sous-ensembles clos.

    Le resultat est automatiquement un espace de connaissance (union-closed) :
    l'intersection et l'union de deux ensembles clos sont closes.

    P = les CONCEPTS du domaine (pas les questions) : Z se construit une
    fois pour toutes sur le graphe de prerequis, independamment de la
    taille de la banque de questions.
    """
    # cloture transitive : required[b] = tous les prerequis de b
    required = {i: set() for i in items}
    for a, b in prereqs:
        required[b].add(a)
    changed = True
    while changed:                      # Floyd-Warshall du pauvre
        changed = False
        for b in items:
            for a in list(required[b]):
                new = required[a] - required[b]
                if new:
                    required[b] |= new
                    changed = True

    for i in items:
        if i in required[i]:
            raise ValueError(f"Cycle de prerequis detecte sur l'item '{i}'")

    powerset = chain.from_iterable(combinations(items, r)
                                   for r in range(len(items) + 1))
    Z = [frozenset(s) for s in powerset
         if all(required[x] <= set(s) for x in s)]
    return sorted(Z, key=lambda z: (len(z), sorted(z)))


# ---------------------------------------------------------------------------
# COUCHE 2 : Le modele de reponse (BLIM) et la croyance (chapitre 3)
# ---------------------------------------------------------------------------

@dataclass
class Concept:
    """Noeud du graphe de prerequis, element de P. Z se construit sur les
    concepts, jamais sur les questions."""
    name: str


@dataclass
class Question:
    """Item de la banque de questions, rattache a un concept (chapitre 3).

    slip = erreur d'inattention, guess = reponse au hasard. Plusieurs
    questions peuvent partager le meme concept avec des (slip, guess)
    differents : c'est ce qui donne un vrai choix a la selection gloutonne
    (chapitres 6-7) une fois qu'un concept a deja ete interroge une fois.

    Contrainte de validite du BLIM : slip + guess < 1, sinon repondre juste
    est une preuve de NON-maitrise et l'inference s'inverse.
    """
    id: str
    concept: str
    slip: float = 0.10
    guess: float = 0.20
    enonce: str = ""

    def __post_init__(self):
        if self.slip + self.guess >= 1.0:
            raise ValueError(f"Question '{self.id}': slip + guess doit etre < 1")


@dataclass
class Domain:
    concepts: list[Concept]
    prereqs: list[tuple[str, str]]
    questions: list[Question]
    Z: list[frozenset] = field(init=False)

    def __post_init__(self):
        names = [c.name for c in self.concepts]
        concept_names = set(names)
        for q in self.questions:
            if q.concept not in concept_names:
                raise ValueError(
                    f"Question '{q.id}' rattachee a un concept inconnu '{q.concept}'")
        self.Z = build_knowledge_space(names, self.prereqs)
        # matrice de vraisemblance L[z, q] = P(correct | z, question), chapitre 3.
        # Indexee sur les QUESTIONS, pas les concepts : deux questions du meme
        # concept partagent la meme condition "concept in z" mais ont chacune
        # leur propre (slip, guess) -> colonnes distinctes.
        self.L = np.array([[self._p_correct(z, q) for q in self.questions]
                           for z in self.Z])

    @staticmethod
    def _p_correct(z: frozenset, question: Question) -> float:
        return (1.0 - question.slip) if question.concept in z else question.guess

    @property
    def n_states(self) -> int:
        return len(self.Z)

    @property
    def n_questions(self) -> int:
        return len(self.questions)


def uniform_prior(domain: Domain) -> np.ndarray:
    return np.full(domain.n_states, 1.0 / domain.n_states)


def bayes_update(p: np.ndarray, domain: Domain, q: int, correct: bool) -> np.ndarray:
    """Mise a jour bayesienne p_{t+1}(z) ~ P(y|a,z) p_t(z), en log pour la stabilite.

    q indexe domain.questions (pas domain.concepts).

    Chapitres 4-5 : le bord du simplexe est a distance infinie, un p(z) qui
    atteint exactement 0 y est pieoge de facon irreversible. On lisse donc p
    ET la vraisemblance avant de combiner (jamais un seul des deux), puis on
    renormalise par la somme reelle -- ce qui est algebriquement identique a
    la formule p <- (p+eps)/(1+|Z|*eps) mais plus robuste aux erreurs
    d'arrondi flottant.
    """
    lik = domain.L[:, q] if correct else 1.0 - domain.L[:, q]
    log_post = np.log(p + EPS) + np.log(lik + EPS)
    log_post -= log_post.max()                # stabilisation
    post = np.exp(log_post)
    return post / post.sum()


def concept_marginals(p: np.ndarray, domain: Domain) -> dict[str, float]:
    """P(concept c maitrise) = somme des p(z) sur les z contenant c (chapitre 3).

    C'est CA qu'on affiche a l'utilisateur, jamais p en entier (chapitre 6).
    """
    return {c.name: float(sum(p[k] for k, z in enumerate(domain.Z)
                              if c.name in z))
            for c in domain.concepts}


# ---------------------------------------------------------------------------
# GEOMETRIE DE FISHER-RAO sur Delta(Z) (chapitres 4-5, Lot 2 du plan)
# ---------------------------------------------------------------------------

def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """d(p,q) = 2.arccos( Sum_z sqrt(p(z).q(z)) ), la distance geodesique
    sur Delta(Z) pour la metrique de Fisher-Rao (chapitres 4-5).

    Vient du plongement "carte racine" x = 2.sqrt(p) : x vit alors sur
    l'octant positif d'une sphere de rayon 2 (||x||^2 = 4.Sum p(z) = 4), et
    d(p,q) est exactement 2 fois l'angle entre x_p et x_q -- d'ou la forme
    fermee, qui ne demande ni integrale ni geodesique explicite.

    Sum_z sqrt(p(z).q(z)) (l'affinite de Bhattacharyya) peut legerement
    depasser 1 par erreur d'arrondi flottant quand p~=q, ce qui rendrait
    arccos indefini (NaN) : on clippe a [-1,1], exact partout ailleurs sur
    le domaine (meme raisonnement que le clip de _binary_entropy).

    d(p,q)=0 ssi p=q ; d(p,q)=pi (maximum) ssi p et q ont des supports
    disjoints (affinite nulle).
    """
    affinity = float(np.sum(np.sqrt(p * q)))
    return 2.0 * float(np.arccos(np.clip(affinity, -1.0, 1.0)))


def cumulative_arc_length(belief_trace: list[np.ndarray]) -> list[float]:
    """Longueur d'arc cumulee le long d'une trajectoire de croyances
    (chapitres 4-5, Lot 2.1-2.2) : somme des distances de Fisher-Rao entre
    pas consecutifs. cumulative_arc_length(trace)[0] == 0.0 (avant toute
    question), meme longueur que belief_trace.
    """
    lengths = [0.0]
    for i in range(1, len(belief_trace)):
        lengths.append(lengths[-1] +
                       fisher_rao_distance(belief_trace[i - 1], belief_trace[i]))
    return lengths


def expected_fisher_rao_step(slip: float, guess: float, prior: float = 0.5) -> float:
    """Distance de Fisher-Rao moyenne parcourue sur Delta({non-maitrise,
    maitrise}) apres UNE reponse a une question (slip, guess), a partir
    d'une croyance `prior` sur la maitrise (chapitres 4-5, Lot 2.4).

    Distinct de item_information (Lot 1, note_calibration.md) : celui-ci
    mesure l'information au sens de la divergence KL/Wald, celui-la le
    DEPLACEMENT GEOMETRIQUE reel sur la variete Delta(Z) -- deux angles
    differents sur le meme phenomene (guess eleve => l'item n'apporte
    presque rien), qui doivent tous deux s'effondrer quand guess -> 1-slip,
    ce qui sert de verification croisee entre Lot 1 et Lot 2.
    """
    p = np.array([1.0 - prior, prior])            # [P(non-maitrise), P(maitrise)]
    l_correct = np.array([guess, 1.0 - slip])      # P(correct | non-maitrise/maitrise)
    l_incorrect = 1.0 - l_correct
    p_correct = float(np.dot(p, l_correct))

    def posterior(l: np.ndarray) -> np.ndarray:
        post = p * l
        return post / post.sum()

    d_correct = fisher_rao_distance(p, posterior(l_correct))
    d_incorrect = fisher_rao_distance(p, posterior(l_incorrect))
    return p_correct * d_correct + (1.0 - p_correct) * d_incorrect


# ---------------------------------------------------------------------------
# COUCHE 4 : Gain d'information et selection (chapitres 6-7)
# ---------------------------------------------------------------------------

def entropy(p: np.ndarray) -> float:
    """H(p) en bits.

    On filtre p>EPS plutot que de lisser : lim x->0 (x log x) = 0, donc
    ignorer les etats quasi nuls est exact, pas une approximation biaisee.
    """
    q = p[p > EPS]
    return float(-np.sum(q * np.log2(q)))


def information_gain_exact(p: np.ndarray, domain: Domain, q: int) -> float:
    """IG(a;p) = H(p) - E_y[ H(posterior) ].

    Identique a E_y[ KL(posterior || p) ] et a l'information mutuelle I(Z;Y|a),
    mais la version entropie evite tout calcul explicite de KL -- donc evite
    le piege de smoothing asymetrique du point suivant. C'est pour ca qu'on
    l'implemente ainsi plutot que la formule KL directement (chapitre 6).

    q indexe domain.questions. Cout O(|Z|) : avec |Y|=2 la somme exacte
    suffit, pas besoin de Monte Carlo.
    """
    lik = domain.L[:, q]
    p_correct = float(np.dot(p, lik))       # predictive P(y=1 | a, p)
    ig = entropy(p)
    for prob_y, l in ((p_correct, lik), (1.0 - p_correct, 1.0 - lik)):
        if prob_y <= EPS:
            continue
        post = p * l
        post = post / post.sum()
        ig -= prob_y * entropy(post)
    return ig


def _binary_entropy(x: np.ndarray | float) -> np.ndarray | float:
    """H_2(x) = -x.log2(x) - (1-x).log2(1-x), l'entropie de Bernoulli(x).

    Clippe a EPS des bords : lim x->0,1 de x.log2(x) est 0, pas indefini,
    donc clipper est exact a EPS pres, pas une approximation biaisee (meme
    raisonnement que le filtre p>EPS de entropy())."""
    x = np.clip(x, EPS, 1.0 - EPS)
    return -(x * np.log2(x) + (1.0 - x) * np.log2(1.0 - x))


def information_gain_mc(p: np.ndarray, domain: Domain, q: int,
                        n_samples: int = 200,
                        mode: str = "sample_y",
                        rng: np.random.Generator | None = None) -> float:
    """Estimateur Monte Carlo de IG(a;p) (chapitres 4-7, Lot 1 du plan
    d'action -- cf. plan_action_code.md pour le preambule methodologique
    complet sur les deux modes).

    mode="sample_y" (estimateur original, diapo 7 de la presentation d'aout)
    Echantillonne les REPONSES y ~ P(.|a,p), via E_y[ KL(posterior || p) ].
    Pour un item binaire |Y|=2, il n'existe que DEUX posteriors possibles
    quel que soit n_samples : on tire un seul compte binomial et on pondere
    les deux posteriors (deja calcules une fois chacun) au lieu de boucler
    bayes_update n_samples fois -- optimisation pure, meme quantite estimee.
    Cout final O(|Z|), independant de n_samples -- **exactement le meme
    ordre que information_gain_exact**, pour une valeur seulement APPROCHEE.
    C'est la demonstration silencieuse de l'argument du preambule du Lot 1 :
    echantillonner y pour un item binaire est strictement pire que calculer
    l'exact (information_gain_exact le fait deja, au meme cout, sans bruit).
    Garde pour comparaison empirique (E1-E3) et parce que l'approche devient
    necessaire des que les reponses sont a choix multiples (|Y|>2) ou que
    l'action est un bloc de plusieurs questions (|Y|=2^k) -- non implemente
    ici, cf. Lot 1. N'attaque PAS le goulot reel, qui est |Z|.

    mode="sample_z" (attaque le goulot |Z|)
    Echantillonne les ETATS z_i ~ p(z) (PAS les reponses), et utilise la
    decomposition duale de l'information mutuelle
        I(Z;Y|a) = H(Y|a) - E_z[ H(Y|a,z) ]
    au lieu de H(Y|a) - E_y[ H(Z|a,y) ]. H(Y|a,z) est l'entropie binaire de
    L[z,q] (P(correct|z,q), une lecture directe -- deterministe une fois z
    fixe, aucun bayes_update). H(Y|a) est estimee par le meme echantillon
    (plug-in sur la moyenne empirique de L[z_i,q]), pas calculee exactement
    sur Z : cout O(n_samples), INDEPENDANT de |Z|. C'est cette variante qui
    repond a la question du Lot 1 -- "en dessous de quel |Z| calculer
    l'exact, au-dela utiliser MC avec quel N" -- puisque c'est la seule a ne
    jamais parcourir Z en entier. Biais de plug-in sur le terme H(Y|a)
    (fonction non-lineaire de la moyenne empirique) qui s'attenue avec N :
    exactement le compromis biais-variance-temps que E1 doit chiffrer.

    PIEGE decouvert en E2 (pas en E1) : a n_samples=1, l'estimateur est
    DEGENERE, pas juste bruite. p_correct_hat (un seul point) est alors
    IDENTIQUE a l'unique element de p_correct_i utilise pour le terme
    conditionnel -- H(Y|a) et E_z[H(Y|a,z)] sont donc calcules sur exactement
    la meme donnee, et leur difference vaut 0.0 EXACTEMENT, pour toute
    question, a chaque appel (pas juste en esperance). pi_hat(mode="sample_z",
    n_samples=1) retombe alors systematiquement sur le premier candidat
    balaye (argmax sur des ex-aequo a 0), et should_stop s'arrete
    immediatement (ig=0 < min_ig) -- simulate_mc(n_samples=1, mode="sample_z")
    ne pose donc JAMAIS aucune question. E1 (qui mesure une decision isolee
    au milieu d'une trajectoire deja avancee) ne revele pas cette pathologie
    aussi clairement que E2 (qui rejoue la boucle complete depuis le prior) --
    exactement pourquoi les deux experiences sont necessaires. N=1 est a
    proscrire avec ce mode ; N>=3 suffit a le rendre non-degenere.

    Piege commun aux deux modes (chapitre 4-5) : lisser SEULEMENT le
    denominateur du KL, log(P / (Q+eps)), biaise l'estimateur et peut le
    rendre negatif. On lisse donc les deux distributions avec la meme
    formule p <- (p+eps)/(1+|Z|*eps) avant de calculer le KL (mode sample_y
    uniquement -- sample_z n'a pas de KL sur Delta(Z), donc pas ce piege).
    """
    rng = rng or np.random.default_rng()

    if mode == "sample_z":
        idx = rng.choice(len(p), size=n_samples, p=p)
        p_correct_i = domain.L[idx, q]                    # P(correct|z_i,q), lecture directe
        p_correct_hat = float(p_correct_i.mean())
        h_y = float(_binary_entropy(p_correct_hat))
        h_y_given_z = float(np.mean(_binary_entropy(p_correct_i)))
        return h_y - h_y_given_z

    if mode != "sample_y":
        raise ValueError(f"information_gain_mc: mode={mode!r} inconnu "
                        "(attendu 'sample_y' ou 'sample_z')")

    p_correct = float(np.dot(p, domain.L[:, q]))
    n_z = len(p)

    def smooth(dist: np.ndarray) -> np.ndarray:
        return (dist + EPS) / (1.0 + n_z * EPS)

    p_s = smooth(p)
    # y est binaire (|Y|=2) : il n'existe que DEUX posteriors possibles,
    # quel que soit n_samples. Plutot que boucler n_samples fois sur
    # bayes_update (identique a chaque tirage correct=True, ou a chaque
    # tirage correct=False), on tire un seul compte binomial et on
    # pondere les deux posteriors deja calcules -- resultat identique a
    # la boucle naive (meme formule, juste factorisee), mais O(1) appels a
    # bayes_update au lieu de O(n_samples). Optimisation pure, aucun
    # changement de la quantite estimee (les tests de convergence
    # utilisent des tolerances, pas des valeurs figees).
    n_correct = int(rng.binomial(n_samples, p_correct))
    total = 0.0
    for correct, count in ((True, n_correct), (False, n_samples - n_correct)):
        if count == 0:
            continue
        post = bayes_update(p, domain, q, correct)
        post_s = smooth(post)
        total += count * float(np.sum(post_s * np.log2(post_s / p_s)))
    return total / n_samples


def item_information(slip: float, guess: float) -> float:
    """Information moyenne (en nats) qu'une question binaire apporte sur la
    maitrise du concept qu'elle teste -- complement ferme aux chapitres 6-7,
    independant de p et de Z (contrairement a information_gain_exact, qui
    est le gain exact pour UNE question dans UN etat de croyance donne).

    C'est la J-divergence (KL symetrisee) entre les deux lois d'emission du
    BLIM -- Bern(1-slip) si le concept est maitrise, Bern(guess) sinon --,
    l'argument de Wald/SPRT sur la derive moyenne du rapport de
    vraisemblance log-log. Sert a estimer le nombre de questions requises
    SANS lancer de simulation (cf. questions_needed, et note_calibration.md
    pour la derivation complete et sa verification empirique contre le
    benchmark A3).

    Degenere vers 0 quand slip+guess -> 1 : la contrainte BLIM slip+guess<1
    (Question.__post_init__) est exactement la condition item_information>0,
    "l'item apporte de l'information".
    """
    p1, p0 = 1.0 - slip, guess
    if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0):
        raise ValueError(
            f"item_information(slip={slip}, guess={guess}) : 1-slip et guess "
            "doivent etre strictement entre 0 et 1")

    def kl(p: float, q: float) -> float:
        return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

    return 0.5 * (kl(p1, p0) + kl(p0, p1))


def questions_needed(slip: float, guess: float, n_concepts: int,
                     target: float = 0.95) -> float:
    """Nombre approximatif de questions pour amener la confiance de 0.5 a
    `target` sur chaque concept, sous l'approximation de Wald (le
    log-rapport de vraisemblance derive en moyenne de item_information nats
    par question) et l'hypothese de n_concepts concepts independants
    interroges un par un.

    Approximation, pas une prediction exacte : ignore le partage
    d'information entre concepts que permet la structure de prerequis (la
    politique adaptative en tire parti), et le depassement (overshoot) au
    franchissement du seuil de decision. Verifiee empiriquement contre le
    benchmark A3 sur piste B (7 concepts, slip=0.10, guess=0.25) : predit
    19.2 questions, benchmark_a3.py en mesure 18.6 -- accord a 3%, cf.
    note_calibration.md pour les reserves.
    """
    log_odds = np.log(target / (1.0 - target))
    return n_concepts * log_odds / item_information(slip, guess)


def pi_star(p: np.ndarray, domain: Domain,
           asked: set[int]) -> tuple[int, float]:
    """Politique optimale exacte π*(p) = argmax_a IG(a;p) (chapitres 6-7) :
    argmax du gain d'information sur les questions non posees.

    C'est la politique que la theorie declare intractable en general et que
    cette implementation calcule EXACTEMENT (pas une approximation) --
    d'ou la valeur du Lot 1 du plan (validation de l'estimateur Monte Carlo
    contre cette meme reference exacte, cf. plan_action_code.md).

    asked contient des indices de QUESTIONS (pas de concepts) : plusieurs
    questions du meme concept restent eligibles tant qu'elles n'ont pas
    elles-memes ete posees. C'est le coeur du refactor concept/question --
    sans banque de plusieurs questions par concept, argmax n'a rien a
    departager une fois chaque concept interroge une fois.
    """
    best, best_ig = None, -np.inf
    for q in range(domain.n_questions):
        if q in asked:
            continue
        ig = information_gain_exact(p, domain, q)
        if ig > best_ig:
            best, best_ig = q, ig
    return best, best_ig


def pi_hat(p: np.ndarray, domain: Domain, asked: set[int],
          n_samples: int = 30, mode: str = "sample_z",
          rng: np.random.Generator | None = None) -> tuple[int, float]:
    """Politique approximee π̂_N(p) = argmax_a IG_MC(a;p) (Lot 1,
    plan_action_code.md) : meme structure gloutonne que pi_star, mais le
    gain d'information de chaque candidat est estime par
    information_gain_mc (mode, n_samples) au lieu d'etre calcule
    exactement -- c'est la politique que E1/E2 comparent a pi_star.

    Un seul rng est partage entre tous les candidats d'un meme appel (pas
    un rng par candidat) : c'est ce qui rend la comparaison entre candidats
    a l'interieur d'un meme appel coherente d'un tirage a l'autre.

    Retourne (indice de la question choisie, son gain d'information ESTIME
    -- pas le gain exact de cette question : comparer information_gain_exact
    de la question retournee a celle de pi_star donne le regret d'E1, pas
    cette valeur-ci).
    """
    rng = rng or np.random.default_rng()
    best, best_ig = None, -np.inf
    for q in range(domain.n_questions):
        if q in asked:
            continue
        ig = information_gain_mc(p, domain, q, n_samples=n_samples, mode=mode, rng=rng)
        if ig > best_ig:
            best, best_ig = q, ig
    return best, best_ig


def should_stop(p: np.ndarray, domain: Domain, asked: set[int],
                max_questions: int = 30,
                confidence: float = 0.85,
                min_ig: float = 0.01,
                ig: float | None = None) -> bool:
    """Trois criteres d'arret (chapitre 7), le premier qui se declenche gagne.

    ig : gain d'information de la meilleure question restante, si deja
    calcule par pi_star (c'est le cas dans simulate). Sinon il est
    recalcule ici -- pratique pour appeler should_stop seul (tests), mais
    coute O(|A|.|Z|) : ne pas l'omettre dans une boucle chaude.
    """
    if len(asked) >= min(max_questions, domain.n_questions):
        return True
    if p.max() >= confidence:                       # un etat domine
        return True
    if ig is None:
        _, ig = pi_star(p, domain, asked)             # plus rien a apprendre
    return ig < min_ig


# ---------------------------------------------------------------------------
# SIMULATION : etudiant synthetique, adaptatif vs aleatoire
# ---------------------------------------------------------------------------

def simulate(domain: Domain, z_true: frozenset, adaptive: bool = True,
             seed: int = 0, max_questions: int = 30,
             verite_slip: float | None = None,
             verite_guess: float | None = None):
    """Fait passer le quiz a un etudiant simule dont on connait l'etat reel.

    verite_slip / verite_guess (Lot 3.1, plan_action_code.md -- decouplage
    verite/moteur) : si fournis, la reponse de l'etudiant est generee avec
    CES parametres au lieu de question.slip/question.guess, alors que la
    mise a jour bayesienne continue d'utiliser domain.L (ce que le moteur
    CROIT, fige a la construction du Domain). None (defaut) : comportement
    inchange d'avant ce lot, verite et moteur coincident.

    C'est le seul changement de signature demande par le Lot 3 -- pas une
    reecriture du simulateur, cf. plan_action_code.md 3.1.
    """
    rng = np.random.default_rng(seed)
    p = uniform_prior(domain)
    asked: set[int] = set()
    trace = [entropy(p)]
    belief_trace = [p.copy()]   # sequence complete des croyances (Lot 2, geometrie)

    while True:
        # pi_star calcule aussi le critere d'arret 3 (chapitre 7) : on le
        # passe a should_stop plutot que le laisser le recalculer, ce qui
        # evite de payer deux fois O(|A|.|Z|) par question posee.
        q_best, ig = pi_star(p, domain, asked)
        if should_stop(p, domain, asked, max_questions=max_questions, ig=ig):
            break
        q = q_best if adaptive else int(rng.choice(
            [i for i in range(domain.n_questions) if i not in asked]))
        question = domain.questions[q]
        # l'etudiant repond selon la VERITE (peut differer de ce que le
        # moteur croit -- domain.L, utilise par bayes_update ci-dessous)
        slip = question.slip if verite_slip is None else verite_slip
        guess = question.guess if verite_guess is None else verite_guess
        true_p = (1 - slip) if question.concept in z_true else guess
        correct = bool(rng.random() < true_p)
        p = bayes_update(p, domain, q, correct)   # <- toujours domain.L (le moteur)
        asked.add(q)
        trace.append(entropy(p))
        belief_trace.append(p.copy())

    z_hat = domain.Z[int(np.argmax(p))]
    return {"n_questions": len(asked), "entropy_trace": trace,
            "belief_trace": belief_trace,
            "z_hat": z_hat, "correct_diagnosis": z_hat == z_true,
            "confidence": float(p.max()), "belief": p}


def simulate_mc(domain: Domain, z_true: frozenset, n_samples: int,
                mode: str = "sample_z", seed: int = 0,
                max_questions: int = 30) -> dict:
    """Variante de simulate() pilotee par la politique APPROXIMEE pi_hat
    (Lot 1, plan_action_code.md, experience E2) au lieu de pi_star -- meme
    structure, meme sortie, seule la ligne de selection change.

    Selection ET critere d'arret (3e critere de should_stop, gain d'info
    residuel) utilisent tous deux l'estimation MC, pas l'exact : c'est
    l'usage realiste de l'approximation en production (on ne paierait pas
    le cout de pi_star juste pour le critere d'arret si le but est
    justement d'eviter ce cout).

    Mesure le cout en aval de l'approximation : comparer n_questions et
    correct_diagnosis a simulate(domain, z_true, adaptive=True, seed=seed)
    (= pi_star, N infini) pour le meme z_true et la meme graine.
    """
    rng = np.random.default_rng(seed)
    p = uniform_prior(domain)
    asked: set[int] = set()
    trace = [entropy(p)]

    while True:
        q_best, ig = pi_hat(p, domain, asked, n_samples=n_samples, mode=mode, rng=rng)
        if should_stop(p, domain, asked, max_questions=max_questions, ig=ig):
            break
        question = domain.questions[q_best]
        true_p = (1 - question.slip) if question.concept in z_true else question.guess
        correct = bool(rng.random() < true_p)
        p = bayes_update(p, domain, q_best, correct)
        asked.add(q_best)
        trace.append(entropy(p))

    z_hat = domain.Z[int(np.argmax(p))]
    return {"n_questions": len(asked), "entropy_trace": trace,
            "z_hat": z_hat, "correct_diagnosis": z_hat == z_true,
            "confidence": float(p.max()), "belief": p}


# ---------------------------------------------------------------------------
# DOMAINE JOUET -- a remplacer par le tien
# ---------------------------------------------------------------------------

_CONCEPT_NAMES = ["fractions", "equations1", "fonctions", "derivees",
                  "limites", "integrales", "proba_base", "var_aleatoire"]

_PREREQS = [
    ("fractions", "equations1"),
    ("equations1", "fonctions"),
    ("fonctions", "limites"),
    ("limites", "derivees"),
    ("derivees", "integrales"),
    ("fractions", "proba_base"),
    ("proba_base", "var_aleatoire"),
    ("fonctions", "var_aleatoire"),
]

_BASE_SLIP_GUESS = {
    "fractions":     (0.05, 0.25),
    "equations1":    (0.10, 0.20),
    "fonctions":     (0.10, 0.20),
    "derivees":      (0.12, 0.20),
    "limites":       (0.12, 0.25),
    "integrales":    (0.15, 0.15),
    "proba_base":    (0.08, 0.25),
    "var_aleatoire": (0.15, 0.20),
}


_OFFSET_POOL = [0.0, 0.02, -0.02, 0.04, -0.04, 0.03, -0.03, 0.01, -0.01, 0.015]


def _question_bank(concept: str, base_slip: float, base_guess: float,
                   n: int) -> list[Question]:
    """n questions synthetiques pour un concept, avec un slip/guess legerement
    varie autour de la valeur de base -- purement pour le domaine jouet.

    Le vrai domaine chargera sa banque depuis un fichier YAML/JSON, pas
    depuis du code en dur (voir PROMPT_DEMARRAGE section 8).
    """
    if n > len(_OFFSET_POOL):
        raise ValueError(
            f"_question_bank({concept!r}, n={n}) : seulement "
            f"{len(_OFFSET_POOL)} offsets predefinis pour le domaine jouet. "
            "Augmenter _OFFSET_POOL, ou passer par un domaine reel (YAML) "
            "pour une banque plus large.")
    offsets = _OFFSET_POOL[:n]
    return [Question(id=f"{concept}_q{i+1}", concept=concept,
                     slip=round(min(0.30, max(0.01, base_slip + off)), 3),
                     guess=round(min(0.35, max(0.05, base_guess - off)), 3))
           for i, off in enumerate(offsets)]


def make_demo_domain(questions_per_concept: int = 3) -> Domain:
    """Domaine jouet parametre par la taille de la banque de questions --
    sert a comparer le comportement avant/apres le refactor
    concept/question (1 question/concept vs plusieurs)."""
    return Domain(
        concepts=[Concept(n) for n in _CONCEPT_NAMES],
        prereqs=_PREREQS,
        questions=[q for name in _CONCEPT_NAMES
                  for q in _question_bank(name, *_BASE_SLIP_GUESS[name],
                                          n=questions_per_concept)],
    )


DEMO = make_demo_domain(questions_per_concept=3)


if __name__ == "__main__":
    d = DEMO
    n_concepts = len(d.concepts)
    print(f"Concepts : {n_concepts}   |   2^n = {2**n_concepts}"
          f"   |   |Z| = {d.n_states}   "
          f"({100*d.n_states/2**n_concepts:.0f}% du power set)")
    print(f"Questions : {d.n_questions}   (banque de "
          f"{d.n_questions // n_concepts} par concept)")
    print(f"Entropie initiale H(p0) = {entropy(uniform_prior(d)):.2f} bits\n")

    # verification de l'exemple numerique de la presentation 3
    print("--- Test : reproduction de l'exemple du collegue ---")
    toy = Domain(concepts=[Concept("integrales")], prereqs=[],
                questions=[Question("integrales_q1", "integrales",
                                    slip=0.10, guess=0.20)])
    p_toy = uniform_prior(toy)
    p_toy = bayes_update(p_toy, toy, 0, correct=True)
    print(f"prior 0.500 -> posterior {concept_marginals(p_toy, toy)['integrales']:.3f}"
          f"   (attendu 0.818)\n")

    # coherence Monte Carlo vs exact (avec le smoothing KL corrige)
    print("--- Test : Monte Carlo converge vers l'exact ---")
    p0 = uniform_prior(d)
    for q in (0, 4, d.n_questions - 1):
        ex = information_gain_exact(p0, d, q)
        mc = information_gain_mc(p0, d, q, n_samples=5000,
                                 rng=np.random.default_rng(1))
        print(f"  {d.questions[q].id:16s}  exact={ex:.4f}  MC={mc:.4f}")

    # benchmark AVANT/APRES le refactor : 1 question/concept vs plusieurs
    print("\n--- Benchmark : avant (1Q/concept) vs apres (3Q/concept) le refactor ---")
    for n_q_per_concept in (1, 3):
        dom = make_demo_domain(questions_per_concept=n_q_per_concept)
        print(f"\n  banque = {n_q_per_concept} question(s)/concept "
              f"(|A| = {dom.n_questions}, |Z| = {dom.n_states})")
        for label, adaptive in (("adaptatif", True), ("aleatoire", False)):
            n_q, acc = [], []
            for seed, z in enumerate(dom.Z):
                r = simulate(dom, z, adaptive=adaptive, seed=seed)
                n_q.append(r["n_questions"])
                acc.append(r["correct_diagnosis"])
            print(f"    {label:10s} : {np.mean(n_q):.2f} questions en moyenne, "
                  f"diagnostic exact {100*np.mean(acc):.0f}%")
