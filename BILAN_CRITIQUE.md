# Bilan critique — où en est le PFA, ce qui tient, ce qui ne tient pas

> État au 4 septembre 2026. Document d'auto-évaluation, écrit pour préparer la rédaction du mémoire et anticiper les objections d'un jury. Volontairement sévère : les forces sont listées sobrement, les faiblesses en détail. Complète `RAPPORT_AVANCEMENT.md` (qui décrit ce qui a été fait) en disant ce que ça vaut.

---

## 1. Le projet est-il fini ?

**Non.** Il faut distinguer trois axes qui n'en sont pas au même point :

| Axe | État |
|---|---|
| **Moteur / code** | **Fini.** Stable, 161 tests, aucun chantier ouvert. |
| **Expériences / résultats numériques** | **Finis.** Le plan d'action (`plan_action_code.md`, Lots 0 à 4) est intégralement exécuté. 16 figures, tous les résultats tracés et reproductibles. |
| **Rapport scientifique** | **Pas commencé.** Zéro page rédigée. C'est pourtant *le* livrable du stage. |

Autrement dit : **on a fini de produire la matière, on n'a pas commencé à écrire.** Le risque principal du projet n'est plus technique, il est rédactionnel.

Deux points d'intendance également en suspens : l'app Streamlit déployée tourne encore sur l'ancien domaine (piste A) et n'a pas été redéployée depuis la bascule vers piste B ; et l'accès du collègue n'a pas été ouvert.

---

## 2. Ce qui marche

- **Le moteur est correct et testé.** 4 couches KST implémentées, cas d'or de la monographie reproduit (0,500 → 0,818), 161 tests dont des tests de récupération de paramètres (EM et EAP). π\* est calculé **exactement**, pas approché — c'est ce qui a permis de mesurer l'erreur d'approximation Monte Carlo au lieu de la supposer.
- **Le pipeline sur données réelles va au bout.** 25,9 M réponses Junyi15 traitées en streaming, prérequis arbitrés par test binomial, calibration EM convergente et stable (écarts-types < 0,001 sur 5 redémarrages indépendants).
- **La règle de provenance des paramètres est appliquée partout.** Chaque nombre est dérivé, mesuré ou balayé — jamais choisi. C'est la garantie méthodologique qui rend le reste défendable.
- **La théorie prédit la mesure.** `questions_needed` prédit 19,2 questions sur piste B, le benchmark en mesure 18,6 — accord à 3 %, en forme fermée, sans avoir regardé le résultat d'abord. C'est le seul endroit du PFA où une formule prédit un chiffre expérimental au lieu de le commenter après coup.
- **Deux mesures indépendantes concordent.** L'information au sens KL (`item_information`) et le déplacement géométrique Fisher-Rao (`expected_fisher_rao_step`) classent les 5 concepts piste A dans exactement le même ordre.
- **La robustesse est testée sur quatre axes** (bruit mal estimé, structure de prérequis fausse, arène miroir, calibration de la confiance), pas seulement affirmée.
- **Les résultats négatifs sont rapportés tels quels** : le test D1, le compromis du Lot 3.6, l'inutilité pratique de l'approximation MC à cette échelle. C'est une force scientifique, pas un manque.
- **Tout est reproductible** : un script = une expérience = une commande, avec graine et commit git enregistrés (`results/REFERENCE.md`).

---

## 3. Points faibles, du plus grave au plus mineur

### 3.1 — Les deux moitiés du projet ne se rejoignent jamais *(faiblesse structurelle n°1)*

La piste A calibre des paramètres sur des données réelles, mais produit un `guess` de 0,50–0,75 qui rend le quiz inutilisable en pratique (74,8 questions mesurées pour un diagnostic). La piste B produit une démo qui marche (18,6 questions), mais sur des paramètres **non calibrés**.

**Il n'existe donc aucun point du projet où « données réelles » et « système qui fonctionne » coexistent.** C'est la première chose qu'un jury attentif remarquera. Le seuil de faisabilité (`guess < 0,43`) explique *pourquoi* proprement, mais ne referme pas le trou.

### 3.2 — La cause du `guess` élevé reste inconnue

Le test D1 a **écarté** l'hétérogénéité par granularité (subdiviser `arithmetic` ne fait pas baisser `guess` ; le fit isolé retombe sur le fit joint). Le test D2.1 (`PROMPT_D2_HYPOTHESE_Z.md`, `results/d2_1_signature_temporelle/`) a **écarté** l'hypothèse `z` fixe : chez les étudiants à taux de réussite intermédiaire (0,40–0,80, ceux qui tirent `guess` vers le haut dans l'EM), le taux de réussite ne progresse pas entre la première et la seconde moitié de leur période d'activité (delta moyen −0,020 sur 41 246 couples étudiant-concept, 47 % seulement en hausse) — et ce résultat, déjà nul voire négatif, ne survit à aucun degré au contrôle par permutation temporelle (p = 1). Ce n'est donc pas non plus un artefact d'apprentissage intra-fenêtre.

**Les deux candidats identifiés étaient les seuls avancés jusqu'ici, et les deux sont maintenant écartés par des tests directs, pas par manque de temps.** On sait ce que ce n'est pas (granularité, apprentissage temporel), pas ce que c'est. La piste D2.3 du document de passation (borner le gain potentiel sous un modèle à une transition de `z`) reste ouverte mais n'a de sens que si une nouvelle hypothèse candidate émerge — ce n'était pas le cas de D2.1, donc D2.2/D2.3 n'ont pas été lancées (conditionnées à un résultat positif de D2.1). À présenter dans le rapport comme un résultat négatif doublement mesuré, pas comme une limite non explorée.

### 3.3 — La baseline CAT-IRT est faible par construction

`a = 1,0` fixe, `b` dérivé d'une difficulté déclarative à trois niveaux, `c` emprunté au `guess` du BLIM. Un jury connaissant l'IRT dira, à raison, que la baseline est un homme de paille. L'arène miroir (Lot 3.4) désamorce **en partie** — elle montre que chaque modèle gagne sur sa propre vérité générative — mais elle ne montre pas qu'un IRT *correctement calibré* perdrait face au KST sur la vérité BLIM.

### 3.4 — Aucun étudiant réel n'a passé le quiz

Toute l'évaluation repose sur des étudiants simulés. La banque de 35 questions n'a jamais été soumise à des humains : ni validation de difficulté, ni taux de réussite empirique, ni vérification que les distracteurs attirent effectivement.

**Circularité douce à signaler** : la difficulté déclarative (1/2/3), posée à la main, alimente le paramètre `b` de la baseline IRT. La baseline est donc paramétrée par un jugement, pas par une mesure.

### 3.5 — La structure de prérequis de piste B est éditoriale

Les 4 arêtes de `domains/piste_b.yaml` viennent d'un raisonnement pédagogique, pas d'une dérivation sur données — contrairement à piste A, dont les prérequis sont arbitrés par test binomial sur les annotations Junyi. À énoncer explicitement dans le rapport pour ne pas laisser croire à une symétrie de rigueur entre les deux pistes.

### 3.6 — 161 tests ne valident pas le modèle

Ils garantissent que le code fait ce que la spécification dit, pas que la spécification décrit la réalité. Confusion facile à faire dans un mémoire — à éviter explicitement.

### 3.7 — Pas de baseline « pratique »

Le benchmark compare adaptatif / aléatoire / CAT-IRT. Il ne compare jamais à ce que fait réellement un enseignant : un QCM fixe de N questions, ou un score global. Le lecteur n'a donc aucun point de repère sur le gain par rapport à l'existant.

### 3.8 — L'algorithme n'est jamais mis sous contrainte à l'échelle

|Z| = 18 (piste A) et 50 (piste B). Le passage à l'échelle n'est démontré que sur domaines synthétiques (Lot 1, E3). Corollaire honnête, d'ailleurs assumé : l'approximation Monte Carlo, qui est un morceau entier du travail théorique, **n'a aucune utilité pratique à l'échelle de ce PFA** (seuil de rentabilité mesuré à |Z| ≈ 1 400).

### 3.9 — Une seule source de données

Junyi15 : un seul contexte (Taïwan, plateforme unique), une seule période (2012–2015), un seul domaine (mathématiques scolaires). Aucune indication de généralisation.

### 3.10 — Reproductibilité partielle

`results/` est tracé (graines, commits, logs d'exécution), mais les données brutes (1,2 Go) et les fichiers `.parquet` dérivés sont hors git. Reproduire depuis zéro suppose de retélécharger le dataset.

### 3.11 — ~~Tout le travail récent vit sur une branche~~ (réglé)

Les Lots 1 à 4 étaient sur `lot5-vocabulaire-theorie`, `master` restait à l'état d'avant. **Fusionné** (commit `b28dabb`) : `master` porte désormais l'intégralité des Lots 0-5 et de D2.1.

---

## 4. Ce qui reste à faire, par priorité

| Priorité | Tâche | Effort | Qui |
|---|---|---|---|
| **1** | **Rédiger le rapport** (partie code : plan validé, ~25-30 p) | Élevé | Younes + Yassine |
| **2** | Redéployer l'app Streamlit (tourne encore sur l'ancien domaine) | Faible | Younes |
| **3** | Ouvrir l'accès du collègue | Faible | Younes |
| 4 | *(si temps)* Calibrer honnêtement `a`/`b` de la baseline IRT sur Junyi (§3.3) | Moyen | — |
| 5 | *(cosmétique)* Point 1.5 du Lot 1 : figure \|Z\| vs nombre de concepts | Faible | — |

~~Fusionner `lot5-vocabulaire-theorie` dans `master`~~ : **fait** (commit `b28dabb`). ~~Tester l'hypothèse `z` évolutif comme cause du `guess` élevé~~ : **fait, résultat négatif** (D2.1, §3.2) — ce n'est plus une tâche en attente mais un résultat acquis.

Le point 4 renforcerait le mémoire mais n'est **pas** un prérequis pour rendre : la faiblesse correspondante (§3.3) est documentable comme limite assumée.

---

## 5. Verdict

Le travail technique est solide, honnête et complet : le plan a été exécuté de bout en bout, les résultats négatifs sont rapportés sans être maquillés, et chaque chiffre est traçable jusqu'à sa graine aléatoire. La faiblesse scientifique centrale — l'absence de jonction entre données réelles et système utilisable (§3.1), et l'énigme non résolue du `guess` (§3.2) — n'est **pas** un défaut d'exécution : c'est un résultat, désagréable mais mesuré, et il se défend s'il est présenté comme tel plutôt que contourné.

Le risque réel du projet est ailleurs : **tout ce matériau ne vaudra rien s'il n'est pas rédigé.** C'est là qu'il faut mettre l'effort restant.
