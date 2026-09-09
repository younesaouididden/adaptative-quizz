"""D2.1 -- signature temporelle de l'intermediarite (PROMPT_D2_HYPOTHESE_Z.md).

Teste l'hypothese D2 (le `guess` calibre eleve en piste A viendrait de
l'hypothese `z` fixe du BLIM, qui ne peut pas representer un etudiant qui
apprend PENDANT la fenetre d'observation 2012-2015) sans lancer un seul EM.

Prediction falsifiable (section 3 du document de passation) : si D2 est
vraie, les etudiants dont le taux de reussite GLOBAL est intermediaire
(ceux qui tirent `guess` vers le haut dans l'EM) doivent presenter une
hausse SYSTEMATIQUE et TEMPORELLE de leur taux de reussite -- pas des
succes disperses au hasard dans le temps. C'est ce que ce script mesure,
directement sur `data/responses.parquet` (colonne `timestamp`, deja
disponible, aucune reextraction necessaire).

Pre-enregistrement (fixe AVANT de regarder un seul resultat, valeurs
reprises telles quelles de la section 4 du document de passation) :
  - seuil minimum d'observations par (etudiant, concept) : n_total >= 20
  - bande de taux de reussite globale dite "intermediaire" : [0.40, 0.80]
  - decoupage temporel : par le TEMPS (point milieu = (min+max)/2 des
    timestamps du couple), JAMAIS par le nombre d'essais -- c'est la
    condition qui distingue D2 (apprentissage dans le temps) de la
    contamination par pratique repetee, deja ecartee en D0.
  - controle : loi hypergeometrique EXACTE (equivalente a une permutation
    de l'ordre temporel des labels correct/incorrect a l'interieur de
    chaque couple), 5000 replications, sur la moyenne AGREGEE de
    delta = taux_seconde_moitie - taux_premiere_moitie.
  - verdict : D2 confirmee si moyenne(delta) >= 0.05 (hausse jugee
    "substantielle") ET p-value unilaterale < 0.01 (survit au controle) ;
    sinon l'hypothese temporelle est rejetee (section 5 du document).

Deux passes chunkees sur le parquet (jamais le fichier entier en memoire
d'un coup, RAM limitee sur cette machine -- meme logique que
data/extract_responses.py) :
  passe 1 -- statistiques par (etudiant, concept) : n_correct, n_total,
             min(timestamp), max(timestamp) ;
  passe 2 -- une fois le point milieu connu par couple retenu, statistiques
             par (etudiant, concept, moitie).

Usage : python d2_1_signature_temporelle.py
Ecrit results/d2_1_signature_temporelle/{raw.csv, summary.csv, figure.pdf,
run.log}.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RESPONSES_PATH = ROOT / "data" / "responses.parquet"
RESULTS_DIR = ROOT / "results" / "d2_1_signature_temporelle"

COLUMNS = ["student_id", "concept_id", "correct", "timestamp"]
BATCH_SIZE = 1_000_000

MIN_N = 20
RATE_BAND = (0.40, 0.80)
N_REPS = 5000
SEED = 0
SUBSTANTIAL = 0.05
ALPHA = 0.01


# ---------------------------------------------------------------------------
# Passe 1 : agregation chunkee par (etudiant, concept)
# ---------------------------------------------------------------------------

def partial_pair_stats(chunk: pd.DataFrame) -> pd.DataFrame:
    """Un chunk -> (n_correct, n_total, min_ts, max_ts) par couple.

    Fonction pure, testee sur fixture synthetique.
    """
    g = chunk.groupby(["student_id", "concept_id"])
    counts = g["correct"].agg(n_correct="sum", n_total="count")
    times = g["timestamp"].agg(min_ts="min", max_ts="max")
    return counts.join(times).reset_index()


def merge_pair_stats(acc: pd.DataFrame | None, partial: pd.DataFrame) -> pd.DataFrame:
    """Combine l'accumulateur courant avec les stats partielles d'un
    nouveau chunk (un meme couple peut apparaitre dans plusieurs chunks --
    le log est ordonne dans le temps, pas par etudiant)."""
    if acc is None:
        return partial
    combined = pd.concat([acc, partial], ignore_index=True)
    return (combined.groupby(["student_id", "concept_id"], as_index=False)
            .agg(n_correct=("n_correct", "sum"), n_total=("n_total", "sum"),
                 min_ts=("min_ts", "min"), max_ts=("max_ts", "max")))


def compute_midpoints(pair_stats: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    """Filtre sur n_total >= min_n et ajoute le point milieu temporel.

    Couples avec min_ts == max_ts (toutes les reponses au meme instant)
    exclus : aucun decoupage temporel possible.
    """
    filtered = pair_stats[pair_stats["n_total"] >= min_n].copy()
    filtered = filtered[filtered["max_ts"] > filtered["min_ts"]]
    filtered["midpoint"] = (filtered["min_ts"] + filtered["max_ts"]) / 2
    return filtered


# ---------------------------------------------------------------------------
# Passe 2 : agregation chunkee par (etudiant, concept, moitie)
# ---------------------------------------------------------------------------

def partial_half_stats(chunk: pd.DataFrame, midpoints: pd.DataFrame) -> pd.DataFrame:
    """Un chunk -> (n_correct, n_total) par (couple, moitie), restreint aux
    couples retenus (jointure interne avec `midpoints`)."""
    merged = chunk.merge(midpoints[["student_id", "concept_id", "midpoint"]],
                         on=["student_id", "concept_id"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["student_id", "concept_id", "half", "n_correct", "n_total"])
    merged["half"] = np.where(merged["timestamp"] < merged["midpoint"], "first", "second")
    return (merged.groupby(["student_id", "concept_id", "half"])["correct"]
            .agg(n_correct="sum", n_total="count").reset_index())


def merge_half_stats(acc: pd.DataFrame | None, partial: pd.DataFrame) -> pd.DataFrame:
    if acc is None:
        return partial
    combined = pd.concat([acc, partial], ignore_index=True)
    return (combined.groupby(["student_id", "concept_id", "half"], as_index=False)
            .agg(n_correct=("n_correct", "sum"), n_total=("n_total", "sum")))


# ---------------------------------------------------------------------------
# Table finale et statistique agregee
# ---------------------------------------------------------------------------

def build_pairs_table(retained: pd.DataFrame, half_stats: pd.DataFrame,
                      rate_band: tuple[float, float] = RATE_BAND) -> pd.DataFrame:
    """Assemble (n_correct, n_total, n1, c1, n2, c2) par couple, restreint
    a la bande de taux de reussite globale dite "intermediaire", et calcule
    delta = taux_seconde_moitie - taux_premiere_moitie."""
    first = half_stats[half_stats["half"] == "first"].rename(
        columns={"n_correct": "c1", "n_total": "n1"})[["student_id", "concept_id", "c1", "n1"]]
    second = half_stats[half_stats["half"] == "second"].rename(
        columns={"n_correct": "c2", "n_total": "n2"})[["student_id", "concept_id", "c2", "n2"]]

    merged = retained.merge(first, on=["student_id", "concept_id"], how="inner")
    merged = merged.merge(second, on=["student_id", "concept_id"], how="inner")
    merged = merged[(merged["n1"] > 0) & (merged["n2"] > 0)].copy()

    merged["rate_global"] = merged["n_correct"] / merged["n_total"]
    merged = merged[(merged["rate_global"] >= rate_band[0]) & (merged["rate_global"] <= rate_band[1])]

    merged["rate1"] = merged["c1"] / merged["n1"]
    merged["rate2"] = merged["c2"] / merged["n2"]
    merged["delta"] = merged["rate2"] - merged["rate1"]

    return merged[["student_id", "concept_id", "n_correct", "n_total", "n1", "n2",
                   "rate_global", "rate1", "rate2", "delta"]].reset_index(drop=True)


def permutation_null(pairs: pd.DataFrame, n_reps: int, rng: np.random.Generator) -> np.ndarray:
    """Controle : distribution nulle de la moyenne agregee de delta sous
    permutation aleatoire de l'ordre temporel des labels correct/incorrect
    a l'interieur de chaque couple.

    Equivalence exacte (pas une approximation) : permuter au hasard quel
    sous-ensemble de n1 reponses (parmi n_total, dont n_correct correctes)
    tombe dans la premiere moitie revient a un tirage sans remise, donc
    c1_null ~ Hypergeometrique(n_total, n_correct, n1). Meme construction
    que le tirage hypergeometrique sur comptes agreges du tour 2 de D0
    (docs/revue_d0_tour2.md), reutilise ici pour la meme raison
    (efficacite -- pas de boucle Python par couple/replication).
    """
    n_total = pairs["n_total"].to_numpy()
    n_correct = pairs["n_correct"].to_numpy()
    n1 = pairs["n1"].to_numpy()
    n2 = pairs["n2"].to_numpy()
    nbad = n_total - n_correct

    c1_null = rng.hypergeometric(n_correct, nbad, n1, size=(n_reps, len(pairs)))
    rate1_null = c1_null / n1[None, :]
    rate2_null = (n_correct[None, :] - c1_null) / n2[None, :]
    delta_null = rate2_null - rate1_null
    return delta_null.mean(axis=1)


def verdict(mean_delta: float, p_value: float,
           substantial: float = SUBSTANTIAL, alpha: float = ALPHA) -> str:
    """Applique le critere pre-enregistre (section 5 du document)."""
    if mean_delta >= substantial and p_value < alpha:
        return ("D2 CONFIRMEE : hausse systematique et substantielle "
                f"(delta moyen={mean_delta:.4f} >= {substantial}), "
                f"survit au controle (p={p_value:.4g} < {alpha})")
    return ("D2 REJETEE : pas de hausse temporelle systematique et "
           f"substantielle qui survit au controle (delta moyen={mean_delta:.4f}, "
           f"p={p_value:.4g}) -- l'hypothese temporelle est morte, cf. section 5 "
           "du document de passation")


# ---------------------------------------------------------------------------
# Sorties (convention results/<experience>/, results/REFERENCE.md)
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True, cwd=ROOT)
        return r.stdout.strip()
    except Exception:
        return "inconnu (git indisponible)"


def fig_signature_temporelle(pairs: pd.DataFrame, null_agg: np.ndarray,
                             observed_mean: float, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.hist(pairs["delta"], bins=60, color="#2E86AB", alpha=0.85)
    ax1.axvline(0, color="black", linewidth=1, linestyle="--")
    ax1.axvline(observed_mean, color="#C0504D", linewidth=2,
               label=f"moyenne = {observed_mean:.4f}")
    ax1.set_xlabel("delta = taux (2e moitie) - taux (1re moitie)")
    ax1.set_ylabel("nombre de couples (etudiant, concept)")
    ax1.set_title("Distribution observee de delta\n(etudiants a taux global intermediaire)")
    ax1.legend()

    ax2.hist(null_agg, bins=60, color="#C0C0C0", alpha=0.85,
            label="controle (permutation temporelle)")
    ax2.axvline(observed_mean, color="#C0504D", linewidth=2, label="delta moyen observe")
    ax2.set_xlabel("delta moyen agrege")
    ax2.set_title("Distribution nulle vs valeur observee")
    ax2.legend()

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("D2.1 -- signature temporelle de l'intermediarite (Junyi15)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_run_log(path: Path, n_pairs_total: int, n_pairs_retained: int,
                  n_pairs_in_band: int, observed_mean: float, p_value: float,
                  v: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"date (UTC)             : {datetime.now(timezone.utc).isoformat()}\n")
        f.write("commande               : python d2_1_signature_temporelle.py\n")
        f.write(f"commit git             : {_git_commit()}\n")
        f.write(f"source                 : {RESPONSES_PATH.relative_to(ROOT)}\n")
        f.write(f"seuil n_total >=       : {MIN_N} (pre-enregistre)\n")
        f.write(f"bande intermediaire    : {RATE_BAND} (pre-enregistre)\n")
        f.write(f"couples (etu,concept) au total (avant filtre) : {n_pairs_total}\n")
        f.write(f"couples retenus (n_total>=seuil, splittable)  : {n_pairs_retained}\n")
        f.write(f"couples dans la bande intermediaire (analyse) : {n_pairs_in_band}\n")
        f.write(f"controle               : hypergeometrique exact, {N_REPS} replications, seed={SEED}\n")
        f.write(f"delta moyen observe    : {observed_mean:.6f}\n")
        f.write(f"p-value (unilaterale)  : {p_value:.6g}\n")
        f.write(f"critere pre-enregistre : delta>={SUBSTANTIAL} ET p<{ALPHA}\n\n")
        f.write(f"VERDICT : {v}\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(RESPONSES_PATH)

    print("Passe 1/2 -- statistiques par (etudiant, concept)...")
    acc = None
    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=COLUMNS)):
        chunk = batch.to_pandas()
        acc = merge_pair_stats(acc, partial_pair_stats(chunk))
        print(f"  ... chunk {i + 1}, {len(chunk):,} lignes, {len(acc):,} couples cumules", flush=True)
    pair_stats = acc
    n_pairs_total = len(pair_stats)

    retained = compute_midpoints(pair_stats, min_n=MIN_N)
    n_pairs_retained = len(retained)
    print(f"couples au total : {n_pairs_total:,}  |  retenus (n_total>={MIN_N}, splittable) : {n_pairs_retained:,}")

    midpoints = retained[["student_id", "concept_id", "midpoint"]]

    print("\nPasse 2/2 -- statistiques par (etudiant, concept, moitie)...")
    acc2 = None
    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=COLUMNS)):
        chunk = batch.to_pandas()
        partial = partial_half_stats(chunk, midpoints)
        if len(partial):
            acc2 = merge_half_stats(acc2, partial)
        print(f"  ... chunk {i + 1}/26 traite", flush=True)
    half_stats = acc2

    pairs = build_pairs_table(retained, half_stats, rate_band=RATE_BAND)
    n_pairs_in_band = len(pairs)
    print(f"couples dans la bande intermediaire {RATE_BAND} : {n_pairs_in_band:,}")

    if n_pairs_in_band == 0:
        raise RuntimeError("Aucun couple (etudiant, concept) dans la bande intermediaire "
                           "-- verifier les seuils avant de conclure quoi que ce soit.")

    observed_mean = float(pairs["delta"].mean())
    observed_median = float(pairs["delta"].median())
    frac_positive = float((pairs["delta"] > 0).mean())
    print(f"delta moyen={observed_mean:.4f}  median={observed_median:.4f}  "
         f"fraction positive={frac_positive:.1%}")

    rng = np.random.default_rng(SEED)
    null_agg = permutation_null(pairs, N_REPS, rng)
    p_value = (1 + int((null_agg >= observed_mean).sum())) / (1 + N_REPS)
    print(f"controle (n_reps={N_REPS}) : null moyen={null_agg.mean():.6f} "
         f"ecart-type={null_agg.std():.6f}  p-value unilaterale={p_value:.4g}")

    v = verdict(observed_mean, p_value)
    print(f"\n{v}")

    pairs.to_csv(RESULTS_DIR / "raw.csv", index=False)
    with open(RESULTS_DIR / "summary.csv", "w", encoding="utf-8") as f:
        f.write("metrique,valeur\n")
        f.write(f"n_pairs_total,{n_pairs_total}\n")
        f.write(f"n_pairs_retained,{n_pairs_retained}\n")
        f.write(f"n_pairs_in_band,{n_pairs_in_band}\n")
        f.write(f"delta_moyen,{observed_mean:.6f}\n")
        f.write(f"delta_median,{observed_median:.6f}\n")
        f.write(f"fraction_positive,{frac_positive:.6f}\n")
        f.write(f"null_moyen,{null_agg.mean():.6f}\n")
        f.write(f"null_ecart_type,{null_agg.std():.6f}\n")
        f.write(f"p_value_unilaterale,{p_value:.6g}\n")
        f.write(f"verdict,{v}\n")
    fig_signature_temporelle(pairs, null_agg, observed_mean, RESULTS_DIR / "figure.pdf")
    write_run_log(RESULTS_DIR / "run.log", n_pairs_total, n_pairs_retained, n_pairs_in_band,
                 observed_mean, p_value, v)

    print(f"\necrit : {RESULTS_DIR}/{{raw.csv, summary.csv, figure.pdf, run.log}}")


if __name__ == "__main__":
    main()
