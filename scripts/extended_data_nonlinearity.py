#!/usr/bin/env python3
"""
Extended Data analyses for Chess960 NHB paper.

Analysis 1: Nonlinearity tests (linear vs quadratic vs piecewise)
Analysis 2: Alternative distance metrics (equal weights, per-piece)
Analysis 3: Offset share bootstrap (cluster bootstrap, 1000 reps)
Analysis 4: Side-specific format effects (White vs Black)

Inputs:  /Users/chris/_claude-cowork/chess960/db/chess960.db
Outputs: /Users/chris/_claude-cowork/chess960Paper/nhb/extended_data_results.txt

Dependencies: numpy, pandas, statsmodels, scipy
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

set_seed = 42
np.random.seed(set_seed)

# ── Paths ──────────────────────────────────────────────────────────
DB_PATH = Path("/Users/chris/_claude-cowork/chess960/db/chess960.db")
OUTPUT_PATH = Path("/Users/chris/_claude-cowork/chess960Paper/nhb/extended_data_results.txt")

sys.path.insert(0, "/Users/chris/_claude-cowork/chess960")
from analysis.template_distance import compute_sp_features

DIVIDER = "=" * 80
SUBDIV = "-" * 60


# ── Data loading ───────────────────────────────────────────────────

def load_data():
    """Load game-level opening data (moves 1-12) for paired rapid players."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 30000")

    query = """
    SELECT
        g.id AS game_id,
        g.player_username AS player,
        g.player_color,
        g.format,
        CASE WHEN g.format = 'standard' THEN 518
             ELSE g.chess960_position END AS sp,
        p.standard_rating AS r_std,
        p.chess960_rating AS r_960,
        COALESCE(pm.file_distance, 0) AS file_distance,
        AVG(LOG(m.centipawn_loss + 1)) AS mean_log_cpl,
        COUNT(*) AS n_moves
    FROM games g
    JOIN move_metrics m ON g.id = m.game_id
    JOIN players p ON g.player_username = p.username
    LEFT JOIN chess960_position_metrics pm ON g.chess960_position = pm.sp
    WHERE g.time_control = 'rapid'
      AND m.centipawn_loss IS NOT NULL
      AND m.player_color = g.player_color
      AND m.move_number BETWEEN 1 AND 12
      AND p.standard_rating IS NOT NULL
      AND p.chess960_rating IS NOT NULL
      AND (g.format = 'standard' OR g.chess960_position IS NOT NULL)
    GROUP BY g.id
    HAVING COUNT(*) >= 10
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # 300-game cap per player per format
    df = (df.sort_values("game_id")
            .groupby(["player", "format"], sort=False)
            .head(300)
            .reset_index(drop=True))

    # Paired players only
    pf = df.groupby("player")["format"].nunique()
    paired = pf[pf == 2].index
    df = df[df["player"].isin(paired)].copy()

    # Merge SP features
    sp_feat = compute_sp_features()
    df = df.merge(sp_feat.drop(columns=["arrangement"]), on="sp", how="left")

    df["is_960"] = (df["format"] == "chess960").astype(int)

    print(f"Loaded {len(df):,} games from {df['player'].nunique()} paired players")
    print(f"  Standard: {(df['is_960']==0).sum():,}  |  Chess960: {(df['is_960']==1).sum():,}")
    return df


def ols_clustered(formula, data, cluster_col="player"):
    """OLS with player-clustered standard errors."""
    mod = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
    )
    return mod


# ── Analysis 1: Nonlinearity tests ────────────────────────────────

def analysis_nonlinearity(df, out):
    out.write(f"\n{DIVIDER}\n")
    out.write("ANALYSIS 1: NONLINEARITY IN DOSE-RESPONSE (Chess960 games only)\n")
    out.write(f"{DIVIDER}\n\n")

    c9 = df[df["is_960"] == 1].copy()
    out.write(f"N = {len(c9):,} Chess960 games, {c9['player'].nunique()} players\n")
    out.write(f"file_distance range: [{c9['file_distance'].min()}, {c9['file_distance'].max()}]\n")
    out.write(f"file_distance mean: {c9['file_distance'].mean():.2f}, SD: {c9['file_distance'].std():.2f}\n\n")

    # Construct variables
    c9["fd_sq"] = c9["file_distance"] ** 2
    c9["fd_above10"] = np.maximum(c9["file_distance"] - 10, 0)

    results = {}

    # Model 1: Linear
    try:
        m1 = smf.mixedlm("mean_log_cpl ~ file_distance", c9, groups=c9["player"]).fit(reml=True)
        results["Linear (RE)"] = m1
        out.write("Model 1: Linear with player random effects\n")
        out.write(f"  file_distance:  coef = {m1.params['file_distance']:.6f}  "
                  f"SE = {m1.bse['file_distance']:.6f}  "
                  f"p = {m1.pvalues['file_distance']:.4g}\n")
        out.write(f"  Intercept:      coef = {m1.params['Intercept']:.6f}\n")
        out.write(f"  AIC = {m1.aic:.1f}  BIC = {m1.bic:.1f}\n")
        out.write(f"  Log-likelihood = {m1.llf:.1f}\n\n")
        used_re = True
    except Exception as e:
        out.write(f"  Mixed model failed ({e}), falling back to OLS\n")
        m1 = ols_clustered("mean_log_cpl ~ file_distance", c9)
        results["Linear (OLS)"] = m1
        out.write(f"  file_distance:  coef = {m1.params['file_distance']:.6f}  "
                  f"SE = {m1.bse['file_distance']:.6f}  "
                  f"p = {m1.pvalues['file_distance']:.4g}\n")
        out.write(f"  R-squared = {m1.rsquared:.6f}\n")
        out.write(f"  AIC = {m1.aic:.1f}  BIC = {m1.bic:.1f}\n\n")
        used_re = False

    # Model 2: Quadratic
    try:
        if used_re:
            m2 = smf.mixedlm("mean_log_cpl ~ file_distance + fd_sq", c9, groups=c9["player"]).fit(reml=True)
            results["Quadratic (RE)"] = m2
            out.write("Model 2: Quadratic with player random effects\n")
            out.write(f"  file_distance:  coef = {m2.params['file_distance']:.6f}  "
                      f"SE = {m2.bse['file_distance']:.6f}  "
                      f"p = {m2.pvalues['file_distance']:.4g}\n")
            out.write(f"  fd_sq:          coef = {m2.params['fd_sq']:.6f}  "
                      f"SE = {m2.bse['fd_sq']:.6f}  "
                      f"p = {m2.pvalues['fd_sq']:.4g}\n")
            out.write(f"  AIC = {m2.aic:.1f}  BIC = {m2.bic:.1f}\n")
            out.write(f"  Log-likelihood = {m2.llf:.1f}\n\n")
        else:
            raise Exception("use OLS")
    except Exception:
        m2 = ols_clustered("mean_log_cpl ~ file_distance + fd_sq", c9)
        results["Quadratic (OLS)"] = m2
        out.write("Model 2: Quadratic with clustered SEs\n")
        out.write(f"  file_distance:  coef = {m2.params['file_distance']:.6f}  "
                  f"SE = {m2.bse['file_distance']:.6f}  "
                  f"p = {m2.pvalues['file_distance']:.4g}\n")
        out.write(f"  fd_sq:          coef = {m2.params['fd_sq']:.6f}  "
                  f"SE = {m2.bse['fd_sq']:.6f}  "
                  f"p = {m2.pvalues['fd_sq']:.4g}\n")
        out.write(f"  R-squared = {m2.rsquared:.6f}\n")
        out.write(f"  AIC = {m2.aic:.1f}  BIC = {m2.bic:.1f}\n\n")

    # Model 3: Piecewise linear (breakpoint at d=10)
    try:
        if used_re:
            m3 = smf.mixedlm("mean_log_cpl ~ file_distance + fd_above10", c9, groups=c9["player"]).fit(reml=True)
            results["Piecewise (RE)"] = m3
            out.write("Model 3: Piecewise linear (break at d=10) with player random effects\n")
            out.write(f"  file_distance:  coef = {m3.params['file_distance']:.6f}  "
                      f"SE = {m3.bse['file_distance']:.6f}  "
                      f"p = {m3.pvalues['file_distance']:.4g}\n")
            out.write(f"  fd_above10:     coef = {m3.params['fd_above10']:.6f}  "
                      f"SE = {m3.bse['fd_above10']:.6f}  "
                      f"p = {m3.pvalues['fd_above10']:.4g}\n")
            out.write(f"  AIC = {m3.aic:.1f}  BIC = {m3.bic:.1f}\n")
            out.write(f"  Log-likelihood = {m3.llf:.1f}\n\n")
        else:
            raise Exception("use OLS")
    except Exception:
        m3 = ols_clustered("mean_log_cpl ~ file_distance + fd_above10", c9)
        results["Piecewise (OLS)"] = m3
        out.write("Model 3: Piecewise linear (break at d=10) with clustered SEs\n")
        out.write(f"  file_distance:  coef = {m3.params['file_distance']:.6f}  "
                  f"SE = {m3.bse['file_distance']:.6f}  "
                  f"p = {m3.pvalues['file_distance']:.4g}\n")
        out.write(f"  fd_above10:     coef = {m3.params['fd_above10']:.6f}  "
                  f"SE = {m3.bse['fd_above10']:.6f}  "
                  f"p = {m3.pvalues['fd_above10']:.4g}\n")
        out.write(f"  R-squared = {m3.rsquared:.6f}\n")
        out.write(f"  AIC = {m3.aic:.1f}  BIC = {m3.bic:.1f}\n\n")

    # Model comparison summary
    out.write(f"{SUBDIV}\n")
    out.write("Model comparison summary:\n")
    for name, mod in results.items():
        aic = mod.aic if hasattr(mod, "aic") else "N/A"
        bic = mod.bic if hasattr(mod, "bic") else "N/A"
        out.write(f"  {name:<25s}  AIC = {aic:>10}  BIC = {bic:>10}\n")
    out.write(f"\nConclusion: Lower AIC/BIC is preferred.\n")
    out.write(f"If quadratic/piecewise terms are insignificant, linearity is supported.\n\n")

    return results


# ── Analysis 2: Alternative distance metrics ──────────────────────

def analysis_alternative_metrics(df, out):
    out.write(f"\n{DIVIDER}\n")
    out.write("ANALYSIS 2: ALTERNATIVE DISTANCE METRICS\n")
    out.write(f"{DIVIDER}\n\n")

    c9 = df[df["is_960"] == 1].copy()

    # 2a: Equal-weight sum of raw displacements
    c9["equal_weight_dist"] = (
        c9["delta_rooks"] + c9["delta_bishops"] + c9["delta_knights"]
        + c9["delta_king"] + c9["delta_queen"]
    )

    out.write("2a. Equal-weight distance (sum of all piece displacements)\n")
    out.write(f"    Range: [{c9['equal_weight_dist'].min()}, {c9['equal_weight_dist'].max()}]\n")
    out.write(f"    Mean: {c9['equal_weight_dist'].mean():.2f}, SD: {c9['equal_weight_dist'].std():.2f}\n")
    out.write(f"    Correlation with file_distance: {c9['equal_weight_dist'].corr(c9['file_distance']):.4f}\n\n")

    try:
        m_eq = smf.mixedlm("mean_log_cpl ~ equal_weight_dist", c9, groups=c9["player"]).fit(reml=True)
        out.write(f"    equal_weight_dist: coef = {m_eq.params['equal_weight_dist']:.6f}  "
                  f"SE = {m_eq.bse['equal_weight_dist']:.6f}  "
                  f"p = {m_eq.pvalues['equal_weight_dist']:.4g}\n")
        out.write(f"    AIC = {m_eq.aic:.1f}\n\n")
    except Exception:
        m_eq = ols_clustered("mean_log_cpl ~ equal_weight_dist", c9)
        out.write(f"    equal_weight_dist: coef = {m_eq.params['equal_weight_dist']:.6f}  "
                  f"SE = {m_eq.bse['equal_weight_dist']:.6f}  "
                  f"p = {m_eq.pvalues['equal_weight_dist']:.4g}\n")
        out.write(f"    R-squared = {m_eq.rsquared:.6f}, AIC = {m_eq.aic:.1f}\n\n")

    # 2b: Per-piece regressions
    pieces = ["delta_rooks", "delta_bishops", "delta_knights", "delta_king", "delta_queen"]

    out.write("2b. Per-piece displacement regressions\n")
    out.write(f"    {'Piece':<18s} {'Coef':>10s} {'SE':>10s} {'t':>8s} {'p':>10s}\n")
    out.write(f"    {SUBDIV}\n")

    piece_results = {}
    for piece in pieces:
        try:
            m_p = smf.mixedlm(f"mean_log_cpl ~ {piece}", c9, groups=c9["player"]).fit(reml=True)
            coef = m_p.params[piece]
            se = m_p.bse[piece]
            pval = m_p.pvalues[piece]
            tstat = coef / se
        except Exception:
            m_p = ols_clustered(f"mean_log_cpl ~ {piece}", c9)
            coef = m_p.params[piece]
            se = m_p.bse[piece]
            pval = m_p.pvalues[piece]
            tstat = m_p.tvalues[piece]

        piece_results[piece] = {"coef": coef, "se": se, "t": tstat, "p": pval}
        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        out.write(f"    {piece:<18s} {coef:>10.6f} {se:>10.6f} {tstat:>8.2f} {pval:>10.4g} {stars}\n")

    out.write(f"\n")

    # 2c: Horse race (all pieces together)
    out.write("2c. Horse race: all piece displacements jointly\n")
    formula = "mean_log_cpl ~ " + " + ".join(pieces)
    try:
        m_all = smf.mixedlm(formula, c9, groups=c9["player"]).fit(reml=True)
        out.write(f"    {'Piece':<18s} {'Coef':>10s} {'SE':>10s} {'p':>10s}\n")
        out.write(f"    {SUBDIV}\n")
        for piece in pieces:
            coef = m_all.params[piece]
            se = m_all.bse[piece]
            pval = m_all.pvalues[piece]
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            out.write(f"    {piece:<18s} {coef:>10.6f} {se:>10.6f} {pval:>10.4g} {stars}\n")
        out.write(f"    AIC = {m_all.aic:.1f}\n\n")
    except Exception:
        m_all = ols_clustered(formula, c9)
        out.write(f"    {'Piece':<18s} {'Coef':>10s} {'SE':>10s} {'p':>10s}\n")
        out.write(f"    {SUBDIV}\n")
        for piece in pieces:
            coef = m_all.params[piece]
            se = m_all.bse[piece]
            pval = m_all.pvalues[piece]
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            out.write(f"    {piece:<18s} {coef:>10.6f} {se:>10.6f} {pval:>10.4g} {stars}\n")
        out.write(f"    R-squared = {m_all.rsquared:.6f}, AIC = {m_all.aic:.1f}\n\n")

    return piece_results


# ── Analysis 3: Offset share bootstrap ────────────────────────────

def analysis_offset_bootstrap(df, out, n_boot=1000):
    out.write(f"\n{DIVIDER}\n")
    out.write(f"ANALYSIS 3: OFFSET SHARE BOOTSTRAP ({n_boot} replications)\n")
    out.write(f"{DIVIDER}\n\n")

    # Point estimate first
    c9 = df[df["is_960"] == 1].copy()
    mean_dist = c9["file_distance"].mean()

    # Format effect (full sample)
    try:
        m_format = smf.mixedlm("mean_log_cpl ~ is_960", df, groups=df["player"]).fit(reml=True)
        beta_F = m_format.params["is_960"]
        beta_F_se = m_format.bse["is_960"]
    except Exception:
        m_format = ols_clustered("mean_log_cpl ~ is_960", df)
        beta_F = m_format.params["is_960"]
        beta_F_se = m_format.bse["is_960"]

    # Within-960 gradient
    try:
        m_grad = smf.mixedlm("mean_log_cpl ~ file_distance", c9, groups=c9["player"]).fit(reml=True)
        gamma = m_grad.params["file_distance"]
        gamma_se = m_grad.bse["file_distance"]
    except Exception:
        m_grad = ols_clustered("mean_log_cpl ~ file_distance", c9)
        gamma = m_grad.params["file_distance"]
        gamma_se = m_grad.bse["file_distance"]

    offset_share = (beta_F - gamma * mean_dist) / beta_F

    out.write(f"Point estimates:\n")
    out.write(f"  beta_F (format effect):      {beta_F:.6f}  (SE = {beta_F_se:.6f})\n")
    out.write(f"  gamma (distance gradient):   {gamma:.6f}  (SE = {gamma_se:.6f})\n")
    out.write(f"  mean file_distance (960):    {mean_dist:.4f}\n")
    out.write(f"  gamma * mean_dist:           {gamma * mean_dist:.6f}\n")
    out.write(f"  Offset share:                {offset_share:.4f} ({offset_share*100:.1f}%)\n\n")

    # Cluster bootstrap
    players = df["player"].unique()
    n_players = len(players)
    boot_shares = []
    boot_betas = []
    boot_gammas = []

    out.write(f"Running cluster bootstrap ({n_boot} reps, {n_players} players)...\n")

    # Pre-group data by player for fast bootstrap
    player_groups = {p: g for p, g in df.groupby("player")}

    for b in range(n_boot):
        # Resample players with replacement
        boot_players = np.random.choice(players, size=n_players, replace=True)

        # Build bootstrap sample using pre-grouped data
        frames = [player_groups[p] for p in boot_players]
        boot_df = pd.concat(frames, ignore_index=True)

        boot_c9 = boot_df[boot_df["is_960"] == 1]
        boot_mean_dist = boot_c9["file_distance"].mean()

        try:
            # Format effect
            m_b_format = smf.ols("mean_log_cpl ~ is_960", boot_df).fit()
            b_beta = m_b_format.params["is_960"]

            # Gradient
            m_b_grad = smf.ols("mean_log_cpl ~ file_distance", boot_c9).fit()
            b_gamma = m_b_grad.params["file_distance"]

            b_share = (b_beta - b_gamma * boot_mean_dist) / b_beta if b_beta != 0 else np.nan

            boot_betas.append(b_beta)
            boot_gammas.append(b_gamma)
            boot_shares.append(b_share)
        except Exception:
            continue

        if (b + 1) % 200 == 0:
            print(f"  Bootstrap: {b+1}/{n_boot}")

    boot_shares = np.array([s for s in boot_shares if np.isfinite(s)])
    boot_betas = np.array(boot_betas)
    boot_gammas = np.array(boot_gammas)

    out.write(f"\nBootstrap results ({len(boot_shares)} successful replications):\n\n")

    # Offset share
    ci_lo, ci_hi = np.percentile(boot_shares, [2.5, 97.5])
    out.write(f"  Offset share:\n")
    out.write(f"    Point estimate: {offset_share:.4f}\n")
    out.write(f"    Boot mean:      {boot_shares.mean():.4f}\n")
    out.write(f"    Boot SD:        {boot_shares.std():.4f}\n")
    out.write(f"    95% CI:         [{ci_lo:.4f}, {ci_hi:.4f}]\n\n")

    # Beta_F
    ci_lo_b, ci_hi_b = np.percentile(boot_betas, [2.5, 97.5])
    out.write(f"  beta_F (format effect):\n")
    out.write(f"    Point estimate: {beta_F:.6f}\n")
    out.write(f"    Boot mean:      {boot_betas.mean():.6f}\n")
    out.write(f"    95% CI:         [{ci_lo_b:.6f}, {ci_hi_b:.6f}]\n\n")

    # Gamma
    ci_lo_g, ci_hi_g = np.percentile(boot_gammas, [2.5, 97.5])
    out.write(f"  gamma (distance gradient):\n")
    out.write(f"    Point estimate: {gamma:.6f}\n")
    out.write(f"    Boot mean:      {boot_gammas.mean():.6f}\n")
    out.write(f"    95% CI:         [{ci_lo_g:.6f}, {ci_hi_g:.6f}]\n\n")

    return {
        "offset_share": offset_share,
        "ci": (ci_lo, ci_hi),
        "beta_F": beta_F,
        "gamma": gamma,
        "mean_dist": mean_dist,
    }


# ── Analysis 4: Side-specific effects ─────────────────────────────

def analysis_side_effects(df, out):
    out.write(f"\n{DIVIDER}\n")
    out.write("ANALYSIS 4: SIDE-SPECIFIC FORMAT EFFECTS\n")
    out.write(f"{DIVIDER}\n\n")

    out.write(f"{'Side':<8s} {'N':>8s} {'beta_F':>10s} {'SE':>10s} {'t':>8s} {'p':>10s}\n")
    out.write(f"{SUBDIV}\n")

    results = {}
    for color in ["white", "black"]:
        sub = df[df["player_color"] == color].copy()
        n_games = len(sub)
        n_players = sub["player"].nunique()

        try:
            m = smf.mixedlm("mean_log_cpl ~ is_960", sub, groups=sub["player"]).fit(reml=True)
            coef = m.params["is_960"]
            se = m.bse["is_960"]
            pval = m.pvalues["is_960"]
            tstat = coef / se
        except Exception:
            m = ols_clustered("mean_log_cpl ~ is_960", sub)
            coef = m.params["is_960"]
            se = m.bse["is_960"]
            pval = m.pvalues["is_960"]
            tstat = m.tvalues["is_960"]

        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        out.write(f"{color:<8s} {n_games:>8,d} {coef:>10.6f} {se:>10.6f} {tstat:>8.2f} {pval:>10.4g} {stars}\n")
        results[color] = {"coef": coef, "se": se, "p": pval, "n": n_games, "n_players": n_players}

    # Full sample for comparison
    try:
        m_full = smf.mixedlm("mean_log_cpl ~ is_960", df, groups=df["player"]).fit(reml=True)
        coef = m_full.params["is_960"]
        se = m_full.bse["is_960"]
        pval = m_full.pvalues["is_960"]
        tstat = coef / se
    except Exception:
        m_full = ols_clustered("mean_log_cpl ~ is_960", df)
        coef = m_full.params["is_960"]
        se = m_full.bse["is_960"]
        pval = m_full.pvalues["is_960"]
        tstat = m_full.tvalues["is_960"]

    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    out.write(f"{'pooled':<8s} {len(df):>8,d} {coef:>10.6f} {se:>10.6f} {tstat:>8.2f} {pval:>10.4g} {stars}\n")

    out.write(f"\n")

    # Test for difference between White and Black
    diff = results["white"]["coef"] - results["black"]["coef"]
    diff_se = np.sqrt(results["white"]["se"]**2 + results["black"]["se"]**2)
    diff_z = diff / diff_se
    diff_p = 2 * (1 - stats.norm.cdf(abs(diff_z)))

    out.write(f"Difference (White - Black): {diff:.6f}  SE = {diff_se:.6f}  z = {diff_z:.2f}  p = {diff_p:.4g}\n")
    out.write(f"Interpretation: {'No significant' if diff_p > 0.05 else 'Significant'} difference in format effect by side.\n\n")

    return results


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_data()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as out:
        out.write(f"{'=' * 80}\n")
        out.write(f"EXTENDED DATA ANALYSES — Chess960 NHB Paper\n")
        out.write(f"{'=' * 80}\n\n")
        out.write(f"Data: {len(df):,} games, {df['player'].nunique()} paired players\n")
        out.write(f"Format: Standard = {(df['is_960']==0).sum():,}, Chess960 = {(df['is_960']==1).sum():,}\n")
        out.write(f"Seed: {set_seed}\n")

        print("\n--- Analysis 1: Nonlinearity tests ---")
        analysis_nonlinearity(df, out)

        print("\n--- Analysis 2: Alternative distance metrics ---")
        analysis_alternative_metrics(df, out)

        print("\n--- Analysis 3: Offset share bootstrap ---")
        analysis_offset_bootstrap(df, out, n_boot=1000)

        print("\n--- Analysis 4: Side-specific effects ---")
        analysis_side_effects(df, out)

        out.write(f"\n{'=' * 80}\n")
        out.write(f"END OF REPORT\n")
        out.write(f"{'=' * 80}\n")

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
