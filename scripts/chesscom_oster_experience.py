#!/usr/bin/env python3
"""
Chess.com: Oster bounds + experience cohort analysis.
Fills the two MAJOR gaps identified by strategist-critic.
"""

import numpy as np
import pandas as pd
import sqlite3
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

DB_PATH = Path('/Users/chris/_claude-cowork/chess960/db/chesscom.db')
OUTPUT_PATH = Path('/Users/chris/_claude-cowork/chess960Paper/nhb/chesscom_oster_experience.txt')

output_lines = []
def log(msg=''):
    print(msg)
    output_lines.append(msg)


def load_game_level():
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT g.game_id, g.player_username as player, g.format, g.player_color,
           AVG(m.centipawn_loss) as mean_cpl, COUNT(*) as n_moves
    FROM games g
    JOIN move_metrics m ON g.game_id = m.game_id
    WHERE m.centipawn_loss IS NOT NULL
      AND m.player_color = g.player_color
      AND m.move_number BETWEEN 1 AND 12
    GROUP BY g.game_id
    HAVING COUNT(*) >= 3
    """
    df = pd.read_sql_query(q, conn)

    # Get player ratings
    players = pd.read_sql_query("SELECT username, standard_blitz_rating FROM players", conn)
    conn.close()

    pf = df.groupby('player')['format'].nunique()
    df = df[df['player'].isin(pf[pf == 2].index)].copy()
    df = df.merge(players, left_on='player', right_on='username', how='left')

    df['is_960'] = (df['format'] == 'chess960').astype(int)
    df['log_cpl'] = np.log(df['mean_cpl'] + 1)
    df['color_binary'] = (df['player_color'] == 'white').astype(int)
    return df


def oster_bounds(df):
    """Compute Oster (2019) coefficient stability bounds."""
    log('=' * 70)
    log('OSTER (2019) COEFFICIENT STABILITY BOUNDS — Chess.com')
    log('=' * 70)

    # Restricted model: just is_960 + player RE
    log('\nStep 1: Restricted model (format only)')
    m_restricted = smf.ols("log_cpl ~ is_960 + C(player)", data=df).fit()
    beta_r = m_restricted.params['is_960']
    r2_r = m_restricted.rsquared
    log(f"  β_restricted = {beta_r:.4f}")
    log(f"  R²_restricted = {r2_r:.4f}")

    # Full model: is_960 + controls + player FE
    log('\nStep 2: Full model (format + controls)')
    m_full = smf.ols("log_cpl ~ is_960 + color_binary + standard_blitz_rating + C(player)", data=df).fit()
    beta_f = m_full.params['is_960']
    r2_f = m_full.rsquared
    log(f"  β_full = {beta_f:.4f}")
    log(f"  R²_full = {r2_f:.4f}")

    # Oster delta
    r2_max = 1.3 * r2_f  # Oster's recommended bound
    if r2_max > 1.0:
        r2_max = min(1.0, r2_max)
        log(f"  R²_max capped at {r2_max:.4f}")
    else:
        log(f"  R²_max = 1.3 × R²_full = {r2_max:.4f}")

    # delta = (beta_f - 0) * (r2_max - r2_f) / ((beta_r - beta_f) * (r2_f - r2_r))
    numerator = beta_f * (r2_max - r2_f)
    denominator = (beta_r - beta_f) * (r2_f - r2_r)

    if abs(denominator) < 1e-10:
        delta = float('inf')
        log(f"\n  δ = ∞ (controls do not move the coefficient)")
    else:
        delta = numerator / denominator
        log(f"\n  δ = {delta:.1f}")

    log(f"\n  Interpretation: unobservables would need to be {abs(delta):.0f}× more")
    log(f"  important than included controls to explain away the format effect.")
    log(f"  (δ > 1 suggests robustness; δ > 3 is strong; δ > 10 is very strong)")

    return delta, r2_max, r2_f


def experience_cohorts(df):
    """Experience non-attenuation analysis on Chess.com."""
    log('\n' + '=' * 70)
    log('EXPERIENCE NON-ATTENUATION — Chess.com titled players')
    log('=' * 70)

    # Get game sequence number within Chess960
    c960 = df[df['format'] == 'chess960'].copy()
    c960 = c960.sort_values(['player', 'game_id'])
    c960['game_seq'] = c960.groupby('player').cumcount() + 1

    log(f"\nChess960 games: {len(c960):,}")
    log(f"Players: {c960['player'].nunique()}")
    log(f"Games per player: mean={c960.groupby('player').size().mean():.0f}, "
        f"median={c960.groupby('player').size().median():.0f}")

    # Cohort analysis
    cohort_bins = [(1, 20), (21, 50), (51, 100), (101, 9999)]
    cohort_labels = ['1-20', '21-50', '51-100', '100+']

    log(f"\nFormat gap by experience cohort:")
    log(f"{'Cohort':<12s} {'β_F':>8s} {'N players':>10s} {'N games':>10s}")
    log(f"{'-' * 42}")

    cohort_betas = []
    for (lo, hi), label in zip(cohort_bins, cohort_labels):
        # Get 960 games in this cohort
        cohort_960 = c960[(c960['game_seq'] >= lo) & (c960['game_seq'] <= hi)]
        cohort_players = cohort_960['player'].unique()

        if len(cohort_players) < 10:
            log(f"{label:<12s} {'---':>8s} {len(cohort_players):>10d} {len(cohort_960):>10d}")
            continue

        # Get ALL games (std + 960) for these players, but only 960 games in this cohort
        std_games = df[(df['player'].isin(cohort_players)) & (df['format'] == 'standard')]
        cohort_all = pd.concat([std_games, cohort_960])

        # Simple mean difference
        std_mean = std_games['log_cpl'].mean()
        c960_mean = cohort_960['log_cpl'].mean()
        gap = c960_mean - std_mean

        log(f"{label:<12s} {gap:>+7.3f} {len(cohort_players):>10d} {len(cohort_960):>10d}")
        cohort_betas.append(gap)

    if len(cohort_betas) >= 2:
        log(f"\nRange: {min(cohort_betas):.3f} to {max(cohort_betas):.3f}")
        log(f"Spread: {max(cohort_betas) - min(cohort_betas):.3f} log-CPL")
        log(f"Attenuation: {'NO' if max(cohort_betas) - min(cohort_betas) < 0.05 else 'POSSIBLE'}")


def main():
    log('Chess.com Oster + Experience Analysis')
    log(f'Database: {DB_PATH}')
    log(f'Seed: 42\n')

    df = load_game_level()
    log(f"Loaded: {len(df):,} games, {df['player'].nunique()} paired players")

    delta, r2_max, r2_full = oster_bounds(df)
    experience_cohorts(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(output_lines))
    log(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
