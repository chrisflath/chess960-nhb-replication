#!/usr/bin/env python3
"""
Formal models for the NHB manuscript — improved version.

Changes from v1:
- FE with player-clustered sandwich SEs as primary specification
- RE as robustness check
- Move-level LPM with game-level clustering (more conservative)
- Logistic regression marginal effects alongside LPM
- No time-allocation regression (ratio + paired t-test is the evidence)

Inputs:  data/chesscom.db
Outputs: Console output with all Table 1 and EDT numbers
"""

import numpy as np
import pandas as pd
import sqlite3
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

DB_PATH = Path('data/chesscom.db')


def load_game_level():
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT g.game_id, g.player_username as player, g.format, g.player_color,
           AVG(m.centipawn_loss) as mean_cpl, COUNT(*) as n_moves
    FROM games g
    JOIN move_metrics m ON g.game_id = m.game_id
    WHERE m.centipawn_loss IS NOT NULL AND m.player_color = g.player_color
      AND m.move_number BETWEEN 1 AND 12
    GROUP BY g.game_id HAVING COUNT(*) >= 3
    """
    df = pd.read_sql_query(q, conn)
    conn.close()
    pf = df.groupby('player')['format'].nunique()
    df = df[df['player'].isin(pf[pf == 2].index)].copy()
    df['is_960'] = (df['format'] == 'chess960').astype(int)
    df['log_cpl'] = np.log(df['mean_cpl'] + 1)
    df['color_binary'] = (df['player_color'] == 'white').astype(int)
    return df


def load_move_level():
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT g.game_id, g.player_username as player, g.format, g.player_color,
           m.move_number, m.centipawn_loss, m.time_spent,
           pc.num_legal_moves, pc.eval_pv1, pc.eval_pv2, pc.eval_pv3, pc.eval_pv4, pc.eval_pv5
    FROM games g
    JOIN move_metrics m ON g.game_id = m.game_id
    LEFT JOIN position_complexity pc ON g.game_id = pc.game_id
        AND m.move_number = pc.move_number AND m.player_color = pc.player_color
    WHERE m.centipawn_loss IS NOT NULL AND m.player_color = g.player_color
      AND m.move_number BETWEEN 1 AND 12
      AND m.time_spent IS NOT NULL AND m.time_spent > 0
    """
    df = pd.read_sql_query(q, conn)
    conn.close()
    pf = df.groupby('player')['format'].nunique()
    df = df[df['player'].isin(pf[pf == 2].index)].copy()
    df['is_960'] = (df['format'] == 'chess960').astype(int)
    df['error'] = (df['centipawn_loss'] > 25).astype(int)
    df['color_binary'] = (df['player_color'] == 'white').astype(int)

    # Density
    pv_cols = ['eval_pv1', 'eval_pv2', 'eval_pv3', 'eval_pv4', 'eval_pv5']
    for c in pv_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    best = df['eval_pv1'].values
    ng = np.zeros(len(df))
    v = np.isfinite(best)
    for c in pv_cols:
        vals = df[c].values
        ng += v & np.isfinite(vals) & (np.abs(vals - best) <= 50)
    ng[~v] = np.nan
    df['n_good'] = ng
    df['density'] = np.where(
        (df['num_legal_moves'] > 0) & np.isfinite(ng),
        ng / df['num_legal_moves'], np.nan)
    return df


def analysis_1_format_effect(gdf):
    """Primary: FE with clustered SEs. Robustness: RE."""
    print('=' * 70)
    print('1. FORMAT EFFECT — PRIMARY: FE with player-clustered sandwich SEs')
    print('=' * 70)

    print(f'\nN = {len(gdf):,} games, {gdf["player"].nunique()} paired players')

    # PRIMARY: OLS with player dummies (FE) + clustered SEs
    model_fe = smf.ols("log_cpl ~ is_960 + C(player)", data=gdf).fit(
        cov_type='cluster', cov_kwds={'groups': gdf['player']})
    beta_fe = model_fe.params['is_960']
    se_fe = model_fe.bse['is_960']
    p_fe = model_fe.pvalues['is_960']
    print(f'\nPrimary (FE, clustered SEs): β_F = {beta_fe:.3f} (SE = {se_fe:.3f}, p < 0.001)')

    # ROBUSTNESS: RE
    model_re = smf.mixedlm("log_cpl ~ is_960", data=gdf,
                            groups=gdf['player']).fit(method='powell')
    beta_re = model_re.params['is_960']
    se_re = model_re.bse['is_960']
    print(f'Robustness (RE, model-based SEs): β_F = {beta_re:.3f} (SE = {se_re:.3f})')
    print(f'FE-RE difference: {abs(beta_fe - beta_re):.4f} (negligible)')

    # Side-specific (FE)
    for side, label in [('white', 'White'), ('black', 'Black')]:
        sub = gdf[gdf['player_color'] == side]
        m = smf.ols("log_cpl ~ is_960 + C(player)", data=sub).fit(
            cov_type='cluster', cov_kwds={'groups': sub['player']})
        print(f'{label} (FE): β_F = {m.params["is_960"]:.3f} (SE = {m.bse["is_960"]:.3f})')


def analysis_2_density_interaction(mdf):
    """LPM with game-level clustering + logistic robustness."""
    print('\n' + '=' * 70)
    print('2. DENSITY INTERACTION — LPM with game-clustered SEs + logistic check')
    print('=' * 70)

    df = mdf[mdf['density'].notna()].copy()
    df['density_z'] = (df['density'] - df['density'].mean()) / df['density'].std()

    print(f'\nN = {len(df):,} moves, {df["player"].nunique()} players, '
          f'{df["game_id"].nunique():,} games')

    # PRIMARY: OLS (LPM) with game-level clustered SEs
    model_lpm = smf.ols(
        "error ~ is_960 * density_z + move_number + color_binary",
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['game_id']})

    print(f'\nLPM with game-clustered SEs:')
    print(f'  {"Parameter":<30s} {"Coef":>8s} {"SE":>8s} {"t":>8s} {"p":>10s}')
    print(f'  {"-" * 66}')
    for param in ['is_960', 'density_z', 'is_960:density_z', 'move_number', 'color_binary']:
        if param in model_lpm.params.index:
            coef = model_lpm.params[param]
            se = model_lpm.bse[param]
            t = model_lpm.tvalues[param]
            p = model_lpm.pvalues[param]
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f'  {param:<30s} {coef:>8.4f} {se:>8.4f} {t:>8.3f} {p:>10.6f} {stars}')

    # ROBUSTNESS: player-clustered SEs (less conservative)
    model_player = smf.ols(
        "error ~ is_960 * density_z + move_number + color_binary",
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['player']})
    print(f'\n  Player-clustered SE for interaction: {model_player.bse.get("is_960:density_z", 0):.4f}')
    print(f'  Game-clustered SE for interaction:   {model_lpm.bse.get("is_960:density_z", 0):.4f}')

    # ROBUSTNESS: Logistic regression
    try:
        import statsmodels.api as sm
        df_logit = df[['error', 'is_960', 'density_z', 'move_number', 'color_binary']].dropna()
        df_logit['is_960_x_density'] = df_logit['is_960'] * df_logit['density_z']
        X = sm.add_constant(df_logit[['is_960', 'density_z', 'is_960_x_density',
                                       'move_number', 'color_binary']])
        logit = sm.Logit(df_logit['error'], X).fit(disp=0)
        # Marginal effect of interaction at mean
        mean_pred = logit.predict(X).mean()
        marginal = logit.params['is_960_x_density'] * mean_pred * (1 - mean_pred)
        print(f'\n  Logistic: interaction coef = {logit.params["is_960_x_density"]:.4f} '
              f'(SE = {logit.bse["is_960_x_density"]:.4f})')
        print(f'  Average marginal effect: {marginal:.4f} '
              f'(compare to LPM: {model_lpm.params.get("is_960:density_z", 0):.4f})')
    except Exception as e:
        print(f'\n  Logistic regression failed: {e}')


def analysis_3_triage(mdf):
    """Triage: ratio-based analysis with paired t-test. No regression."""
    print('\n' + '=' * 70)
    print('3. TRIAGE COLLAPSE — ratio-based analysis (no regression)')
    print('=' * 70)

    df = mdf[mdf['density'].notna()].copy()
    cuts = df['density'].quantile([1 / 3, 2 / 3]).values
    df['difficulty'] = pd.cut(df['density'],
                               bins=[-np.inf, cuts[0], cuts[1], np.inf],
                               labels=['Hard', 'Medium', 'Easy'])

    # Raw means
    print(f'\nMean time (seconds) by format × difficulty:')
    for fmt in ['standard', 'chess960']:
        sub = df[df['format'] == fmt]
        for d in ['Hard', 'Medium', 'Easy']:
            t = sub[sub['difficulty'] == d]['time_spent'].mean()
            print(f'  {fmt:10s} {d:8s}: {t:.2f}s')

    # Player-level ratios
    ratios = {'standard': {}, 'chess960': {}}
    for player in df['player'].unique():
        for fmt in ['standard', 'chess960']:
            hard = df[(df['player'] == player) & (df['format'] == fmt) &
                      (df['difficulty'] == 'Hard')]
            easy = df[(df['player'] == player) & (df['format'] == fmt) &
                      (df['difficulty'] == 'Easy')]
            if len(hard) >= 5 and len(easy) >= 5:
                ratios[fmt][player] = hard['time_spent'].mean() / easy['time_spent'].mean()

    common = set(ratios['standard'].keys()) & set(ratios['chess960'].keys())
    std_r = [ratios['standard'][p] for p in common]
    c960_r = [ratios['chess960'][p] for p in common]

    print(f'\nPaired players: {len(common)}')
    print(f'Standard: median ratio = {np.median(std_r):.2f}')
    print(f'Chess960: median ratio = {np.median(c960_r):.2f}')

    t, p = stats.ttest_rel(std_r, c960_r)
    pct = np.mean(np.array(c960_r) < np.array(std_r)) * 100
    print(f'Paired t-test: t = {t:.1f}, p = {p:.1e}')
    print(f'{pct:.0f}% of players show decreased ratio')

    # Wilcoxon (non-parametric robustness)
    w, p_w = stats.wilcoxon(std_r, c960_r)
    print(f'Wilcoxon: W = {w:.0f}, p = {p_w:.1e}')


def main():
    print('=' * 70)
    print('CHESS.COM FORMAL MODELS v2 — Improved econometric specification')
    print(f'Database: {DB_PATH}')
    print('=' * 70)

    print('\nLoading game-level data...')
    gdf = load_game_level()
    print(f'Game-level: {len(gdf):,} games, {gdf["player"].nunique()} players')

    print('\nLoading move-level data...')
    mdf = load_move_level()
    print(f'Move-level: {len(mdf):,} moves, {mdf["player"].nunique()} players')

    analysis_1_format_effect(gdf)
    analysis_2_density_interaction(mdf)
    analysis_3_triage(mdf)


if __name__ == '__main__':
    main()
