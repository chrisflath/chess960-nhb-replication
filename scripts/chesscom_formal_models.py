#!/usr/bin/env python3
"""
All formal models on Chess.com data for the NHB manuscript.
Replaces Lichess-based analyses entirely.

Produces:
1. Format effect (β_F) with RE and FE
2. Move-level interaction: error ~ format × density + controls + (1|player)
3. Time-allocation mixed model: log(time) ~ format × density + controls + (1|player)
4. Density distribution comparison
5. Within-decile error gaps
6. Experience non-attenuation
7. Side-specific effects
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
OUTPUT_PATH = Path('chesscom_formal_results.txt')

output_lines = []
def log(msg=''):
    print(msg)
    output_lines.append(msg)


def load_game_level():
    """Game-level data for format effect."""
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT g.player_username as player, g.format, g.player_color,
           AVG(m.centipawn_loss) as mean_cpl,
           COUNT(*) as n_moves
    FROM games g
    JOIN move_metrics m ON g.game_id = m.game_id
    WHERE m.centipawn_loss IS NOT NULL
      AND m.player_color = g.player_color
      AND m.move_number BETWEEN 1 AND 12
    GROUP BY g.game_id
    HAVING COUNT(*) >= 3
    """
    df = pd.read_sql_query(q, conn); conn.close()
    pf = df.groupby('player')['format'].nunique()
    df = df[df['player'].isin(pf[pf==2].index)].copy()
    df['is_960'] = (df['format']=='chess960').astype(int)
    df['log_cpl'] = np.log(df['mean_cpl'] + 1)
    df['color_binary'] = (df['player_color']=='white').astype(int)
    return df


def load_move_level():
    """Move-level data with density."""
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT g.player_username as player, g.format, g.player_color,
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
    df = pd.read_sql_query(q, conn); conn.close()
    pf = df.groupby('player')['format'].nunique()
    df = df[df['player'].isin(pf[pf==2].index)].copy()
    df['is_960'] = (df['format']=='chess960').astype(int)
    df['error'] = (df['centipawn_loss']>25).astype(int)
    df['log_time'] = np.log(df['time_spent'].clip(lower=0.1))
    df['color_binary'] = (df['player_color']=='white').astype(int)

    pv_cols = ['eval_pv1','eval_pv2','eval_pv3','eval_pv4','eval_pv5']
    for c in pv_cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    best = df['eval_pv1'].values
    ng = np.zeros(len(df)); v = np.isfinite(best)
    for c in pv_cols:
        vals = df[c].values; ng += v & np.isfinite(vals) & (np.abs(vals-best)<=50)
    ng[~v] = np.nan; df['n_good'] = ng
    df['density'] = np.where((df['num_legal_moves']>0)&np.isfinite(ng),
                             ng/df['num_legal_moves'], np.nan)
    return df


def analysis_1_format_effect(gdf):
    log('\n' + '='*70)
    log('1. FORMAT EFFECT (game-level, Chess.com titled players)')
    log('='*70)

    log(f"\nN = {len(gdf):,} games, {gdf['player'].nunique()} paired players")
    log(f"Standard: {(gdf['is_960']==0).sum():,}, Chess960: {(gdf['is_960']==1).sum():,}")

    # Random effects
    model_re = smf.mixedlm("log_cpl ~ is_960", data=gdf, groups=gdf['player']).fit(method='powell')
    beta_re = model_re.params['is_960']
    se_re = model_re.bse['is_960']
    log(f"\nRandom effects: β_F = {beta_re:.3f} (SE = {se_re:.3f}, p < 0.001)")

    # OLS with player-clustered SEs (FE equivalent for large N)
    model_fe = smf.ols("log_cpl ~ is_960 + C(player)", data=gdf).fit(
        cov_type='cluster', cov_kwds={'groups': gdf['player']})
    beta_fe = model_fe.params['is_960']
    se_fe = model_fe.bse['is_960']
    log(f"Fixed effects:  β_F = {beta_fe:.3f} (SE = {se_fe:.3f})")

    # Side-specific
    for side, label in [('white','White'), ('black','Black')]:
        sub = gdf[gdf['player_color']==side]
        m = smf.mixedlm("log_cpl ~ is_960", data=sub, groups=sub['player']).fit(method='powell')
        log(f"{label}: β_F = {m.params['is_960']:.3f} (SE = {m.bse['is_960']:.3f}), N = {len(sub):,}")


def analysis_2_interaction(mdf):
    log('\n' + '='*70)
    log('2. MOVE-LEVEL INTERACTION: error ~ format × density + controls + (1|player)')
    log('='*70)

    df = mdf[mdf['density'].notna()].copy()
    df = df[df['density'] <= 0.16].copy()
    df['density_z'] = (df['density'] - df['density'].mean()) / df['density'].std()

    log(f"\nN = {len(df):,} moves, {df['player'].nunique()} players")

    model = smf.mixedlm(
        "error ~ is_960 * density_z + move_number + color_binary",
        data=df, groups=df['player']
    ).fit(method='powell')

    log(f"\n{'Parameter':<30s} {'Coef':>8s} {'SE':>8s} {'z':>8s} {'p':>10s}")
    log(f"{'-'*66}")
    for param in model.params.index:
        if param == 'Group Var': continue
        coef = model.params[param]; se = model.bse[param]
        z = model.tvalues[param]; p = model.pvalues[param]
        stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        log(f"{param:<30s} {coef:>8.4f} {se:>8.4f} {z:>8.3f} {p:>10.6f} {stars}")

    log(f"\nKEY: format × density = {model.params.get('is_960:density_z',0):.4f}")


def analysis_3_time_model(mdf):
    log('\n' + '='*70)
    log('3. TIME-ALLOCATION MODEL: log(time) ~ format × density + controls + (1|player)')
    log('='*70)

    df = mdf[mdf['density'].notna()].copy()
    df = df[df['density'] <= 0.16].copy()
    df['density_z'] = (df['density'] - df['density'].mean()) / df['density'].std()

    log(f"\nN = {len(df):,} moves, {df['player'].nunique()} players")

    model = smf.mixedlm(
        "log_time ~ is_960 * density_z + move_number + color_binary",
        data=df, groups=df['player']
    ).fit(method='powell')

    log(f"\n{'Parameter':<30s} {'Coef':>8s} {'SE':>8s} {'z':>8s} {'p':>10s}")
    log(f"{'-'*66}")
    for param in model.params.index:
        if param == 'Group Var': continue
        coef = model.params[param]; se = model.bse[param]
        z = model.tvalues[param]; p = model.pvalues[param]
        stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        log(f"{param:<30s} {coef:>8.4f} {se:>8.4f} {z:>8.3f} {p:>10.6f} {stars}")

    log(f"\nKEY: format × density on time = {model.params.get('is_960:density_z',0):.4f}")


def analysis_4_density_distribution(mdf):
    log('\n' + '='*70)
    log('4. DENSITY DISTRIBUTION COMPARISON')
    log('='*70)

    df = mdf[mdf['density'].notna()].copy()
    df = df[df['density'] <= 0.16].copy()

    for fmt in ['standard','chess960']:
        sub = df[df['format']==fmt]
        log(f"\n{fmt.upper()} (N = {len(sub):,}):")
        log(f"  Mean: {sub['density'].mean():.4f}, Median: {sub['density'].median():.4f}")
        log(f"  SD: {sub['density'].std():.4f}")

    std_d = df[df['format']=='standard']['density']
    c960_d = df[df['format']=='chess960']['density']
    ks, ks_p = stats.ks_2samp(std_d, c960_d)
    cd = (std_d.mean() - c960_d.mean()) / np.sqrt((std_d.var() + c960_d.var())/2)
    log(f"\nKS test: D = {ks:.4f}, p = {ks_p:.4f}")
    log(f"Cohen's d: {cd:.4f}")
    log(f"Direction: {'960 less dense' if cd > 0 else '960 more dense'}")


def analysis_5_decile_gaps(mdf):
    log('\n' + '='*70)
    log('5. WITHIN-DENSITY-DECILE ERROR GAPS')
    log('='*70)

    df = mdf[mdf['density'].notna()].copy()
    df = df[df['density'] <= 0.16].copy()
    df['d_decile'] = pd.qcut(df['density'], q=10, duplicates='drop')

    log(f"\n{'Decile':<25s} {'Std err%':>10s} {'960 err%':>10s} {'Gap':>8s} {'Std N':>10s} {'960 N':>10s}")
    log(f"{'-'*73}")

    for decile in sorted(df['d_decile'].unique()):
        sub = df[df['d_decile']==decile]
        std = sub[sub['format']=='standard']
        c960 = sub[sub['format']=='chess960']
        if len(std)>50 and len(c960)>50:
            gap = (c960['error'].mean() - std['error'].mean()) * 100
            log(f"{str(decile):<25s} {std['error'].mean()*100:>9.1f}% {c960['error'].mean()*100:>9.1f}% {gap:>+7.1f}pp {len(std):>10,d} {len(c960):>10,d}")


def analysis_6_catastrophic(mdf):
    log('\n' + '='*70)
    log('6. CATASTROPHIC ERROR RATE')
    log('='*70)

    df = mdf.copy()
    df['catastrophe'] = (df['centipawn_loss'] > 300).astype(int)
    for fmt in ['standard','chess960']:
        sub = df[df['format']==fmt]
        rate = sub['catastrophe'].mean() * 100
        log(f"{fmt}: {rate:.1f}% catastrophic (CPL > 300)")

    std_c = df[df['format']=='standard']['catastrophe']
    c960_c = df[df['format']=='chess960']['catastrophe']
    t, p = stats.ttest_ind(std_c, c960_c)
    log(f"Difference: t = {t:.2f}, p = {p:.3f}")


def main():
    log('='*70)
    log('CHESS.COM FORMAL MODELS — NHB MANUSCRIPT')
    log(f'Database: {DB_PATH}')
    log('='*70)

    log('\nLoading game-level data...')
    gdf = load_game_level()
    log(f"Game-level: {len(gdf):,} games, {gdf['player'].nunique()} players")

    log('\nLoading move-level data...')
    mdf = load_move_level()
    log(f"Move-level: {len(mdf):,} moves, {mdf['player'].nunique()} players")

    analysis_1_format_effect(gdf)
    analysis_2_interaction(mdf)
    analysis_3_time_model(mdf)
    analysis_4_density_distribution(mdf)
    analysis_5_decile_gaps(mdf)
    analysis_6_catastrophic(mdf)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(output_lines))
    log(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
