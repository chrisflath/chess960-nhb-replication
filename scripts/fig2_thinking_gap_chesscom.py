#!/usr/bin/env python3
"""
Figure 2: Thinking doesn't close the gap — Chess.com titled players.

Uses shorter time bins appropriate for blitz (3+0):
0-1s, 1-2s, 2-4s, 4-8s, 8+ seconds.
"""

import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

DB_PATH = Path('/Users/chris/_claude-cowork/chess960/db/chesscom.db')
OUT_DIR = Path('/Users/chris/_claude-cowork/chess960Paper/nhb/figures')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

COL = {'easy': '#B2182B', 'med': '#999999', 'hard': '#2166AC'}


def load_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT g.player_username as player, g.format, g.player_color,
           m.move_number, m.centipawn_loss, m.time_spent,
           pc.num_legal_moves, pc.eval_pv1, pc.eval_pv2,
           pc.eval_pv3, pc.eval_pv4, pc.eval_pv5
    FROM games g
    JOIN move_metrics m ON g.game_id = m.game_id
    LEFT JOIN position_complexity pc ON g.game_id = pc.game_id
        AND m.move_number = pc.move_number AND m.player_color = pc.player_color
    WHERE m.centipawn_loss IS NOT NULL AND m.player_color = g.player_color
      AND m.move_number BETWEEN 1 AND 12
      AND m.time_spent IS NOT NULL AND m.time_spent > 0
    """
    df = pd.read_sql_query(query, conn); conn.close()
    pf = df.groupby('player')['format'].nunique()
    df = df[df['player'].isin(pf[pf==2].index)].copy()
    df['error'] = (df['centipawn_loss']>25).astype(int)

    pv_cols = ['eval_pv1','eval_pv2','eval_pv3','eval_pv4','eval_pv5']
    for c in pv_cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    best = df['eval_pv1'].values
    ng = np.zeros(len(df)); v = np.isfinite(best)
    for c in pv_cols:
        vals = df[c].values; ng += v & np.isfinite(vals) & (np.abs(vals-best)<=50)
    ng[~v] = np.nan; df['n_good'] = ng
    df['density'] = np.where((df['num_legal_moves']>0)&np.isfinite(ng),
                             ng/df['num_legal_moves'], np.nan)
    print(f"Loaded {len(df):,} moves, {df['player'].nunique()} players")
    return df


def main():
    df = load_data()
    df_d = df[df['density'].notna()].copy()
    df_d = df_d[df_d['density'] <= 0.16].copy()

    cuts = df_d['density'].quantile([1/3, 2/3]).values
    df_d['difficulty'] = pd.cut(df_d['density'],
        bins=[-np.inf, cuts[0], cuts[1], np.inf],
        labels=['Hard', 'Medium', 'Easy'])

    # Blitz-appropriate time bins
    time_bins = [0, 1, 2, 4, 8, 999]
    time_labels = ['0\u20131', '1\u20132', '2\u20134', '4\u20138', '8+']
    df_d['time_bin'] = pd.cut(df_d['time_spent'], bins=time_bins, labels=time_labels)

    diff_colors = {'Easy': COL['easy'], 'Medium': COL['med'], 'Hard': COL['hard']}
    diff_markers = {'Easy': 'o', 'Medium': 's', 'Hard': '^'}

    fig, ax = plt.subplots(figsize=(90/25.4, 80/25.4))
    x = np.arange(len(time_labels))

    for difficulty in ['Easy', 'Medium', 'Hard']:
        gaps, cis = [], []
        for tl in time_labels:
            std = df_d[(df_d['difficulty']==difficulty)&(df_d['format']=='standard')&(df_d['time_bin']==tl)]
            c960 = df_d[(df_d['difficulty']==difficulty)&(df_d['format']=='chess960')&(df_d['time_bin']==tl)]
            if len(std)>=50 and len(c960)>=50:
                gap = (c960['error'].mean() - std['error'].mean()) * 100
                se = np.sqrt(std['error'].sem()**2 + c960['error'].sem()**2) * 100 * 1.96
                gaps.append(gap); cis.append(se)
            else:
                gaps.append(np.nan); cis.append(np.nan)

        gaps = np.array(gaps); cis = np.array(cis)
        valid = np.isfinite(gaps)
        ax.plot(x[valid], gaps[valid], marker=diff_markers[difficulty],
                color=diff_colors[difficulty], linewidth=1.8, markersize=6,
                label=f'{difficulty} positions', zorder=3,
                markeredgecolor='white', markeredgewidth=0.3)
        ax.fill_between(x[valid], (gaps-cis)[valid], (gaps+cis)[valid],
                        color=diff_colors[difficulty], alpha=0.12, zorder=2)
        mean_gap = np.nanmean(gaps[valid])
        print(f"{difficulty}: mean gap = {mean_gap:.1f}pp, values = {[f'{g:.1f}' for g in gaps if np.isfinite(g)]}")

    ax.axhline(0, color='#888', linewidth=0.6, linestyle='--', zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(time_labels, fontsize=7.5)
    ax.set_xlabel('Time per move (seconds)')
    ax.set_ylabel('Format gap in error rate (pp)')
    ax.set_ylim(-5, 30)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')

    total = len(df_d)
    ax.text(0.03, 0.97, f'N = {total/1e6:.1f}M moves\n{df_d["player"].nunique()} titled players',
            transform=ax.transAxes, fontsize=6.5, va='top', color='#777')

    plt.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(OUT_DIR / f'fig2_thinking_gap_chesscom.{ext}')
    plt.close()
    print(f"\nSaved fig2_thinking_gap_chesscom.pdf/png")


if __name__ == '__main__':
    main()
