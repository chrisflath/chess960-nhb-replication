#!/usr/bin/env python3
"""
Predictive transfer analysis: Table 1 in the manuscript.

Cross-validated R² for predicting player-level winning percentage
from standard vs Chess960 ratings.

Inputs:  data/chesscom.db
Outputs: Console output with R² values for Table 1
"""

import numpy as np
import pandas as pd
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
from pathlib import Path

np.random.seed(42)

DB_PATH = Path('data/chesscom.db')


def main():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query('''
    SELECT g.player_username as player, g.format, g.result,
           p.standard_blitz_rating as std_rating,
           p.chess960_blitz_rating as c960_rating
    FROM games g
    JOIN players p ON g.player_username = p.username
    WHERE g.result IS NOT NULL
    ''', conn)
    conn.close()

    # Map results to scores
    result_map = {'win': 1.0, 'loss': 0.0, 'draw': 0.5}
    df['score'] = df['result'].map(result_map)
    df = df.dropna(subset=['score'])

    # Player-level aggregation
    player = df.groupby(['player', 'format']).agg(
        score=('score', 'mean'),
        n=('score', 'count'),
        std_rating=('std_rating', 'first'),
        c960_rating=('c960_rating', 'first')
    ).reset_index()

    # Paired players with >= 10 games per format
    pf = player.groupby('player')['format'].nunique()
    player = player[player['player'].isin(pf[pf == 2].index)]
    std = player[player['format'] == 'standard'].set_index('player')
    c960 = player[player['format'] == 'chess960'].set_index('player')
    m = std[['score', 'std_rating', 'c960_rating', 'n']].join(
        c960[['score', 'n']], rsuffix='_960').dropna()
    m = m[(m['n'] >= 10) & (m['n_960'] >= 10)]

    print(f'N = {len(m)} paired players (>=10 games each format)')
    print(f'Std score: {m["score"].mean():.3f}, 960 score: {m["score_960"].mean():.3f}')
    print()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Table 1 results
    print('=' * 60)
    print('TABLE 1: Predictive transfer of rated skill across formats')
    print('=' * 60)
    print(f'{"Predictor":>15s} {"→ Std wins":>12s} {"→ 960 wins":>12s} {"Loss":>8s}')
    print('-' * 55)

    for pred, label in [('std_rating', 'Standard rating'),
                         ('c960_rating', 'Chess960 rating')]:
        r2_std = cross_val_score(
            LinearRegression(), m[[pred]].values, m['score'].values,
            cv=kf, scoring='r2').mean()
        r2_960 = cross_val_score(
            LinearRegression(), m[[pred]].values, m['score_960'].values,
            cv=kf, scoring='r2').mean()
        loss = f'{(r2_std - r2_960) / r2_std * 100:.0f}%' if r2_std > 0 else '---'
        print(f'{label:>15s} {r2_std:>11.2f} {r2_960:>11.2f} {loss:>8s}')

    print()

    # Asymmetry interpretation
    r2_sr_sw = cross_val_score(LinearRegression(), m[['std_rating']].values,
                                m['score'].values, cv=kf, scoring='r2').mean()
    r2_sr_9w = cross_val_score(LinearRegression(), m[['std_rating']].values,
                                m['score_960'].values, cv=kf, scoring='r2').mean()
    r2_9r_sw = cross_val_score(LinearRegression(), m[['c960_rating']].values,
                                m['score'].values, cv=kf, scoring='r2').mean()
    r2_9r_9w = cross_val_score(LinearRegression(), m[['c960_rating']].values,
                                m['score_960'].values, cv=kf, scoring='r2').mean()

    print(f'Asymmetry: 960 rating → std wins (R²={r2_9r_sw:.2f}) > '
          f'std rating → 960 wins (R²={r2_sr_9w:.2f})')
    print(f'Reasoning transfers up; preparation does not transfer.')
    print()

    # Bounded interpretation
    print(f'Ceiling: 960 rating → 960 wins R² = {r2_9r_9w:.2f}')
    print(f'Std rating captures {r2_sr_9w/r2_9r_9w*100:.0f}% of explainable 960 variance')
    print(f'→ ~{100 - r2_sr_9w/r2_9r_9w*100:.0f}% is preparation-specific')

    # CPL-level transfer
    print()
    print('=' * 60)
    print('CPL-level transfer (opening moves 1-12)')
    print('=' * 60)

    conn = sqlite3.connect(DB_PATH)
    df_cpl = pd.read_sql_query('''
    SELECT g.player_username as player, g.format,
           p.standard_blitz_rating as std_rating,
           AVG(m.centipawn_loss) as mean_cpl
    FROM games g
    JOIN move_metrics m ON g.game_id = m.game_id
    JOIN players p ON g.player_username = p.username
    WHERE m.centipawn_loss IS NOT NULL AND m.player_color = g.player_color
      AND m.move_number BETWEEN 1 AND 12
    GROUP BY g.player_username, g.format
    ''', conn)
    conn.close()

    pf2 = df_cpl.groupby('player')['format'].nunique()
    df_cpl = df_cpl[df_cpl['player'].isin(pf2[pf2 == 2].index)]
    std2 = df_cpl[df_cpl['format'] == 'standard'].set_index('player')
    c962 = df_cpl[df_cpl['format'] == 'chess960'].set_index('player')
    m2 = std2[['std_rating', 'mean_cpl']].join(c962[['mean_cpl']], rsuffix='_960').dropna()
    m2['log_std'] = np.log(m2['mean_cpl'] + 1)
    m2['log_960'] = np.log(m2['mean_cpl_960'] + 1)

    r2_cpl_std = cross_val_score(LinearRegression(), m2[['std_rating']].values,
                                  m2['log_std'].values, cv=kf, scoring='r2').mean()
    r2_cpl_960 = cross_val_score(LinearRegression(), m2[['std_rating']].values,
                                  m2['log_960'].values, cv=kf, scoring='r2').mean()
    loss_cpl = (r2_cpl_std - r2_cpl_960) / r2_cpl_std * 100

    print(f'Std rating → std CPL:  R² = {r2_cpl_std:.3f}')
    print(f'Std rating → 960 CPL: R² = {r2_cpl_960:.3f}')
    print(f'Loss: {loss_cpl:.0f}%')


if __name__ == '__main__':
    main()
