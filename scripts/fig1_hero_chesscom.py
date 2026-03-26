#!/usr/bin/env python3
"""
Hero figure for NHB — Chess.com primary version.

Panel (a): Dose-response — 11 matched GMs (same as before, OTB + Chess.com)
Panel (b): Density gradient — Chess.com titled players (player-level dots + LOWESS)
Panel (c): Time allocation ratio — Chess.com titled players (paired slope plot)
"""

import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

PROJECT = Path('/Users/chris/_claude-cowork/chess960')
PAPER = Path('/Users/chris/_claude-cowork/chess960Paper/nhb')
OUTPUT_DIR = PAPER / 'figures'
DB_CHESSCOM = PROJECT / 'db' / 'chesscom.db'
OTB_JSON = PROJECT / 'data' / 'otb_analysis_sf18.json'

sys.path.insert(0, str(PROJECT))
from analysis.template_distance import compute_sp_features

CHESSCOM_GMS = [
    'MagnusCarlsen', 'Hikaru', 'FabianoCaruana', 'alireza2003',
    'keymer_vincent', 'LevonAronian', 'Javokhir_Sindarov',
    'FrederikSvane', 'D_Dardha', 'Daniel-Oparin', 'RayRobson',
]
OTB_NAMES = [
    'Carlsen, Magnus', 'Nakamura, Hikaru', 'Caruana, Fabiano',
    'Firouzja, Alireza', 'Keymer, Vincent', 'Aronian, Levon',
    'Sindarov, Javokhir', 'Svane, Frederik', 'Dardha, Daniel',
    'Oparin, Grigoriy', 'Robson, Ray',
]

COL = {
    'otb': '#2c3e50', 'otb_light': '#bdc3c7',
    'cc': '#8e44ad', 'cc_light': '#d7bde2',
    'std': '#2166AC', 'c960': '#B2182B', 'gap': '#B2182B',
}


def set_pub_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'lines.linewidth': 1.0, 'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    })


# ══════════════════════════════════════════════════════════════
# Panel (a): Dose-response — 11 matched GMs (identical to v13)
# ══════════════════════════════════════════════════════════════

def fen_to_td(fen):
    if fen is None: return None
    r8 = fen.split('/')[0].upper()
    exp = ''
    for c in r8: exp += '.'*int(c) if c.isdigit() else c
    if len(exp) != 8: return None
    std = 'RNBQKBNR'; d = 0
    for p in set(std):
        sp = [i for i,c in enumerate(std) if c==p]
        fp = [i for i,c in enumerate(exp) if c==p]
        if len(sp)!=len(fp): return None
        if len(sp)<=2:
            d += min(sum(abs(s-f) for s,f in zip(sp,pm)) for pm in permutations(fp))
        else:
            d += sum(abs(s-f) for s,f in zip(sorted(sp),sorted(fp)))
    return d


def load_otb():
    with open(OTB_JSON) as f: data = json.load(f)
    rows = []
    skip_tc = {'10+5','10+10','5+2','15+10','3+0','3+1'}
    for g in data['games']:
        if g.get('time_control','') in skip_tc: continue
        for color in ['white','black']:
            name = g.get(color,'')
            if not any(o in name or name in o for o in OTB_NAMES): continue
            pm = [m for m in g.get('moves',[]) if m.get('move_number',99)<=12
                  and m.get('player_color','')==color and m.get('eval_after') is not None]
            if len(pm)<3: continue
            cpls = [max(0,m['eval_before']-m['eval_after']) if color=='white'
                    else max(0,m['eval_after']-m['eval_before'])
                    for m in pm if m.get('eval_before') is not None and m.get('eval_after') is not None]
            if len(cpls)<3: continue
            fmt = g.get('format','unknown')
            td = -2 if fmt!='chess960' else (fen_to_td(g.get('fen','')) or 10)
            rows.append({'player':name,'format':fmt,'mean_cpl':np.mean(cpls),'td':td})
    return pd.DataFrame(rows)


def load_cc_gm():
    conn = sqlite3.connect(DB_CHESSCOM)
    gml = [n.lower() for n in CHESSCOM_GMS]
    ph = ','.join(['?']*len(gml))
    q = f"""SELECT g.player_username as player, g.format, g.chess960_position as sp,
           AVG(m.centipawn_loss) as mean_cpl, COUNT(*) as n
    FROM games g JOIN move_metrics m ON g.game_id = m.game_id
    WHERE LOWER(g.player_username) IN ({ph}) AND m.centipawn_loss IS NOT NULL
      AND m.move_number BETWEEN 1 AND 12 AND m.player_color = g.player_color
    GROUP BY g.game_id HAVING COUNT(*)>=5"""
    df = pd.read_sql_query(q, conn, params=gml); conn.close()
    if len(df)==0:
        conn = sqlite3.connect(DB_CHESSCOM)
        df = pd.read_sql_query("""SELECT g.player_username as player, g.format, g.chess960_position as sp,
               AVG(m.centipawn_loss) as mean_cpl, COUNT(*) as n
        FROM games g JOIN move_metrics m ON g.game_id = m.game_id
        WHERE m.centipawn_loss IS NOT NULL AND m.move_number BETWEEN 1 AND 12 AND m.player_color = g.player_color
        GROUP BY g.game_id HAVING COUNT(*)>=5""", conn); conn.close()
        df = df[df['player'].str.lower().isin(gml)]
    spf = compute_sp_features()
    spf['fd'] = spf['delta_rooks']+spf['delta_bishops']+spf['delta_knights']+spf['delta_king']+spf['delta_queen']
    sd = dict(zip(spf['sp'], spf['fd']))
    df['td'] = df.apply(lambda r: sd.get(r['sp'],10) if r['format']=='chess960' else -2, axis=1)
    return df[['player','format','mean_cpl','td']]


def panel_a(ax):
    otb = load_otb(); cc = load_cc_gm()
    print(f"Panel a: OTB={len(otb)}, CC={len(cc)}")

    # Bins centered on integers where data clusters
    bins = [(1,5),(5,8),(8,11),(11,14),(14,17),(17,21)]
    bcs = [3, 6, 9, 12, 15, 19]

    # OTB more visible (fewer points), CC fainter (many points)
    for df,cm,cl,mk,lab,off,pt_alpha,pt_size in [
        (otb,COL['otb'],COL['otb_light'],'s',f'OTB classical (N={len(otb)})',-0.15, 0.35, 12),
        (cc,COL['cc'],COL['cc_light'],'o',f'Chess.com blitz (N={len(cc)})',0.15, 0.12, 6)]:
        if len(df)==0: continue
        c9 = df[df.format=='chess960']; st = df[df.format=='standard']
        # Standard game points at x=-2 (no mean dot — scatter speaks for itself)
        if len(st)>0:
            ax.scatter(np.full(len(st), -2) + off + np.random.normal(0,0.2,len(st)),
                       st.mean_cpl, alpha=pt_alpha, s=pt_size, color=cm, marker='o', zorder=1,
                       linewidths=0, label=lab)
        # Chess960 game points — filled dots in dataset color
        ax.scatter(c9.td+np.random.normal(0,0.3,len(c9)), c9.mean_cpl,
                   alpha=pt_alpha, s=pt_size, color=cm, marker='o', zorder=1,
                   linewidths=0)
        # Trend line fitted through individual game points — start at first bin
        if len(c9)>=10:
            z=np.polyfit(c9.td, c9.mean_cpl, 1); xl=np.linspace(bcs[0],20,50)
            ax.plot(xl,np.polyval(z,xl),'--',color=cm,alpha=.6,linewidth=1.0,zorder=2)

    # Offset brackets — arrow top is the fit line's value at the first bin
    # Store fit coefficients per dataset (from scatter, not bin means)
    fit_coeffs = {}
    for df,cm,cl,mk,lab,off in [
        (otb,COL['otb'],COL['otb_light'],'s','OTB',-0.15),
        (cc,COL['cc'],COL['cc_light'],'o','CC',0.15)]:
        if len(df)==0: continue
        c9 = df[df.format=='chess960']
        if len(c9)>=10:
            fit_coeffs[lab] = np.polyfit(c9.td, c9.mean_cpl, 1)

    for df,color,off,mk_label,bx in [(otb,COL['otb'],-0.15,'OTB',1.0),
                                      (cc,COL['cc'],0.15,'CC',1.5)]:
        if len(df)==0 or mk_label not in fit_coeffs: continue
        st = df[df.format=='standard']
        if len(st)==0: continue
        std_mean = st.mean_cpl.mean()
        # Use fit line prediction at first bin center for arrow top
        z = fit_coeffs[mk_label]
        c9_predicted = np.polyval(z, bcs[0])  # predicted at first 960 bin
        ov = c9_predicted - std_mean
        # Horizontal dashed line from standard point to bracket
        ax.plot([-2+off, bx], [std_mean, std_mean], '--', color=color,
                alpha=0.4, linewidth=0.8, zorder=2)
        # Vertical offset arrow (bottom = baseline, top = fit line)
        ax.annotate('',xy=(bx,std_mean),xytext=(bx,c9_predicted),
                    arrowprops=dict(arrowstyle='<->',color=color,lw=1.5,shrinkA=0,shrinkB=0))
        ax.text(bx-0.6,(std_mean+c9_predicted)/2,f'+{ov:.0f}',fontsize=9,color=color,
                fontweight='bold',ha='center',va='center')
        print(f"  {mk_label} offset: std={std_mean:.1f}, fit@d={bcs[0]}={c9_predicted:.1f}, +{ov:.0f}")

    # Annotation arrows — both OTB and CC, aligned with fit line left endpoints
    # Compute offsets and gradients for both datasets
    arrow_data = {}
    for label, df_data, color in [('OTB', otb, COL['otb']), ('CC', cc, COL['cc'])]:
        if label not in fit_coeffs or len(df_data) == 0: continue
        z = fit_coeffs[label]
        st = df_data[df_data.format == 'standard']
        if len(st) == 0: continue
        std_mean = st.mean_cpl.mean()
        y_at_first = np.polyval(z, bcs[0])
        y_at_last = np.polyval(z, bcs[-1])
        arrow_data[label] = {
            'std_mean': std_mean,
            'y_first': y_at_first,
            'y_last': y_at_last,
            'offset': y_at_first - std_mean,
            'gradient': y_at_last - y_at_first,
            'color': color,
        }

    # Arrow 1: Standard → first 960 bin (OFFSET)
    arrow_y = -1.5
    ax.annotate('', xy=(bcs[0]-0.3, arrow_y), xytext=(-1.5, arrow_y),
                arrowprops=dict(arrowstyle='-|>', color='#333', lw=2.0,
                               mutation_scale=12))
    # Label with both datasets
    offset_parts = []
    for label in ['OTB', 'CC']:
        if label in arrow_data:
            d = arrow_data[label]
            offset_parts.append(f'{label}: +{d["offset"]:.0f}')
    ax.text((-1.5+bcs[0]-0.3)/2, arrow_y+1.2, '\n'.join(offset_parts),
            fontsize=7, color='#333', ha='center', va='bottom',
            fontweight='bold', linespacing=1.3)

    # Arrow 2: first 960 bin → last bin (GRADIENT)
    ax.annotate('', xy=(bcs[-1]+0.3, arrow_y), xytext=(bcs[0]+0.3, arrow_y),
                arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.2,
                               mutation_scale=10, alpha=0.5))
    grad_parts = []
    for label in ['OTB', 'CC']:
        if label in arrow_data:
            d = arrow_data[label]
            grad_parts.append(f'{label}: +{d["gradient"]:.0f}')
    ax.text((bcs[0]+bcs[-1])/2, arrow_y+1.2, '\n'.join(grad_parts),
            fontsize=7, color='#333', ha='center', va='bottom',
            fontstyle='italic', alpha=0.7, linespacing=1.3)

    for label in ['OTB', 'CC']:
        if label in arrow_data:
            d = arrow_data[label]
            print(f"  {label} offset: +{d['offset']:.0f} CPL, gradient: +{d['gradient']:.0f} CPL")

    ax.set_xlabel('Template distance from standard chess', labelpad=22)
    ax.set_ylabel('Mean CPL (moves 1\u201312)')
    ax.set_xlim(-3.5, 20.5); ax.set_ylim(-3, 35)
    # X-ticks: "Standard" at -2, then bin centers
    # X-tick labels show bin ranges
    bin_labels = [f'{lo}\u2013{hi-1}' for lo,hi in bins]
    ax.set_xticks([-2] + bcs)
    ax.set_xticklabels(['Std'] + bin_labels, fontsize=5.5, rotation=0)
    ax.legend(frameon=False, loc='upper left', fontsize=6.5, markerscale=0.9)
    ax.text(-0.06, 1.05, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

    # Back-rank vignettes at selected distances
    # Standard: RNBQKBNR, displaced pieces in red
    standard = list('RNBQKBNR')
    piece_to_sym = {'R': '\u265C', 'N': '\u265E', 'B': '\u265D',
                    'Q': '\u265B', 'K': '\u265A'}
    vignettes = [
        (-2,  'RNBQKBNR'),   # d=0: standard
        (3,   'NRBQKBNR'),   # d~3
        (6,   'NRBQNBKR'),   # d~6
        (9,   'RNKQBBNR'),   # d~9
        (12,  'QNBRNBKR'),   # d~12
        (15,  'BQNNRBKR'),   # d~15
        (19,  'BBQNNRKR'),   # d~18
    ]
    y_vig = -7.5

    for x_pos, arrangement in vignettes:
        arr = list(arrangement)
        for j, (piece, std_piece) in enumerate(zip(arr, standard)):
            sym = piece_to_sym[piece]
            color = '#B2182B' if piece != std_piece else '#555555'
            ax.text(x_pos + (j - 3.5) * 0.25, y_vig, sym,
                    fontsize=7.5, ha='center', va='center', color=color,
                    fontfamily='DejaVu Sans', clip_on=False)


# ══════════════════════════════════════════════════════════════
# Chess.com move-level data loader (cached)
# ══════════════════════════════════════════════════════════════

_cache = None

def load_chesscom_moves():
    global _cache
    if _cache is not None: return _cache

    conn = sqlite3.connect(DB_CHESSCOM)
    q = """
    SELECT g.player_username as player, g.format, g.player_color,
           m.move_number, m.centipawn_loss, m.time_spent,
           pc.num_legal_moves, pc.eval_pv1, pc.eval_pv2, pc.eval_pv3, pc.eval_pv4, pc.eval_pv5,
           pc.position_fen
    FROM games g
    JOIN move_metrics m ON g.game_id = m.game_id
    LEFT JOIN position_complexity pc ON g.game_id = pc.game_id
        AND m.move_number = pc.move_number AND m.player_color = pc.player_color
    WHERE m.centipawn_loss IS NOT NULL
      AND m.player_color = g.player_color
      AND m.move_number BETWEEN 1 AND 12
      AND m.time_spent IS NOT NULL AND m.time_spent > 0
    """
    df = pd.read_sql_query(q, conn); conn.close()

    # Paired players only
    pf = df.groupby('player')['format'].nunique()
    df = df[df['player'].isin(pf[pf==2].index)].copy()

    df['is_960'] = (df['format']=='chess960').astype(int)
    df['error'] = (df['centipawn_loss']>25).astype(int)
    df['log_time'] = np.log(df['time_spent'].clip(lower=0.1))
    df['color_binary'] = (df['player_color']=='white').astype(int)

    # Density
    pv_cols = ['eval_pv1','eval_pv2','eval_pv3','eval_pv4','eval_pv5']
    for c in pv_cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    best = df['eval_pv1'].values
    ng = np.zeros(len(df)); v = np.isfinite(best)
    for c in pv_cols:
        vals = df[c].values; ng += v & np.isfinite(vals) & (np.abs(vals-best)<=50)
    ng[~v] = np.nan; df['n_good'] = ng
    df['density'] = np.where((df['num_legal_moves']>0)&np.isfinite(ng),
                             ng/df['num_legal_moves'], np.nan)

    print(f"Loaded Chess.com: {len(df):,} moves, {df['player'].nunique()} paired players")
    print(f"  Standard: {(df['format']=='standard').sum():,}")
    print(f"  Chess960: {(df['format']=='chess960').sum():,}")
    print(f"  With density: {df['density'].notna().sum():,}")

    _cache = df
    return df


# ══════════════════════════════════════════════════════════════
# Panel (b): Density gradient — Chess.com titled players
# ══════════════════════════════════════════════════════════════

def panel_b(ax):
    """Decile dot plot with player-level jitter behind each decile point."""
    df = load_chesscom_moves()
    df_d = df[df['density'].notna()].copy()
    print(f"Panel b: {len(df_d):,} moves with density")

    # Decile bins
    df_d['decile'] = pd.qcut(df_d['density'], q=10, labels=False, duplicates='drop')

    # Player-level gaps per decile
    player_gaps_all = []
    decile_summaries = []
    for dec in sorted(df_d['decile'].unique()):
        grp = df_d[df_d['decile'] == dec]
        # Overall decile gap
        std_all = grp[grp.format == 'standard']
        c9_all = grp[grp.format == 'chess960']
        if len(std_all) < 50 or len(c9_all) < 50:
            continue
        gap_overall = (c9_all['error'].mean() - std_all['error'].mean()) * 100
        se = np.sqrt(std_all['error'].sem()**2 + c9_all['error'].sem()**2) * 100 * 1.96
        decile_summaries.append({'decile': dec + 1, 'gap': gap_overall, 'ci': se})

        # Per-player gaps within this decile
        for player in grp['player'].unique():
            std_p = grp[(grp['player'] == player) & (grp['format'] == 'standard')]
            c9_p = grp[(grp['player'] == player) & (grp['format'] == 'chess960')]
            if len(std_p) >= 5 and len(c9_p) >= 5:
                pg = (c9_p['error'].mean() - std_p['error'].mean()) * 100
                player_gaps_all.append({'decile': dec + 1, 'gap': pg})

    gdf = pd.DataFrame(decile_summaries)
    pgdf = pd.DataFrame(player_gaps_all)

    # Player-level jittered dots (faded, behind)
    jitter = np.random.normal(0, 0.15, len(pgdf))
    ax.scatter(pgdf['gap'], pgdf['decile'] + jitter,
               s=3, alpha=0.12, color=COL['gap'], edgecolors='none', zorder=1)

    # Decile summary dots (no whiskers — player dots show the spread)
    ax.scatter(gdf['gap'], gdf['decile'], s=50, color=COL['gap'],
               edgecolors='white', linewidths=0.5, zorder=3)

    # Connect with a line
    ax.plot(gdf['gap'], gdf['decile'], '-', color=COL['gap'], alpha=0.4, linewidth=1, zorder=2)

    ax.set_xlabel('Format gap in error rate (pp)')
    ax.set_ylabel('Position difficulty decile')
    ax.set_yticks(range(1, 11))
    ax.set_yticklabels(['1\n(hardest)', '2', '3', '4', '5', '6', '7', '8', '9', '10\n(easiest)'],
                        fontsize=6)
    ax.set_xlim(-15, 45)
    ax.invert_yaxis()

    # N annotation
    ax.text(0.97, 0.03, f'N = {len(pgdf):,} player\u00d7deciles',
            transform=ax.transAxes, fontsize=6, ha='right', va='bottom', color='#777')

    ax.text(-0.14, 1.05, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

    print(f"  Gap range: {gdf['gap'].min():.1f} to {gdf['gap'].max():.1f} pp")
    print(f"  Player-decile observations: {len(pgdf):,}")


# ══════════════════════════════════════════════════════════════
# Panel (c): Time allocation ratio — Chess.com titled players
# ══════════════════════════════════════════════════════════════

def panel_c(ax):
    df = load_chesscom_moves()
    df_d = df[df['density'].notna()].copy()

    cuts = df_d['density'].quantile([1/3,2/3]).values
    df_d['difficulty'] = pd.cut(df_d['density'],
        bins=[-np.inf,cuts[0],cuts[1],np.inf], labels=['Hard','Medium','Easy'])

    paired = []
    for player in df_d['player'].unique():
        row = {'player': player}
        for fmt in ['standard','chess960']:
            h = df_d[(df_d.player==player)&(df_d.format==fmt)&(df_d.difficulty=='Hard')]
            e = df_d[(df_d.player==player)&(df_d.format==fmt)&(df_d.difficulty=='Easy')]
            if len(h)>=5 and len(e)>=5:
                row[fmt] = h['time_spent'].mean() / e['time_spent'].mean()
        if 'standard' in row and 'chess960' in row:
            paired.append(row)
    pdf = pd.DataFrame(paired)
    print(f"Panel c: {len(pdf)} paired players")

    # Slope lines (subsample)
    show = pdf.sample(min(40,len(pdf)), random_state=42) if len(pdf)>40 else pdf
    for _, row in show.iterrows():
        down = row['chess960'] < row['standard']
        ax.plot([0,1],[row['standard'],row['chess960']],
                color=COL['c960'] if down else COL['std'], alpha=0.06, linewidth=0.4, zorder=1)

    # All dots
    for x,fmt,color in [(0,'standard',COL['std']),(1,'chess960',COL['c960'])]:
        jit = np.random.normal(0,0.025,len(pdf))
        ax.scatter(np.full(len(pdf),x)+jit, pdf[fmt],
                   s=4, alpha=0.15, color=color, edgecolors='none', zorder=2)

    # Median dot + IQR
    for x,fmt,color in [(0,'standard',COL['std']),(1,'chess960',COL['c960'])]:
        med = pdf[fmt].median()
        q25,q75 = pdf[fmt].quantile(.25), pdf[fmt].quantile(.75)
        ax.plot([x,x],[q25,q75],color=color,linewidth=2.5,zorder=6,solid_capstyle='round')
        ax.plot(x,med,'o',color=color,markersize=9,zorder=7,markeredgecolor='white',markeredgewidth=0.5)
        ax.text(x,q75+0.1,f'{med:.2f}',ha='center',fontsize=8.5,color=color,fontweight='bold',zorder=8,
                bbox=dict(facecolor='white',edgecolor='none',alpha=0.8,pad=1))

    pct = (pdf['chess960']<pdf['standard']).mean()*100
    ax.text(0.5,0.04,f'{pct:.0f}% of players decrease',ha='center',va='bottom',fontsize=6.5,color='#555',
            transform=ax.transAxes,fontstyle='italic')
    ax.axhline(1,color='#999',linewidth=1.0,linestyle='--',zorder=0)
    ax.text(0.97,0.97,f'N = {len(pdf)}',ha='right',va='top',fontsize=6.5,color='#777',transform=ax.transAxes)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Standard','Chess960'],fontsize=8)
    ax.set_ylabel('Time ratio (Hard / Easy)')
    ax.set_xlim(-0.4,1.4); ax.set_ylim(0.3,3.0)

    t,p = stats.ttest_rel(pdf['standard'],pdf['chess960'])
    print(f"  Medians: {pdf['standard'].median():.2f} vs {pdf['chess960'].median():.2f}, t={t:.1f}, p={p:.1e}, {pct:.0f}%")
    ax.text(-0.16,1.05,'c',transform=ax.transAxes,fontsize=12,fontweight='bold',va='bottom')


def main():
    set_pub_style()
    fig = plt.figure(figsize=(180/25.4, 150/25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2,1], width_ratios=[1.1,0.7], hspace=0.45, wspace=0.4)
    ax_a = fig.add_subplot(gs[0,:])
    ax_b = fig.add_subplot(gs[1,0])
    ax_c = fig.add_subplot(gs[1,1])

    print("Panel a: dose-response (OTB + Chess.com GMs)...")
    panel_a(ax_a)
    print("\nPanel b: density gradient (Chess.com titled)...")
    panel_b(ax_b)
    print("\nPanel c: time allocation (Chess.com titled)...")
    panel_c(ax_c)

    for ext in ['pdf','png']:
        fig.savefig(OUTPUT_DIR / f'fig1_hero_chesscom.{ext}')
    print(f"\nSaved fig1_hero_chesscom.pdf/png")
    plt.close()


if __name__ == '__main__':
    main()
