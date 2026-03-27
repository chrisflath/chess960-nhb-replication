"""Template distance computation for Chess960 starting positions."""

def decode_sp(n):
    """Decode Chess960 SP number to piece positions using Scharnagl scheme."""
    n2, bl = divmod(n, 4)
    n3, bd = divmod(n2, 4)
    n4, qi = divmod(n3, 6)
    b1 = [1, 3, 5, 7][bl]
    b2 = [0, 2, 4, 6][bd]
    rem = [i for i in range(8) if i not in (b1, b2)]
    q = rem[qi]
    rem2 = [i for i in range(8) if i not in (b1, b2, q)]
    kt = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
          (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    n1i, n2i = rem2[kt[n4][0]], rem2[kt[n4][1]]
    rem3 = [i for i in range(8) if i not in (b1, b2, q, n1i, n2i)]
    return {
        "rooks": sorted([rem3[0], rem3[2]]),
        "knights": sorted([n1i, n2i]),
        "bishops": sorted([b2, b1]),
        "queen": q,
        "king": rem3[1],
    }


STANDARD = decode_sp(518)

def compute_sp_features():
    """
    Compute template-distance features for all 960 starting positions.

    Returns DataFrame with columns:
        sp, arrangement,
        # Piece displacement (5)
        delta_rooks, delta_bishops, delta_knights, delta_king, delta_queen,
        # Structural vulnerability (2)
        kq_same_color, n_unprotected,
    """
    rows = []
    for sp in range(960):
        p = decode_sp(sp)

        # Back-rank array for adjacency / vulnerability checks
        arr = [None] * 8
        for pt, files in p.items():
            if isinstance(files, list):
                for f in files:
                    arr[f] = pt
            else:
                arr[files] = pt

        arrangement = ""
        for a in arr:
            arrangement += "N" if a == "knights" else a[0].upper()

        # ── 1. Piece displacement (Manhattan distance from standard file) ──
        delta_rooks = sum(abs(a - b) for a, b in zip(p["rooks"], STANDARD["rooks"]))
        delta_bishops = sum(abs(a - b) for a, b in zip(p["bishops"], STANDARD["bishops"]))
        delta_knights = sum(abs(a - b) for a, b in zip(p["knights"], STANDARD["knights"]))
        delta_king = abs(p["king"] - STANDARD["king"])
        delta_queen = abs(p["queen"] - STANDARD["queen"])

        # ── 2. Structural vulnerability ──
        # K-Q same square color → diagonal fork risk
        kq_same_color = int(p["king"] % 2 == p["queen"] % 2)

        # Unprotected pawns (no vertical defender behind nor diagonal defender adjacent)
        n_unprotected = 0
        for f in range(8):
            vertical = arr[f] in VERTICAL_DEFENDERS
            diag_left = f > 0 and arr[f - 1] in DIAGONAL_DEFENDERS
            diag_right = f < 7 and arr[f + 1] in DIAGONAL_DEFENDERS
            if not (vertical or diag_left or diag_right):
                n_unprotected += 1

        rows.append({
            "sp": sp,
            "arrangement": arrangement,
            "delta_rooks": delta_rooks,
            "delta_bishops": delta_bishops,
            "delta_knights": delta_knights,
            "delta_king": delta_king,
            "delta_queen": delta_queen,
            "kq_same_color": kq_same_color,
            "n_unprotected": n_unprotected,
        })

    return pd.DataFrame(rows)
