"""
decode_robust.py — Enhanced Decoder with Confidence Scoring
============================================================
This builds on decode.py and adds:
  1. Confidence scoring: how sure are we about the detected T_i?
  2. Soft QPSK metrics: which individual symbols are most uncertain?
  3. decode_with_confidence(): returns message + full diagnostics

WHY a separate file?
────────────────────
The baseline decode() in decode.py is clean and production-ready.
This file adds observability on top without modifying the core.
During the demo, you'll call decode() for the actual result,
and decode_with_confidence() during testing to understand failures.
"""

import math
import numpy as np
from decode import (
    r, ALPHABET, PILOT_CANDIDATES, INV_ROTATION,
    to_pairs, detect_rotation, invert_rotation,
    viterbi_decode_soft, bits_to_message
)


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

def rotation_confidence(distances):
    """
    Compute a confidence score for the rotation detection decision.

    WHY do we need this?
    ────────────────────
    The spec says wrong i_hat detection happens with probability ~10⁻⁵.
    That's rare — but across 10,000 test runs you'd expect ~1 garbled message.
    During the live demo you get 2 attempts. If the first attempt returns
    garbage, you want to KNOW it was likely a bad detection (and retry)
    rather than assume the message was just that garbled.

    The confidence gap = (2nd best distance) - (best distance).
    - Large gap → the best candidate is clearly better than the others → high confidence
    - Small gap → two candidates are nearly equidistant → borderline decision → low confidence

    WHY gap and not just best distance?
    ────────────────────────────────────
    The absolute best distance depends on noise magnitude, which varies per run.
    The GAP is relative and captures how decisive the choice was.
    Even in noisy conditions, a large gap means the detection is reliable.

    Parameters
    ----------
    distances : dict {1: float, 2: float, 3: float, 4: float}
        Squared distances from received pilot to each candidate.

    Returns
    -------
    dict with keys:
        'gap'          : float — distance gap between best and 2nd best
        'normalised'   : float in [0,1] — gap / (worst - best)
        'low_conf'     : bool — True if gap is below threshold (suspicious)
        'sorted_dists' : list of (i, dist) sorted best to worst
    """
    sorted_items = sorted(distances.items(), key=lambda x: x[1])
    best_dist   = sorted_items[0][1]
    second_dist = sorted_items[1][1]
    worst_dist  = sorted_items[-1][1]

    gap = second_dist - best_dist

    # Normalise: 0 means completely ambiguous, 1 means maximally confident
    denom = worst_dist - best_dist
    normalised = gap / denom if denom > 1e-9 else 0.0

    # Threshold: empirically, gap < 2.0 is suspicious under unit-variance noise.
    # With 4 pilot pairs, the expected gap under correct detection is ≈ 4r² ≈ 9.9
    # A gap below 2.0 means you're in the tail of the noise distribution.
    LOW_CONF_THRESHOLD = 2.0

    return {
        'gap':          gap,
        'normalised':   normalised,
        'low_conf':     gap < LOW_CONF_THRESHOLD,
        'sorted_dists': sorted_items,
    }


# =============================================================================
# SOFT SYMBOL METRICS
# =============================================================================

def symbol_margins(combined_pairs):
    """
    For each QPSK symbol, compute how far each component is from the
    decision boundary (the zero axis).

    WHY track margins?
    ──────────────────
    A symbol at (0.01, 3.5) has bit 1 decoded almost certainly correctly
    (y2=3.5 is far from boundary) but bit 1 decoded with high uncertainty
    (y1=0.01 is right on the boundary).

    The margin |y_k| tells you: "how much noise would it take to flip this bit?"
    - Large margin → robust decision
    - Small margin → near the decision boundary → more likely to be wrong

    This is useful for:
    - Identifying which characters are most likely wrong after decoding
    - Debugging: if many symbols have small margins, your SNR estimate is off

    Parameters
    ----------
    combined_pairs : list of (float, float)

    Returns
    -------
    list of dict, one per symbol, with:
        'b1_margin' : abs(y1) — margin for first bit
        'b2_margin' : abs(y2) — margin for second bit
        'min_margin': min of both — overall symbol reliability
        'b1'        : decoded bit 1
        'b2'        : decoded bit 2
    """
    result = []
    for (y1, y2) in combined_pairs:
        result.append({
            'b1_margin':  abs(y1),
            'b2_margin':  abs(y2),
            'min_margin': min(abs(y1), abs(y2)),
            'b1':         0 if y1 >= 0 else 1,
            'b2':         0 if y2 >= 0 else 1,
        })
    return result


def weakest_symbols(margins, n=10):
    """
    Return the n symbols closest to the decision boundary (most likely wrong).

    Parameters
    ----------
    margins : list of dict from symbol_margins()
    n       : int — how many to return

    Returns
    -------
    list of (symbol_index, min_margin) sorted weakest first
    """
    indexed = [(i, m['min_margin']) for i, m in enumerate(margins)]
    return sorted(indexed, key=lambda x: x[1])[:n]


# =============================================================================
# ENHANCED DECODE FUNCTION
# =============================================================================

def decode_with_confidence(y):
    """
    Full decode pipeline with diagnostics at every stage.

    Returns a dict with:
        'message'     : str — the decoded 40-char message
        'i_hat'       : int — detected rotation (1-4)
        'confidence'  : dict — rotation detection confidence metrics
        'symbol_info' : list — per-symbol margin data
        'weakest'     : list — 10 symbols closest to decision boundary
        'warning'     : str or None — human-readable warning if something is suspicious

    Use this during testing, not during the live demo
    (the plain decode() is faster and cleaner for the demo itself).
    """
    y = list(y)
    if len(y) != 500:
        raise ValueError(f"Expected y of length 500, got {len(y)}")

    warning = None

    # Step 1: pilots (16 values = 8 pairs)
    y_pilot = np.array(y[0:16])

    # Step 2: rotation detection + confidence
    i_hat, distances = detect_rotation(y_pilot)
    conf = rotation_confidence(distances)

    if conf['low_conf']:
        warning = (
            f"⚠ LOW CONFIDENCE rotation detection: gap={conf['gap']:.3f} "
            f"(threshold=2.0). Detected T{i_hat} but this may be wrong. "
            f"Message may be garbled."
        )

    # Step 3: invert rotation on all 242 data pairs
    data_pairs = to_pairs(y[16:500])
    corrected_pairs = invert_rotation(data_pairs, i_hat)

    # Step 4: per-trellis-step soft margins (min of |y1|, |y2|) — a coarse
    # proxy for which trellis steps were noisiest. The Viterbi metric itself
    # is what actually drives the decision; these margins are just diagnostics.
    margins = symbol_margins(corrected_pairs)

    # Step 5: soft-decision Viterbi → 240 info bits
    info_bits = viterbi_decode_soft(corrected_pairs, n_info_bits=240)

    # Step 6: bits to message
    message = bits_to_message(info_bits)

    # Weakest trellis steps — each step's 2 coded bits cover (on average) 1/3
    # of an info bit (rate-1/2, but Viterbi spreads decisions over the whole
    # trellis). Map step index back to approximate character position via the
    # 2 coded bits per step → 1 info bit per step → 6 info bits per character.
    weak = weakest_symbols(margins, n=10)
    weak_chars = []
    for step_idx, margin in weak:
        info_bit_idx = step_idx                          # ~1 info bit per step
        char_idx = min(info_bit_idx // 6, len(message) - 1)
        weak_chars.append({
            'symbol_idx': step_idx,
            'char_idx':   char_idx,
            'char':       message[char_idx],
            'margin':     margin,
        })

    return {
        'message':     message,
        'i_hat':       i_hat,
        'confidence':  conf,
        'distances':   distances,
        'symbol_info': margins,
        'weakest':     weak_chars,
        'warning':     warning,
    }
