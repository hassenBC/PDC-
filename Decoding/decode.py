"""
decode.py — Core Decoder
========================
Person 3 deliverable: decode(y) -> message

This file contains the main decode function and every building block it uses.
Every single design choice is explained in comments so you can defend each one
at the presentation.

SYSTEM OVERVIEW
---------------
The channel does this:   Y = T_i(x) + Z
Your job reverses it:    Y  →  T_i  →  x  →  bits  →  characters  →  message

The transmitted vector x has this structure (500 elements total):
  x[0:16]    = 8 pilot pairs (16 values, 8 non-zero), used to detect T_i
  x[16:500]  = 484 BPSK channel uses carrying a K=3 (7,5) rate-1/2 convolutional
               code over (240 info bits + 2 termination bits) → 484 coded bits
               Each pair (x[2k], x[2k+1]) is one trellis step's (c1, c2) outputs.

WHY a convolutional code instead of 2× repetition?
──────────────────────────────────────────────────
The previous design (rate-1/2 repetition: same 240-bit message sent twice) has
minimum distance d=2 and gives post-combine BER ≈ Q(s√2) ≈ 1.3%, so
P(perfect 240-bit decode) ≈ (1-0.013)^240 ≈ 4%.

A K=3 convolutional code with generators (7,5) octal has free distance d_free=5.
Soft-decision Viterbi at Eb/N0 ≈ 3.87 dB gives BER ≈ Q(√(d_free·2·R·Eb/N0))
                                                  = Q(√12.2) ≈ 2.4×10⁻⁴
which pushes P(perfect 240-bit decode) ≈ (1 - 2.4×10⁻⁴)^240 ≈ 94%.
Same channel uses, same energy budget — purely an algorithmic upgrade.
"""

import math
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

# WHY r = sqrt(1200/492)?
# ───────────────────────
# The problem enforces a total energy constraint: ‖x‖² ≤ 1200.
#   Pilots:  8 pairs at (r, 0)  → 16 elements,  8 non-zero
#   Data:    484 BPSK channel uses → 484 non-zero
#   Total:   500 elements,  492 non-zero
#
# Equal energy per non-zero component (the budget-optimal allocation when the
# pilot detection error rate is already low):
#   ‖x‖² = 492 · r² = 1200  →  r² = 1200/492 ≈ 2.439,  r ≈ 1.562
#
# WHY 8 pilot pairs?
# ──────────────────
# Inter-candidate distance is d² = 16·r² ≈ 39.0, so P(wrong T_i adjacent) =
# Q(d/2) = Q(2r) ≈ Q(3.12) ≈ 0.09%. With 4 pairs it would be Q(r√2) ≈ 1.3%,
# i.e. ~15× more catastrophic failures. 8 pairs is the sweet spot before
# the marginal data-energy cost dominates.

r = math.sqrt(1200 / 492)   # ≈ 1.562

# WHY this specific ALPHABET?
# ───────────────────────────
# The project specifies 64 characters: a-z (26) + A-Z (26) + 0-9 (10) + ' ' + '.'
# That's exactly 2^6 = 64, so each character maps to a 6-bit integer.
# The ORDER matters and must exactly match Person 2's encoder.
# If even one character is in the wrong position, every character with that
# 6-bit index will be decoded wrong.
# Verify with Person 2 that their ALPHABET list uses this exact same string.

ALPHABET = list(
    'abcdefghijklmnopqrstuvwxyz'   # indices  0-25
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'   # indices 26-51
    '0123456789'                   # indices 52-61
    ' .'                           # indices 62-63
)
# Quick sanity check at import time
assert len(ALPHABET) == 64, f"ALPHABET length is {len(ALPHABET)}, expected 64"
assert len(set(ALPHABET)) == 64, "ALPHABET contains duplicate characters"


# =============================================================================
# THE FOUR ROTATION CANDIDATES (pilot vectors without noise)
# =============================================================================

# WHY pilots at all?
# ──────────────────
# Problem 3(a) in the theory sheet PROVES that if you use a code where -c is
# also a codeword (which QPSK does — every symbol has an antipodal pair),
# the error probability is AT LEAST 1/2 if you don't know T_i.
# That means without pilots you can't do better than random guessing.
# The pilots solve this completely: 4 known test vectors let you identify T_i
# before you touch the data.
#
# WHY (r, 0) as the pilot symbol?
# ────────────────────────────────
# When T_i is applied to (r, 0):
#   T1: (r, 0) → (r,  0)   same, real axis
#   T2: (r, 0) → (0,  r)   rotated 90°, imaginary axis
#   T3: (r, 0) → (-r, 0)   rotated 180°, negative real axis
#   T4: (r, 0) → (0, -r)   rotated 270°, negative imaginary axis
#
# These 4 outcomes are maximally separated — they point in 4 orthogonal
# directions. This gives the largest possible noise margin for detection.

# Build the 4 candidate pilot vectors (16 values = 8 repeated pairs)
# Each is what the receiver WOULD see if that T_i was applied (noiseless)
PILOT_CANDIDATES = {
    1: np.array([ r,  0,  r,  0,  r,  0,  r,  0,  r,  0,  r,  0,  r,  0,  r,  0]),  # T1: all real
    2: np.array([ 0,  r,  0,  r,  0,  r,  0,  r,  0,  r,  0,  r,  0,  r,  0,  r]),  # T2: all imaginary
    3: np.array([-r,  0, -r,  0, -r,  0, -r,  0, -r,  0, -r,  0, -r,  0, -r,  0]),  # T3: all negative real
    4: np.array([ 0, -r,  0, -r,  0, -r,  0, -r,  0, -r,  0, -r,  0, -r,  0, -r]),  # T4: all negative imaginary
}

# WHY are these candidates orthogonal?
# cand_1 · cand_2 = 0  (different positions are non-zero)
# cand_1 · cand_3 = -4r²  (antipodal — maximum separation possible)
# The 4 candidates partition ℝ⁸ into 4 equal Voronoi regions with large gaps.
# The minimum distance between any two candidates is ‖cand_1 - cand_2‖ = 2r√4 ≈ 6.3
# With noise std=1, this gives a very small detection error probability.


# =============================================================================
# INVERSE ROTATION MATRICES
# =============================================================================

# WHY 2×2 matrices?
# ─────────────────
# T_i acts on 2D pairs (a, b) as a linear map, so it has a 2×2 matrix.
# The inverse of a rotation is another rotation in the opposite direction.
# In complex number terms:
#   T1 multiplied by 1   → inverse: multiply by  1   → (a, b) → ( a,  b)
#   T2 multiplied by j   → inverse: multiply by -j   → (a, b) → ( b, -a)
#   T3 multiplied by -1  → inverse: multiply by -1   → (a, b) → (-a, -b)
#   T4 multiplied by -j  → inverse: multiply by  j   → (a, b) → (-b,  a)
#
# Each 2×2 matrix is stored as (m00, m01, m10, m11) so that:
#   new_a = m00*a + m01*b
#   new_b = m10*a + m11*b

INV_ROTATION = {
    1: ( 1,  0,  0,  1),   # identity
    2: ( 0,  1, -1,  0),   # multiply by -j:  (a,b) → (b, -a)
    3: (-1,  0,  0, -1),   # multiply by -1:  (a,b) → (-a, -b)
    4: ( 0, -1,  1,  0),   # multiply by  j:  (a,b) → (-b, a)
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_pairs(flat):
    """
    Convert a flat list/array of length 2k into a list of k tuples (a, b).

    WHY pairs?
    ─────────
    The entire system operates on 2D vectors because T_i acts on pairs.
    QPSK also encodes 2 bits per pair. Reshaping into pairs makes both
    the rotation inversion and the QPSK decoding cleaner and less error-prone
    than working with flat indices.

    Example: [1, 2, 3, 4] → [(1, 2), (3, 4)]
    """
    flat = list(flat)
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def detect_rotation(y_pilot):
    """
    ML estimate of which T_i was applied, using the 8 received pilot values.

    WHY maximum likelihood (ML)?
    ────────────────────────────
    The received pilot is:  y_pilot = T_i(pilot) + Z,  Z ~ N(0, I_8)
    The likelihood of observing y_pilot given hypothesis i is:
        p(y_pilot | i) = (2π)^(-4) * exp(-‖y_pilot - candidate_i‖² / 2)
    Maximising the likelihood over i is the same as MINIMISING ‖y_pilot - candidate_i‖².
    That's the minimum squared Euclidean distance rule.

    WHY is this optimal?
    ───────────────────
    Because the noise is Gaussian and equal-variance in all directions,
    the nearest-neighbour rule minimises the probability of error.
    This is the Bayes-optimal rule when all 4 hypotheses are equally likely
    (which they are — the problem says T_i is chosen uniformly at random).

    Returns
    -------
    i_hat : int in {1, 2, 3, 4}
        Detected rotation index.
    distances : dict
        Squared distances to all 4 candidates (useful for confidence scoring).
    """
    y_pilot = np.array(y_pilot)
    distances = {
        i: float(np.sum((y_pilot - PILOT_CANDIDATES[i]) ** 2))
        for i in PILOT_CANDIDATES
    }
    i_hat = min(distances, key=distances.get)
    return i_hat, distances


def invert_rotation(symbol_pairs, i_hat):
    """
    Apply the inverse of T_{i_hat} to a list of (a, b) pairs.

    WHY apply the inverse instead of just re-encoding?
    ──────────────────────────────────────────────────
    After the channel, every (a, b) pair in the data has been rotated by T_i.
    To recover the original QPSK symbol, we undo that rotation.
    The inverse is applied BEFORE QPSK decoding, not after, because QPSK
    decoding assumes symbols are axis-aligned. Trying to decode a rotated
    QPSK constellation would give completely wrong results.

    Parameters
    ----------
    symbol_pairs : list of (float, float)
    i_hat        : int in {1, 2, 3, 4}

    Returns
    -------
    list of (float, float) — rotation corrected pairs, still noisy
    """
    m = INV_ROTATION[i_hat]
    return [(m[0] * a + m[1] * b,
             m[2] * a + m[3] * b)
            for (a, b) in symbol_pairs]


def combine_copies(copy1, copy2):
    """
    Combine two received copies of the same data by element-wise summation.

    WHY sum instead of average or something else?
    ──────────────────────────────────────────────
    Both copies carry the same signal s but independent noise Z1 and Z2.
    After rotation inversion, copy k looks like: s + Z_k.

    Summing gives: (s + Z1) + (s + Z2) = 2s + (Z1 + Z2)
    - Signal part:  2s  (doubles)
    - Noise part:   Z1 + Z2 ~ N(0, 2)  (standard deviation grows by √2)

    Signal-to-noise ratio:
        Before combining:  SNR = s² / 1  (variance of single noise)
        After  combining:  SNR = (2s)² / 2 = 4s²/2 = 2s²

    So SNR doubles (3 dB gain). This is the maximum achievable from 2 equal copies.
    Summing (rather than averaging) is equivalent for the sign-based QPSK decision
    rule that follows — the factor of 2 doesn't matter when you only care about sign.

    Parameters
    ----------
    copy1, copy2 : list of (float, float) — same length

    Returns
    -------
    list of (float, float) — combined pairs
    """
    assert len(copy1) == len(copy2), "Copies must have the same length"
    return [
        (copy1[k][0] + copy2[k][0],
         copy1[k][1] + copy2[k][1])
        for k in range(len(copy1))
    ]


def qpsk_decode(combined_pairs):
    """
    Decode a list of (y1, y2) combined pairs into a flat list of bits.

    WHY sign-based decision?
    ────────────────────────
    QPSK encodes 2 bits into one 2D symbol. The 4 constellation points are at:
        (+1, +1)  →  bits 00
        (-1, +1)  →  bits 10
        (-1, -1)  →  bits 11
        (+1, -1)  →  bits 01

    Because the noise is isotropic Gaussian (N(0, I)), the optimal decision
    boundary between (+x, *) and (-x, *) is the vertical axis (y1 = 0).
    Similarly for y2. So each component is decoded independently by its sign.

    WHY is this the ML rule for QPSK?
    ──────────────────────────────────
    The ML rule picks the constellation point maximising p(y | symbol).
    For Gaussian noise, p(y | (s1, s2)) ∝ exp(-(y1-s1)² - (y2-s2)²).
    Since y1 and y2 are independent, each bit is decoded independently.
    For b1: p(y1 | s1=+1) > p(y1 | s1=-1)  iff  y1 > 0.
    So sign(y1) gives the ML decision for b1. Same for b2.

    Bit encoding convention (must match Person 2's encoder):
        b1 = 0 if y1 ≥ 0,  b1 = 1 if y1 < 0
        b2 = 0 if y2 ≥ 0,  b2 = 1 if y2 < 0

    Returns
    -------
    list of int (0 or 1), length = 2 * len(combined_pairs)
    """
    bits = []
    for (y1, y2) in combined_pairs:
        bits.append(0 if y1 >= 0 else 1)   # MSB of the pair
        bits.append(0 if y2 >= 0 else 1)   # LSB of the pair
    return bits


# =============================================================================
# CONVOLUTIONAL CODE — K=3 RATE-1/2 (GENERATORS 7, 5 OCTAL)
# =============================================================================
#
# Encoder structure:
#   state register holds (b_{k-1}, b_{k-2})  →  packed as state = 2·b_{k-1} + b_{k-2}
#   on input b_k, outputs are
#       c1 = b_k ⊕ b_{k-1} ⊕ b_{k-2}    (generator G1 = 111₂ = 7₈ = 1 + D + D²)
#       c2 = b_k ⊕ b_{k-2}              (generator G2 = 101₂ = 5₈ = 1 + D²)
#   then state shifts: (b_{k-1}, b_{k-2}) ← (b_k, b_{k-1})
#
# Free distance d_free = 5 (the minimum-weight non-zero codeword starting and
# returning to state 0 has Hamming weight 5). For BPSK over AWGN with soft
# Viterbi decoding, BER ≈ Q(√(d_free · 2R · Eb/N0)).
#
# We append (K-1) = 2 zero termination bits so the encoder ends in state 0,
# letting the Viterbi decoder anchor its traceback at a known final state.

CONV_K = 3                         # constraint length
CONV_NSTATES = 1 << (CONV_K - 1)   # 4 trellis states
CONV_TERM_BITS = CONV_K - 1        # 2 termination bits


def conv_encode(info_bits):
    """
    Encode info_bits with the K=3 (G1=7, G2=5) rate-1/2 convolutional code.

    The encoder starts in state 0; two zero termination bits are appended
    so it ends in state 0 (clean traceback for the decoder).

    Parameters
    ----------
    info_bits : iterable of int (0 or 1), length N

    Returns
    -------
    list of int (0 or 1), length 2*(N + 2)
    """
    state = 0
    output = []
    bits = list(info_bits) + [0] * CONV_TERM_BITS

    for bit in bits:
        s1 = (state >> 1) & 1   # b_{k-1}
        s0 = state & 1          # b_{k-2}
        c1 = bit ^ s1 ^ s0      # G1 = 1 + D + D²
        c2 = bit ^ s0           # G2 = 1 + D²
        output.append(c1)
        output.append(c2)
        # Shift register: new b_{k-1} = bit, new b_{k-2} = old b_{k-1}
        state = ((bit << 1) | s1) & (CONV_NSTATES - 1)

    return output


def viterbi_decode_soft(rotated_pairs, n_info_bits):
    """
    Soft-decision Viterbi decoder for the K=3 (7,5) rate-1/2 conv code.

    WHY soft decision?
    ──────────────────
    The decoder operates directly on the post-rotation-inversion real values
    rotated_pairs[k] = (y1, y2), without thresholding to bits first. The branch
    metric is the correlation between the received pair and the candidate
    transmitted pair (1-2c1, 1-2c2)·s. Soft decoding gives ~2 dB extra coding
    gain compared to hard-decision Viterbi.

    WHY one trellis step per pair?
    ──────────────────────────────
    The encoder emits 2 coded bits per input bit, and the transmitter packs each
    pair (c1, c2) into one channel pair (x[2k], x[2k+1]). The channel applies
    T_i pair-wise. After inverse rotation the pair (y1, y2) is exactly the soft
    observation of (c1, c2) — i.e. one trellis step.

    Parameters
    ----------
    rotated_pairs : sequence of (float, float)
        Soft observations after T_i⁻¹. Sign convention:
            bit 0 → +s (positive y is evidence for bit 0)
            bit 1 → -s
    n_info_bits : int
        Number of information bits to recover. The 2 termination bits used by
        the encoder are dropped on output.

    Returns
    -------
    list of int (0 or 1), length n_info_bits
    """
    n_steps = len(rotated_pairs)

    # Precompute trellis transitions:
    #   trans[state][bit] = (new_state, c1, c2)
    trans = [[None, None] for _ in range(CONV_NSTATES)]
    for state in range(CONV_NSTATES):
        s1 = (state >> 1) & 1
        s0 = state & 1
        for bit in (0, 1):
            c1 = bit ^ s1 ^ s0
            c2 = bit ^ s0
            new_state = ((bit << 1) | s1) & (CONV_NSTATES - 1)
            trans[state][bit] = (new_state, c1, c2)

    NEG_INF = -1e18
    metrics = [NEG_INF] * CONV_NSTATES
    metrics[0] = 0.0   # encoder starts in state 0

    # paths[step][new_state] = (prev_state, input_bit) on the surviving path
    paths = [[(0, 0)] * CONV_NSTATES for _ in range(n_steps)]

    for step in range(n_steps):
        y1, y2 = rotated_pairs[step]
        new_metrics = [NEG_INF] * CONV_NSTATES
        for state in range(CONV_NSTATES):
            m = metrics[state]
            if m <= NEG_INF / 2:
                continue
            for bit in (0, 1):
                new_state, c1, c2 = trans[state][bit]
                # Correlation: matches +y if c=0 (tx +s), -y if c=1 (tx -s)
                branch = (1 - 2 * c1) * y1 + (1 - 2 * c2) * y2
                cand = m + branch
                if cand > new_metrics[new_state]:
                    new_metrics[new_state] = cand
                    paths[step][new_state] = (state, bit)
        metrics = new_metrics

    # Encoder terminates in state 0 → start traceback there. Fall back to the
    # best state only if state 0 was somehow unreachable (shouldn't happen).
    final_state = 0
    if metrics[0] <= NEG_INF / 2:
        final_state = max(range(CONV_NSTATES), key=lambda s: metrics[s])

    decoded = [0] * n_steps
    state = final_state
    for step in range(n_steps - 1, -1, -1):
        prev_state, bit = paths[step][state]
        decoded[step] = bit
        state = prev_state

    # Drop the K-1 termination bits.
    return decoded[:n_info_bits]


def bits_to_message(bits):
    """
    Convert a flat list of 240 bits into a 40-character string.

    WHY 6 bits per character?
    ─────────────────────────
    The alphabet has 64 characters = 2^6. So exactly 6 bits are needed
    to index into it. 40 characters × 6 bits = 240 bits total.
    The 240 bits come from 120 QPSK symbols × 2 bits per symbol.
    Everything closes perfectly.

    WHY big-endian (MSB first)?
    ───────────────────────────
    The first bit of the 6-bit group is the most significant bit.
    int('011010', 2) = 26, which maps to 'A' (index 26 in ALPHABET).
    This is the natural binary convention and must match Person 2's encoder.

    Parameters
    ----------
    bits : list of int (0 or 1), length must be multiple of 6

    Returns
    -------
    str of length len(bits) // 6
    """
    assert len(bits) == 240, f"Expected 240 bits, got {len(bits)}"
    message = ''
    for i in range(0, 240, 6):
        chunk = bits[i:i + 6]
        # Convert 6-bit binary to integer: [0,1,1,0,1,0] → 26
        k = int(''.join(map(str, chunk)), 2)
        message += ALPHABET[k]
    return message


# =============================================================================
# MAIN DECODE FUNCTION
# =============================================================================

def decode(y):
    """
    Decode a received 500-element vector into the original 40-character message.

    Pipeline:
        y[0:16]    → 8 pilot pairs  → ML rotation detection (i_hat)
        y[16:500]  → 242 data pairs → inverse rotation → soft Viterbi → 240 bits
        240 bits   → 40 characters (bits_to_message)

    Parameters
    ----------
    y : array-like of length 500
        The noisy, rotated output from the channel.

    Returns
    -------
    str of length 40

    Raises
    ------
    ValueError if y does not have length 500.
    """
    y = list(y)

    if len(y) != 500:
        raise ValueError(f"Expected y of length 500, got {len(y)}")

    # ── Step 1: pilots → rotation detection ───────────────────────────────
    y_pilot = np.array(y[0:16])
    i_hat, _ = detect_rotation(y_pilot)

    # ── Step 2: data pairs → inverse rotation ─────────────────────────────
    # 484 data values = 242 pairs. Each pair carries one trellis step's
    # (c1, c2) BPSK symbols. After T_i⁻¹ each pair is back to (±s, ±s) + noise.
    data_pairs = to_pairs(y[16:500])
    corrected_pairs = invert_rotation(data_pairs, i_hat)

    # ── Step 3: soft-decision Viterbi → 240 info bits ─────────────────────
    info_bits = viterbi_decode_soft(corrected_pairs, n_info_bits=240)

    # ── Step 4: bits → message ────────────────────────────────────────────
    return bits_to_message(info_bits)
