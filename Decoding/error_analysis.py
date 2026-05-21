"""
error_analysis.py — Error Analysis and Reliability Evaluation
==============================================================
Answers the questions:
  - Where do errors come from? (wrong T_i vs QPSK noise)
  - Which symbols are most vulnerable?
  - What is the theoretical vs empirical BER?
  - What grade can we expect on the demo?

HOW TO RUN
──────────
    python error_analysis.py

Results are printed to stdout.
"""

import math
import numpy as np
import scipy.special as sp
from collections import Counter

from decode      import decode, r, ALPHABET
from decode_robust import decode_with_confidence, rotation_confidence
from channel_sim import channel, channel_fixed_Ti
from test_harness import stub_encode


# =============================================================================
# ERROR SOURCE CLASSIFICATION
# =============================================================================

def classify_errors(message, n_trials=2000, seed=0):
    """
    Run n_trials transmissions and classify each failure by its root cause.

    WHY classify?
    ─────────────
    The decoder has exactly two independent failure modes:

    1. WRONG T_i DETECTION (catastrophic)
       - The detected i_hat ≠ true i
       - Every symbol is decoded in the wrong frame → all bits random
       - Expect roughly 20 wrong characters out of 40
       - Probability: ~10⁻⁵ per transmission (from the spec)
       - Cause: noise pushed the received pilot past a decision boundary

    2. QPSK SYMBOL ERROR (scattered)
       - i_hat is correct, but individual symbol components cross zero
       - Each bit flips independently with probability BER ≈ Q(√(2·SNR))
       - Expect 0–5 wrong characters per failure at this SNR level
       - Probability: ~BER per bit, roughly a few ×10⁻³

    Classification rule:
       > 20 wrong chars → almost certainly catastrophic (wrong T_i)
       ≤ 20 wrong chars → scattered QPSK noise errors

    This threshold is conservative: even with completely random bits,
    you'd expect exactly 40 × (63/64) ≈ 39.4 wrong characters on average.
    So > 20 is a strong signal of the catastrophic case.
    """
    np.random.seed(seed)
    x = stub_encode(message)

    results = {
        'perfect':      [],
        'scattered':    [],   # 1-20 char errors
        'catastrophic': [],   # >20 char errors
        'Ti_detected':  [],
        'confidences':  [],
    }

    for _ in range(n_trials):
        y = channel(x)
        result = decode_with_confidence(y)
        recovered = result['message']
        errors = sum(a != b for a, b in zip(message, recovered))

        results['Ti_detected'].append(result['i_hat'])
        results['confidences'].append(result['confidence']['gap'])

        if errors == 0:
            results['perfect'].append(0)
        elif errors <= 20:
            results['scattered'].append(errors)
        else:
            results['catastrophic'].append(errors)

    return results


def print_classification_report(results, n_trials):
    n_perfect      = len(results['perfect'])
    n_scattered    = len(results['scattered'])
    n_catastrophic = len(results['catastrophic'])

    print("\n── Error Source Classification ─────────────────────────────────")
    print(f"  Trials:        {n_trials}")
    print(f"  Perfect:       {n_perfect}  ({n_perfect/n_trials*100:.2f}%)")
    print(f"  Scattered:     {n_scattered}  ({n_scattered/n_trials*100:.2f}%)  ← QPSK noise")
    print(f"  Catastrophic:  {n_catastrophic}  ({n_catastrophic/n_trials*100:.4f}%)  ← wrong T_i")
    print()
    ti_counts = Counter(results['Ti_detected'])
    print(f"  T_i distribution: { {k: ti_counts[k] for k in sorted(ti_counts)} }")
    expected_each = n_trials // 4
    print(f"  (Expected ~{expected_each} each for uniform T_i)")
    print()
    confs = results['confidences']
    print(f"  Confidence gap stats:")
    print(f"    Mean: {np.mean(confs):.2f}")
    print(f"    Min:  {np.min(confs):.3f}")
    print(f"    Std:  {np.std(confs):.2f}")
    if n_catastrophic > 0:
        # Find confidences on catastrophic runs
        all_errors = (
            [0]*n_perfect +
            results['scattered'] +
            results['catastrophic']
        )
        print(f"\n  ⚠ {n_catastrophic} catastrophic failure(s) detected.")
        print(f"    These are expected to have very low confidence gaps.")


# =============================================================================
# BER vs SNR ANALYSIS
# =============================================================================

def ber_vs_snr():
    """
    Compare theoretical vs empirical BER for this system.

    THEORETICAL BER FOR QPSK WITH DIVERSITY COMBINING
    ───────────────────────────────────────────────────
    Standard QPSK BER (no diversity, 1 copy):
        BER₁ = Q(√(2·s²))

    With 2 copies summed (MRC combining):
        Combined signal amplitude: 2s
        Combined noise variance:   2 (sum of two independent unit variances)
        Effective SNR:             (2s)²/2 = 2s²

    So BER with 2-copy combining:
        BER₂ = Q(√(2 · 2s²)) = Q(√(4s²)) = Q(2s)

    The 3 dB SNR gain from diversity means BER drops significantly.

    WHY does BER matter for the grade?
    ───────────────────────────────────
    The grade formula is: max(10 - χ, 0) where χ = wrong characters.
    Wrong character probability per char ≈ 1 - (1 - BER)^6
    Expected χ = 40 × P(char wrong)
    Expected score = max(10 - 40 × P(char wrong), 0)
    """
    s_sq = (1200 - 4 * r**2) / 480

    # QPSK component amplitude s = sqrt(s_sq)
    # After combining two copies: signal per component = 2s, noise std = sqrt(2)
    # BER = Q(2s/sqrt(2)) = Q(s*sqrt(2))
    # Using Q(x) = 0.5*erfc(x/sqrt(2)): BER = 0.5*erfc(s) = 0.5*erfc(sqrt(s_sq))

    # Without diversity (1 copy): BER = Q(s/1) = 0.5*erfc(s/sqrt(2)) = 0.5*erfc(sqrt(s_sq/2))
    snr1 = s_sq
    ber1 = 0.5 * sp.erfc(math.sqrt(s_sq / 2))   # Q(s) = 0.5*erfc(s/sqrt(2))

    # With 2-copy combining: BER = Q(s*sqrt(2)) = 0.5*erfc(sqrt(s_sq))
    snr2 = 2 * s_sq
    ber2 = 0.5 * sp.erfc(math.sqrt(s_sq))        # Q(s*sqrt(2)) = 0.5*erfc(s)

    # Char error probabilities
    p_char_wrong_1 = 1 - (1 - ber1) ** 6
    p_char_wrong_2 = 1 - (1 - ber2) ** 6

    expected_char_errors_1 = 40 * p_char_wrong_1
    expected_char_errors_2 = 40 * p_char_wrong_2

    expected_score_1 = max(10 - expected_char_errors_1, 0)
    expected_score_2 = max(10 - expected_char_errors_2, 0)

    print("\n── BER vs SNR Analysis ─────────────────────────────────────────")
    print(f"  QPSK component amplitude s = {math.sqrt(s_sq):.4f}")
    print(f"  s² = {s_sq:.4f}")
    print()
    print(f"  WITHOUT diversity combining (1 copy):")
    print(f"    SNR per component:       {snr1:.4f}  ({10*math.log10(snr1):.2f} dB)")
    print(f"    BER:                     {ber1:.6f}")
    print(f"    Expected wrong chars:    {expected_char_errors_1:.4f} / 40")
    print(f"    Expected demo score:     {expected_score_1:.2f} / 13")
    print()
    print(f"  WITH diversity combining (2 copies summed):")
    print(f"    SNR per component:       {snr2:.4f}  ({10*math.log10(snr2):.2f} dB)")
    print(f"    BER:                     {ber2:.6f}")
    print(f"    Expected wrong chars:    {expected_char_errors_2:.4f} / 40")
    print(f"    Expected demo score:     {expected_score_2:.2f} / 13")
    print()
    print(f"  Diversity gain: {10*math.log10(snr2/snr1):.1f} dB  (factor {snr2/snr1:.1f}× in SNR)")
    print(f"  Score improvement: {expected_score_1:.2f} → {expected_score_2:.2f}")


# =============================================================================
# PER-SYMBOL VULNERABILITY ANALYSIS
# =============================================================================

def symbol_vulnerability_analysis(message, n_trials=500, seed=1):
    """
    Find which symbol positions are most frequently wrong across many runs.

    WHY?
    ────
    Some characters may be more vulnerable than others — not because of their
    content, but because of random noise. By tracking which symbol POSITIONS
    accumulate errors over many trials, you can identify systematic weaknesses.
    (In this design there shouldn't be any since noise is i.i.d., but it's
    good to verify empirically.)
    """
    np.random.seed(seed)
    x = stub_encode(message)
    error_counts = np.zeros(40, dtype=int)   # per character position

    for _ in range(n_trials):
        y = channel(x)
        recovered = decode(y)
        for pos, (orig, rec) in enumerate(zip(message, recovered)):
            if orig != rec:
                error_counts[pos] += 1

    print("\n── Per-Character Error Frequency ────────────────────────────────")
    print(f"  (over {n_trials} trials)")
    print()
    total_errors = error_counts.sum()
    if total_errors == 0:
        print("  No errors observed in any trial — decoder is highly reliable.")
    else:
        print("  Positions with errors (most frequent first):")
        sorted_pos = np.argsort(error_counts)[::-1]
        for pos in sorted_pos:
            if error_counts[pos] == 0:
                break
            rate = error_counts[pos] / n_trials * 100
            char = message[pos]
            print(f"    Position {pos:>2} ('{char}'): {error_counts[pos]} errors  ({rate:.2f}%)")
    print()
    print(f"  Total char-errors across {n_trials} trials: {total_errors}")
    print(f"  Total char-positions tested: {n_trials * 40}")
    empirical_per_char_ber = total_errors / (n_trials * 40)
    print(f"  Empirical per-char error rate: {empirical_per_char_ber:.6f}")

    return error_counts


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    MESSAGE = 'Hello this is a PDC test message ok.'
    MESSAGE = (MESSAGE + ' ' * 40)[:40]
    N_TRIALS = 1000

    print(f"PDC Decoder Error Analysis")
    print(f"Message: '{MESSAGE}'")
    print(f"Trials:  {N_TRIALS}")

    # 1. BER analysis (no simulation needed)
    ber_vs_snr()

    # 2. Error source classification
    print(f"\nRunning {N_TRIALS} Monte Carlo trials for classification...")
    results = classify_errors(MESSAGE, n_trials=N_TRIALS)
    print_classification_report(results, N_TRIALS)

    # 3. Per-symbol vulnerability (lighter run)
    symbol_vulnerability_analysis(MESSAGE, n_trials=500)
