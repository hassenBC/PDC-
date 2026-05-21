"""
test_harness.py — Test Suite and Reliability Evaluation
========================================================
Three categories of tests:
  1. Unit tests    — test each decoder component in isolation
  2. Integration   — test the full pipeline end-to-end
  3. Monte Carlo   — run 1000+ trials, measure error statistics

HOW TO RUN
──────────
Before running, you need a working encode() function from Person 2.
Until then, a stub encoder is provided that generates a valid x vector
from a known message so you can test the decoder independently.

Run all tests:
    python test_harness.py

Run just Monte Carlo:
    python test_harness.py --mc --trials 2000
"""

import sys
import math
import argparse
import numpy as np

from decode       import decode, r, ALPHABET, PILOT_CANDIDATES, conv_encode
from decode_robust import decode_with_confidence
from channel_sim  import channel, channel_fixed_Ti, channel_noiseless


# =============================================================================
# STUB ENCODER
# =============================================================================

def stub_encode(message):
    """
    Minimal encoder that produces a valid x vector for a given message.
    Replace this with Person 2's actual encode() once it's ready.

    Design (must match decode.py):
    - 16 pilot values:  [r,0,r,0,...,r,0]   (8 pairs, 8 non-zero)
    - 484 BPSK values:  rate-1/2 K=3 convolutional code over
                        (240 info bits + 2 termination bits) → 484 coded bits
    - Total:            16 + 484 = 500 values  (= server limit)

    Energy budget:
    - 8 + 484 = 492 non-zero components.
    - Equal energy per non-zero component (s = r):
          492·r² = 1200  →  r² = 1200/492 ≈ 2.439

    WHY a convolutional code instead of 2× repetition?
    ──────────────────────────────────────────────────
    Same energy budget, same channel uses, but free distance jumps from d=2
    (repetition) to d_free=5 (K=3 conv code). Soft-decision Viterbi at our
    Eb/N0 ≈ 3.87 dB gives BER ≈ 2.4×10⁻⁴ vs 1.33% for the repetition scheme —
    a ~55× reduction, pushing P(perfect 240-bit decode) from ~4% to ~94%.
    """
    assert len(message) == 40, f"Message must be 40 chars, got {len(message)}"
    assert all(c in ALPHABET for c in message), "Message contains invalid characters"

    # Message → 240 info bits (MSB first, 6 bits per character)
    bits = []
    for char in message:
        k = ALPHABET.index(char)
        for shift in range(5, -1, -1):
            bits.append((k >> shift) & 1)
    assert len(bits) == 240

    # Convolutional encode: 240 info + 2 termination → 484 coded bits
    coded_bits = conv_encode(bits)
    assert len(coded_bits) == 484, f"conv_encode produced {len(coded_bits)} bits"

    # BPSK modulate: bit 0 → +s, bit 1 → -s   (s = r, equal energy)
    s = r
    symbols = [s if cb == 0 else -s for cb in coded_bits]

    # Build x: 8 pilot pairs followed by 484 BPSK symbols
    pilots = [r, 0, r, 0, r, 0, r, 0, r, 0, r, 0, r, 0, r, 0]
    x = np.array(pilots + symbols, dtype=float)
    assert len(x) == 500
    assert np.sum(x**2) <= 1200 + 1e-6, f"Energy constraint violated: {np.sum(x**2):.2f}"
    return x


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_pilot_candidates():
    """
    Verify that PILOT_CANDIDATES match what the channel actually produces
    for a known input. If this fails, the rotation detection will be wrong
    for at least one T_i.
    """
    print("Unit test: pilot candidates...", end=' ')
    pilot_input = np.array([r, 0, r, 0, r, 0, r, 0, r, 0, r, 0, r, 0, r, 0])

    for i in range(1, 5):
        rotated = channel_noiseless(pilot_input, i)
        expected = PILOT_CANDIDATES[i]
        assert np.allclose(rotated, expected, atol=1e-10), (
            f"T{i}: channel gives {rotated}, candidate is {expected}"
        )
    print("PASS")


def test_rotation_inversion():
    """
    Verify that applying T_i then the inverse recovers the original vector exactly.
    Tests the algebraic correctness of INV_ROTATION for all 4 rotations.
    """
    from decode import invert_rotation, to_pairs

    print("Unit test: rotation inversion...", end=' ')
    test_pairs = [(1.0, 2.0), (-3.0, 4.0), (0.5, -1.5), (2.0, 0.0)]

    for i in range(1, 5):
        # Flatten pairs, apply channel rotation (noiseless), then invert
        flat = [v for p in test_pairs for v in p]
        rotated_flat = channel_noiseless(np.array(flat), i)
        rotated_pairs = to_pairs(rotated_flat)
        recovered = invert_rotation(rotated_pairs, i)

        for j, (orig, rec) in enumerate(zip(test_pairs, recovered)):
            assert abs(orig[0] - rec[0]) < 1e-10 and abs(orig[1] - rec[1]) < 1e-10, (
                f"T{i} inversion failed at pair {j}: original {orig}, recovered {rec}"
            )
    print("PASS")


def test_alphabet_coverage():
    """
    Verify the ALPHABET covers all 64 characters exactly once,
    and that the index mapping is invertible.
    """
    print("Unit test: alphabet coverage...", end=' ')
    assert len(ALPHABET) == 64
    assert len(set(ALPHABET)) == 64
    # Round-trip: index → char → index
    for k, char in enumerate(ALPHABET):
        assert ALPHABET.index(char) == k, f"Duplicate or wrong index for '{char}'"
    print("PASS")


def test_noiseless_decode_all_Ti():
    """
    Verify that decode() recovers the exact message for all 4 rotations
    when there is no noise.

    WHY this test?
    ──────────────
    This isolates the rotation detection + inversion logic from noise effects.
    If this fails, there's a deterministic bug (wrong candidate, wrong inverse matrix).
    If this passes but noisy decoding fails, the issue is statistical (SNR related).
    """
    print("Unit test: noiseless decode for all T_i...", end=' ')
    message = "Hello World from PDC project 2026."
    # Pad to 40 chars with spaces
    message = (message + ' ' * 40)[:40]
    x = stub_encode(message)

    for i in range(1, 5):
        y = channel_noiseless(x, i)
        recovered = decode(y)
        assert recovered == message, (
            f"T{i}: expected '{message}', got '{recovered}'"
        )
    print("PASS")


def test_energy_constraint():
    """
    Verify the encoded vector satisfies ‖x‖² ≤ 1200.
    The server rejects inputs that violate this — crucial for the demo.
    """
    print("Unit test: energy constraint...", end=' ')
    message = 'a' * 40
    x = stub_encode(message)
    energy = float(np.sum(x**2))
    assert energy <= 1200.0 + 1e-6, f"Energy constraint violated: {energy:.4f} > 1200"
    assert len(x) == 500, f"Vector length wrong: {len(x)}"
    assert len(x) % 2 == 0, "Vector length must be even"
    print(f"PASS  (energy={energy:.2f})")


def test_single_decode():
    """
    End-to-end test: encode → channel (random T_i) → decode.
    Verifies the full pipeline works for a typical message.
    """
    print("Integration test: single encode/channel/decode...", end=' ')
    np.random.seed(42)
    message = "The quick brown fox jumps over."
    message = (message + ' ' * 40)[:40]
    x = stub_encode(message)
    y = channel(x)
    recovered = decode(y)
    errors = sum(a != b for a, b in zip(message, recovered))
    status = "PASS" if errors == 0 else f"WARN ({errors} char errors — noise)"
    print(status)
    return errors


# =============================================================================
# MONTE CARLO RELIABILITY
# =============================================================================

def monte_carlo(message, n_trials=1000, seed=None, verbose=True):
    """
    Run n_trials independent transmissions and collect error statistics.

    WHY Monte Carlo?
    ────────────────
    A single test run tells you nothing about reliability. The QPSK BER
    is roughly Q(√(2·SNR)) ≈ a few ×10⁻³. With 240 bits per message,
    you expect ~0 to 1 bit errors per run on average, but the variance is high.
    1000 trials give you a stable estimate of:
    - Success rate (0 char errors)
    - Mean char errors on failure
    - Whether catastrophic failures (wrong T_i) ever occur

    Parameters
    ----------
    message  : str of length 40
    n_trials : int
    seed     : int or None (set for reproducibility)
    verbose  : bool

    Returns
    -------
    dict with full statistics
    """
    if seed is not None:
        np.random.seed(seed)

    x = stub_encode(message)

    stats = {
        'n_trials':       n_trials,
        'n_perfect':      0,
        'n_failures':     0,
        'char_errors':    [],
        'Ti_counts':      {1: 0, 2: 0, 3: 0, 4: 0},
        'catastrophic':   0,   # > 20 char errors → almost certainly wrong i_hat
        'scattered':      0,   # 1-20 char errors → QPSK noise
    }

    for trial in range(n_trials):
        y = channel(x)
        result = decode_with_confidence(y)
        recovered = result['message']
        i_hat = result['i_hat']
        stats['Ti_counts'][i_hat] += 1

        errors = sum(a != b for a, b in zip(message, recovered))

        if errors == 0:
            stats['n_perfect'] += 1
        else:
            stats['n_failures'] += 1
            stats['char_errors'].append(errors)
            if errors > 20:
                stats['catastrophic'] += 1
            else:
                stats['scattered'] += 1

        if verbose and (trial + 1) % 200 == 0:
            rate = stats['n_perfect'] / (trial + 1) * 100
            print(f"  Trial {trial+1:>5}/{n_trials}  |  success rate: {rate:.1f}%")

    # Summary statistics
    stats['success_rate'] = stats['n_perfect'] / n_trials
    stats['failure_rate'] = stats['n_failures'] / n_trials
    if stats['char_errors']:
        stats['mean_char_errors'] = float(np.mean(stats['char_errors']))
        stats['max_char_errors']  = int(np.max(stats['char_errors']))
    else:
        stats['mean_char_errors'] = 0.0
        stats['max_char_errors']  = 0

    # Expected grading score: max(10 - χ, 0), averaged over all trials
    # On perfect runs: score = 13 (full marks from spec)
    # On failure runs: score = max(10 - χ, 0)
    scores = []
    for errs in stats['char_errors']:
        scores.append(max(10 - errs, 0))
    for _ in range(stats['n_perfect']):
        scores.append(13)
    stats['expected_grade'] = float(np.mean(scores)) if scores else 13.0

    return stats


def print_mc_report(stats, message):
    """Pretty-print a Monte Carlo result report."""
    n = stats['n_trials']
    print()
    print("=" * 60)
    print("MONTE CARLO RELIABILITY REPORT")
    print("=" * 60)
    print(f"Message:       '{message}'")
    print(f"Trials:        {n}")
    print()
    print(f"Success rate:  {stats['success_rate']*100:.2f}%  ({stats['n_perfect']}/{n} perfect)")
    print(f"Failure rate:  {stats['failure_rate']*100:.2f}%  ({stats['n_failures']}/{n} with errors)")
    print()
    if stats['n_failures'] > 0:
        print(f"Among failures:")
        print(f"  Scattered errors (1–20 chars):  {stats['scattered']}")
        print(f"  Catastrophic     (>20 chars):   {stats['catastrophic']}  ← likely wrong T_i")
        print(f"  Mean char errors: {stats['mean_char_errors']:.2f}")
        print(f"  Max  char errors: {stats['max_char_errors']}")
    print()
    print(f"T_i distribution: {stats['Ti_counts']}  (should be ~uniform ≈ {n//4} each)")
    print()
    print(f"Expected grade:   {stats['expected_grade']:.2f} / 13  (demo scoring)")
    print("=" * 60)


# =============================================================================
# BER ANALYSIS
# =============================================================================

def theoretical_ber():
    """
    Theoretical BER for the K=3 (7,5) rate-1/2 conv-coded design.

    Channel parameters
    ──────────────────
    - 8 pilot pairs (16 elements, 8 non-zero) + 484 BPSK coded bits = 500 elements
    - 492 non-zero components share 1200 energy → s² = r² ≈ 2.439
    - Per channel-use SNR: 2s² (Es/(N0/2)), since noise variance = 1 = N0/2

    Per info bit (rate 1/2 → 2 channel uses per info bit):
        Eb/N0 = 2·s² / 2 = s²    (in linear)

    Soft-decision Viterbi, first-event bound (tight at high SNR):
        BER ≤ N_e · Q(√(d_free · 2R · Eb/N0))
              with N_e = 1, d_free = 5, R = 1/2  for the (7,5) code
            = Q(√(5 · s²))

    Pilot detection (8 pairs, adjacent-candidate distance d² = 16·r²):
        P(wrong T_i adjacent) = Q(d/2) = Q(2r)
        Union bound over 2 adjacent candidates: P(wrong T_i) ≈ 2·Q(2r)
    """
    import scipy.special as sp

    def Q(x): return 0.5 * sp.erfc(x / math.sqrt(2))

    s_sq = r ** 2                              # equal energy: s = r
    ber  = Q(math.sqrt(5 * s_sq))              # soft Viterbi, d_free = 5
    pilot_err = 2 * Q(2 * r)                   # 2 adjacent T_i hypotheses

    print(f"\nTheoretical BER Analysis  (K=3 conv code, soft Viterbi)")
    print(f"  s²  (energy/non-zero component):         {s_sq:.4f}")
    print(f"  Eb/N0 per info bit (linear):             {s_sq:.4f}  ({10*math.log10(s_sq):.2f} dB)")
    print(f"  BER ≈ Q(√(5·s²)):                        {ber:.6e}")
    print(f"  Expected wrong info bits per message:    {240 * ber:.4f}")
    print(f"  P(all 240 info bits correct | right T_i):{(1-ber)**240:.4f}")
    print(f"  P(correct T_i detection):                {1-pilot_err:.6f}")
    print(f"  Overall P(perfect decode):               {(1-pilot_err)*(1-ber)**240:.4f}")
    return ber


# =============================================================================
# MAIN
# =============================================================================

def run_all_unit_tests():
    print("\n── Unit Tests ──────────────────────────────────────")
    test_alphabet_coverage()
    test_energy_constraint()
    test_pilot_candidates()
    test_rotation_inversion()
    test_noiseless_decode_all_Ti()
    test_single_decode()
    print("All unit tests passed.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PDC Decoder Test Harness')
    parser.add_argument('--mc',      action='store_true', help='Run Monte Carlo test')
    parser.add_argument('--trials',  type=int, default=1000, help='Number of MC trials')
    parser.add_argument('--seed',    type=int, default=None, help='Random seed for MC')
    parser.add_argument('--ber',     action='store_true', help='Print BER analysis')
    parser.add_argument('--message', type=str,
                        default='Hello this is a PDC test message ok.',
                        help='40-char message to use (will be truncated/padded)')
    args = parser.parse_args()

    # Normalise message to 40 chars
    msg = (args.message + ' ' * 40)[:40]

    # Always run unit tests
    run_all_unit_tests()

    if args.ber:
        theoretical_ber()

    if args.mc:
        print(f"\n── Monte Carlo: {args.trials} trials ──────────────────────")
        print(f"Message: '{msg}'")
        stats = monte_carlo(msg, n_trials=args.trials, seed=args.seed, verbose=True)
        print_mc_report(stats, msg)
    else:
        # Quick demo run: 200 trials
        print(f"── Quick MC (200 trials) ────────────────────────────")
        print(f"Message: '{msg}'")
        print(f"(Run with --mc --trials 1000 for full evaluation)")
        stats = monte_carlo(msg, n_trials=200, seed=42, verbose=False)
        print_mc_report(stats, msg)
