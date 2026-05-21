"""
End-to-end tests: your encoder (Encoding/) + decoder (Decoding/).

Does not modify any Decoding source. Decoding/test_harness.py keeps stub_encode
as the decoder team's simulator; this file uses encode() from encode.py only.

Run from Encoding/:
    python test_with_decoder.py
    python test_with_decoder.py --mc --trials 1000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_DECODING_DIR = _ROOT / "Decoding"
if str(_DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(_DECODING_DIR))

from encode import encode, check_constraints, ALPHABET as ENCODER_ALPHABET
from decode import decode, r, ALPHABET, PILOT_CANDIDATES
from decode_robust import decode_with_confidence
from channel_sim import channel, channel_noiseless


def team_encode(message: str) -> np.ndarray:
    """Person 2 transmitter — not the Decoding stub simulator."""
    assert ALPHABET == ENCODER_ALPHABET
    x = np.asarray(encode(message), dtype=float)
    ok, details = check_constraints(x)
    assert ok, details
    return x


def pad_message(message: str) -> str:
    return (message + " " * 40)[:40]


def test_alphabet_match():
    print("Alphabet match encoder/decoder...", end=" ")
    assert ALPHABET == ENCODER_ALPHABET
    print("PASS")


def test_energy_and_length():
    print("Encoder constraints...", end=" ")
    x = team_encode("a" * 40)
    ok, info = check_constraints(x)
    assert ok
    assert info["length"] == 500
    print(f"PASS  (energy={info['energy']:.2f})")


def test_noiseless_all_rotations():
    print("Noiseless decode (all T_i)...", end=" ")
    message = pad_message("Hello World from PDC project 2026.")
    x = team_encode(message)
    for i in range(1, 5):
        y = channel_noiseless(x, i)
        assert decode(y) == message, f"T{i} failed"
    print("PASS")


def test_single_noisy():
    print("Noisy encode/channel/decode...", end=" ")
    np.random.seed(42)
    message = pad_message("The quick brown fox jumps over.")
    x = team_encode(message)
    recovered = decode(channel(x))
    errors = sum(a != b for a, b in zip(message, recovered))
    status = "PASS" if errors == 0 else f"WARN ({errors} char errors — noise)"
    print(status)
    return errors


def monte_carlo(message: str, n_trials: int = 200, seed: int | None = 42) -> dict:
    if seed is not None:
        np.random.seed(seed)
    x = team_encode(message)
    stats = {
        "n_trials": n_trials,
        "n_perfect": 0,
        "n_failures": 0,
        "char_errors": [],
        "Ti_counts": {1: 0, 2: 0, 3: 0, 4: 0},
        "catastrophic": 0,
        "scattered": 0,
    }
    for _ in range(n_trials):
        result = decode_with_confidence(channel(x))
        recovered = result["message"]
        stats["Ti_counts"][result["i_hat"]] += 1
        errors = sum(a != b for a, b in zip(message, recovered))
        if errors == 0:
            stats["n_perfect"] += 1
        else:
            stats["n_failures"] += 1
            stats["char_errors"].append(errors)
            if errors > 20:
                stats["catastrophic"] += 1
            else:
                stats["scattered"] += 1
    stats["success_rate"] = stats["n_perfect"] / n_trials
    stats["failure_rate"] = stats["n_failures"] / n_trials
    if stats["char_errors"]:
        stats["mean_char_errors"] = float(np.mean(stats["char_errors"]))
        stats["max_char_errors"] = int(np.max(stats["char_errors"]))
    else:
        stats["mean_char_errors"] = 0.0
        stats["max_char_errors"] = 0
    scores = [13] * stats["n_perfect"] + [
        max(10 - e, 0) for e in stats["char_errors"]
    ]
    stats["expected_grade"] = float(np.mean(scores)) if scores else 13.0
    return stats


def print_mc_report(stats: dict, message: str) -> None:
    n = stats["n_trials"]
    print()
    print("=" * 60)
    print("MONTE CARLO (Person 2 encoder + Decoding receiver)")
    print("=" * 60)
    print(f"Message:       '{message}'")
    print(f"Trials:        {n}")
    print(f"Success rate:  {stats['success_rate']*100:.2f}%  ({stats['n_perfect']}/{n})")
    print(f"Failure rate:  {stats['failure_rate']*100:.2f}%  ({stats['n_failures']}/{n})")
    if stats["n_failures"]:
        print(f"  Scattered (1-20): {stats['scattered']}")
        print(f"  Catastrophic (>20): {stats['catastrophic']}")
        print(f"  Mean char errors: {stats['mean_char_errors']:.2f}")
    print(f"T_i counts: {stats['Ti_counts']}")
    print(f"Expected grade: {stats['expected_grade']:.2f} / 13")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder + decoder integration tests")
    parser.add_argument("--mc", action="store_true")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--message",
        type=str,
        default="Hello this is a PDC test message ok.",
    )
    args = parser.parse_args()
    msg = pad_message(args.message)

    print("\n-- Integration tests (real encoder) --")
    test_alphabet_match()
    test_energy_and_length()
    test_noiseless_all_rotations()
    test_single_noisy()
    print("Unit/integration checks done.\n")

    trials = args.trials if args.mc else 200
    label = f"Monte Carlo ({trials} trials)"
    if not args.mc:
        print(f"(Use --mc --trials 1000 for full run)\n{label}:")
    else:
        print(label + ":")
    stats = monte_carlo(msg, n_trials=trials, seed=args.seed)
    print_mc_report(stats, msg)


if __name__ == "__main__":
    main()
