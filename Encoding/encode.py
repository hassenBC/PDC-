"""
encode.py - Transmitter
Person 2: encode(message) -> x

Frame (n=500):
  x[0:16]   - 8 pilot pairs (r, 0) for rotation detection
  x[16:500] - 484 BPSK samples (conv coded message)

Constraints: n even, n<=500, ||x||^2 <= 1200
"""

import math
from typing import List, Sequence, Tuple, Union
import numpy as np

# alphabet must match decoder exactly - don't change order
ALPHABET = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " ."
)

MESSAGE_LEN = 40
INFO_BITS = 240        # 40 chars * 6 bits
VECTOR_LEN = 500
MAX_ENERGY = 1200.0
NUM_PILOT_PAIRS = 8
CODED_BITS = 484       # 2 * (240 + 2 termination bits)

# 8 pilots + 484 data = 492 non-zero values, each at amplitude r
# 492 * r^2 = 1200  =>  r = sqrt(1200/492)
r = math.sqrt(MAX_ENERGY / (NUM_PILOT_PAIRS + CODED_BITS))

# conv encoder params
CONV_K = 3
CONV_NSTATES = 1 << (CONV_K - 1)   # 4 states
CONV_TERM_BITS = CONV_K - 1         # 2 flush bits


def validate_message(message: str) -> None:
    if len(message) != MESSAGE_LEN:
        raise ValueError(f"need exactly {MESSAGE_LEN} chars, got {len(message)}")
    bad = [c for c in message if c not in ALPHABET]
    if bad:
        raise ValueError(f"chars not in alphabet: {bad}")


def message_to_bits(message: str) -> List[int]:
    """40 chars -> 240 bits, 6 bits per char MSB first"""
    validate_message(message)
    bits = []
    for ch in message:
        k = ALPHABET.index(ch)
        for shift in range(5, -1, -1):
            bits.append((k >> shift) & 1)
    return bits


def bits_to_message(bits: Sequence[int]) -> str:
    """240 bits -> 40 chars, inverse of message_to_bits"""
    if len(bits) != INFO_BITS:
        raise ValueError(f"need {INFO_BITS} bits, got {len(bits)}")
    msg = ""
    for i in range(0, INFO_BITS, 6):
        k = int("".join(str(b) for b in bits[i:i+6]), 2)
        msg += ALPHABET[k]
    return msg


def conv_encode(info_bits: Sequence[int]) -> List[int]:
    """
    K=3 rate-1/2 conv encoder, generators G1=7 G2=5 (octal).
    Appends 2 zero bits to flush the state back to 0.
    Output length = 2 * (len(info_bits) + 2) = 484
    """
    state = 0
    output = []
    for bit in list(info_bits) + [0] * CONV_TERM_BITS:
        s1 = (state >> 1) & 1
        s0 = state & 1
        output.append(bit ^ s1 ^ s0)   # c1, generator 111
        output.append(bit ^ s0)         # c2, generator 101
        state = ((bit << 1) | s1) & (CONV_NSTATES - 1)
    return output


def bpsk_modulate(coded_bits: Sequence[int]) -> List[float]:
    """bit 0 -> +r, bit 1 -> -r"""
    return [r if b == 0 else -r for b in coded_bits]


def build_pilots() -> List[float]:
    """8 pilot pairs (r, 0) -> 16 values"""
    out = []
    for _ in range(NUM_PILOT_PAIRS):
        out.extend([r, 0.0])
    return out


def check_constraints(x: Sequence[float]) -> Tuple[bool, dict]:
    n = len(x)
    energy = float(sum(v*v for v in x))
    ok = (n % 2 == 0) and (n <= VECTOR_LEN) and (energy <= MAX_ENERGY + 1e-6)
    return ok, {"length": n, "energy": energy}


def encode(message: str, *, as_numpy: bool = False) -> Union[List[float], np.ndarray]:
    """
    Main function. Converts a 40-char string into 500 floats ready to send.

    Steps:
      1. message -> 240 bits
      2. conv encode -> 484 coded bits
      3. BPSK -> 484 floats
      4. prepend 8 pilot pairs -> [pilots | data], length 500
    """
    bits = message_to_bits(message)
    coded = conv_encode(bits)
    data = bpsk_modulate(coded)
    x = build_pilots() + data

    ok, info = check_constraints(x)
    if not ok:
        raise ValueError(f"constraint violated: {info}")

    return np.array(x, dtype=float) if as_numpy else x


def write_channel_input(x: Sequence[float], path: str = "input.txt") -> None:
    """write x to file for use with client.py"""
    np.savetxt(path, np.asarray(x, dtype=float))
    print(f"wrote {len(x)} values to {path}")


if __name__ == "__main__":
    msg = ("Hello this is a PDC test message ok." + " " * MESSAGE_LEN)[:MESSAGE_LEN]
    x = encode(msg)
    ok, info = check_constraints(x)
    print(f"message: {msg!r}")
    print(f"n:       {info['length']}")
    print(f"energy:  {info['energy']:.4f} / {MAX_ENERGY}")
    print(f"r:       {r:.6f}")
    print(f"ok:      {ok}")