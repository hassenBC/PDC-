import argparse
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# demo.py on main reads/writes these when run from Decoding/
DEMO_IO_DIR = REPO_ROOT / "Decoding"
DEFAULT_INPUT = DEMO_IO_DIR / "input.txt"
DEFAULT_OUTPUT = DEMO_IO_DIR / "output.txt"


ALPHABET = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " ."
)

MESSAGE_LEN    = 40
INFO_BITS      = 240   # 40 chars * 6 bits each
VECTOR_LEN     = 500
MAX_ENERGY     = 1200.0
NUM_PILOT_PAIRS = 8
CODED_BITS     = 484   # 2 * (240 info + 2 termination bits)

# energy budget: 8 pilot pairs + 484 data samples = 492 non-zero values
# each has amplitude r, so 492 * r^2 = 1200
r = math.sqrt(MAX_ENERGY / (NUM_PILOT_PAIRS + CODED_BITS))

# K=3 rate-1/2 convolutional code, generators G1=7 G2=5 (octal)
# 4 encoder states (2^(K-1)), 2 termination bits to flush state back to 0
CONV_K         = 3
CONV_NSTATES   = 4
CONV_TERM_BITS = 2


# --- text <-> bits --------------------------------------------------------

def validate_message(message):
    if len(message) != MESSAGE_LEN:
        raise ValueError(f"need {MESSAGE_LEN} chars, got {len(message)}")
    for c in message:
        if c not in ALPHABET:
            raise ValueError(f"bad character: {c!r}")


def message_to_bits(message):
    # each char becomes its 6-bit index in ALPHABET, MSB first
    validate_message(message)
    bits = []
    for ch in message:
        k = ALPHABET.index(ch)
        for shift in range(5, -1, -1):
            bits.append((k >> shift) & 1)
    return bits   # length 240


def bits_to_message(bits):
    # inverse of message_to_bits: groups of 6 bits -> character index -> char
    if len(bits) != INFO_BITS:
        raise ValueError(f"expected {INFO_BITS} bits")
    msg = ""
    for i in range(0, INFO_BITS, 6):
        val = 0
        for b in bits[i:i + 6]:
            val = (val << 1) | b
        msg += ALPHABET[val]
    return msg


# --- convolutional encoder ------------------------------------------------

def conv_encode(info_bits):
    # one input bit -> two output bits (c1, c2) using shift register state
    # c1 = bit XOR s1 XOR s0  (generator 111 = 7 octal)
    # c2 = bit XOR s0          (generator 101 = 5 octal)
    # two zero bits appended so encoder ends in state 0 (clean Viterbi traceback)
    state = 0
    out = []
    for bit in list(info_bits) + [0, 0]:
        s1 = (state >> 1) & 1
        s0 = state & 1
        out.append(bit ^ s1 ^ s0)  # c1
        out.append(bit ^ s0)       # c2
        state = ((bit << 1) | s1) & (CONV_NSTATES - 1)
    return out   # length 484


# --- modulation and pilots ------------------------------------------------

def bpsk_modulate(coded_bits):
    # bit 0 -> +r,  bit 1 -> -r
    return [r if b == 0 else -r for b in coded_bits]


def build_pilots():
    # (r, 0) rotates to (r,0), (0,r), (-r,0), (0,-r) under T1-T4
    # four distinct cardinal points -> receiver can identify which Ti occurred
    pilots = []
    for _ in range(NUM_PILOT_PAIRS):
        pilots.extend([r, 0.0])
    return pilots   # length 16


# --- constraints ----------------------------------------------------------

def check_constraints(x):
    n = len(x)
    energy = sum(v * v for v in x)
    ok = (n % 2 == 0) and n <= VECTOR_LEN and energy <= MAX_ENERGY + 1e-6
    return ok, {"length": n, "energy": energy}


def validate_for_client(x):
    # mirrors checks done by the EPFL server before processing
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    ok, info = check_constraints(x)
    if not ok:
        raise ValueError(f"bad signal: n={info['length']}, energy={info['energy']:.2f}")


# --- main API -------------------------------------------------------------

def encode(message, as_numpy=False):
    bits  = message_to_bits(message)
    coded = conv_encode(bits)
    x     = build_pilots() + bpsk_modulate(coded)
    ok, info = check_constraints(x)
    if not ok:
        raise ValueError(info)
    return np.array(x, dtype=float) if as_numpy else x


def write_channel_input(x, path="input.txt"):
    validate_for_client(x)
    np.savetxt(path, np.asarray(x, dtype=float))
    print(f"wrote {len(x)} lines to {path}")


# --- CLI ------------------------------------------------------------------

def pad_message(message):
    if len(message) > MESSAGE_LEN:
        print(f"truncating to {MESSAGE_LEN} chars")
    return (message + " " * MESSAGE_LEN)[:MESSAGE_LEN]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("message", nargs="?", help="40-char message (padded with spaces)")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_INPUT)
    args = p.parse_args()

    msg = args.message
    if msg is None:
        msg = input(f"message ({MESSAGE_LEN} chars): ").strip()
    msg = pad_message(msg)

    DEMO_IO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    x = encode(msg)
    write_channel_input(x, str(out_path))

    _, info = check_constraints(x)
    client = REPO_ROOT / "Python client" / "client.py"
    print(f"message: {msg!r}")
    print(f"n={info['length']}, energy={info['energy']:.2f}")
    print(f"wrote: {out_path.resolve()}")
    print()
    print("then run (on EPFL network or VPN):")
    print(
        f'  python "{client}" '
        f'--input_file="{out_path.resolve()}" '
        f'--output_file="{DEFAULT_OUTPUT.resolve()}" '
        "--srv_hostname=iscsrv72.epfl.ch --srv_port=80"
    )