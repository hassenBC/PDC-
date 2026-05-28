# Encoder

## File Structure

```
Encoding/
├── encode.py              Core encode(message) → x function
├── test_encode.py         Unit tests (constraints, alphabet, round-trip)
├── test_with_decoder.py   End-to-end + Monte Carlo with Decoding/decode.py
└── README.md              This file
```

## Quick Start

```bash
pip install numpy

# run unit tests
python test_encode.py

# encode → ../Decoding/input.txt (where demo.py expects it)
python encode.py "your 40 character message here      "

# then send to server (EPFL network or VPN); command printed by encode.py
```

## Integration with Person 3 (Decoder)

The encoder and decoder share two things that must stay identical:

1. `ALPHABET` order — `'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789 .'`
2. Frame layout — `x[0:16]` pilots, `x[16:500]` BPSK data

To run end-to-end tests from this folder:

```bash
python test_with_decoder.py
python test_with_decoder.py --mc --trials 1000
```

`Decoding/demo.py` imports `encode()` from here for the May 29 demo.

## Design Choices

### 1. Why pilots?
Theory Problem 3(a) proves that if your codebook is symmetric (−c is also a valid codeword), error probability is at least 1/2 regardless of the decoder. BPSK is antipodal so this applies. The pilots break the symmetry: the receiver identifies which rotation Ti occurred before touching the data, then undoes it. Without pilots the decoder is guessing.

### 2. Why (r, 0) as the pilot pair?
Under the four rotations, (r, 0) maps to four cardinal points:
- T1 → ( r,  0) — positive real axis
- T2 → ( 0,  r) — positive imaginary axis
- T3 → (−r,  0) — negative real axis
- T4 → ( 0, −r) — negative imaginary axis

These are orthogonal and maximally separated (distance 2r between adjacent ones). Any other pilot choice, like (r, r), would give less separation and higher rotation detection error.

### 3. Why 8 pilot pairs and not 4?
With 4 pairs the pilot detection error is around 0.1% per transmission. With 8 it drops to essentially 0 in all MC runs. The cost is 8 extra dimensions out of 500, which reduces r by about 0.006 — negligible. Wrong rotation detection causes ~20 wrong characters (catastrophic failure), so it's worth the small energy cost to eliminate it.

### 4. Why BPSK and not QPSK?
QPSK encodes 2 bits per pair at (±r, ±r). Under rotation T2, every QPSK symbol maps exactly onto an adjacent symbol — a systematic cyclic shift that causes 75% error rate even with no noise. BPSK encodes 1 bit per sample on one axis only. After the receiver undoes the rotation, BPSK decoding is just a sign check on each sample — no cross-axis confusion.

### 5. Why a convolutional code?
Without error correction, one flipped bit = one wrong character. At r ≈ 1.56 and σ²=1, raw BPSK BER is around 5% — nearly guaranteed errors across 240 bits. The K=3 rate-1/2 convolutional code (generators G1=7, G2=5 octal) adds structured redundancy: 240 info bits become 484 coded bits. The decoder (Viterbi) uses all 484 received values jointly to correct errors, giving BER around 10⁻⁴ and P(perfect 240-bit decode) ≈ 94%.

Simple 2× repetition would give roughly 4% perfect messages for the same energy. The conv code gives 94% for free.

### 6. Why r = sqrt(1200 / 492)?
The energy constraint is ‖x‖² ≤ 1200. The frame has 492 non-zero values:
- 8 pilot pairs of (r, 0) → 8 values of magnitude r
- 484 BPSK samples → 484 values of magnitude r

So 492 · r² = 1200, giving r = sqrt(1200/492) ≈ 1.562. This uses the full energy budget, which maximises r and minimises BER.

### 7. Why terminate the convolutional encoder with 2 zero bits?
The Viterbi decoder does a traceback starting from the final trellis state. If the encoder ends in an unknown state, the traceback anchor is wrong and the last few decoded bits are unreliable. Appending 2 zero bits flushes the shift register back to state 0, so the decoder knows exactly where to start the traceback.

### 8. What can go wrong?
Two failure modes, same as the decoder side:
1. Wrong Ti detection (probability ≈ 0 with 8 pilots): entire message is garbled
2. Conv code + noise (probability ≈ 6% per message): 1–3 wrong characters scattered randomly

At 89% success per attempt and two demo attempts, P(failing both) ≈ 1.2%.
