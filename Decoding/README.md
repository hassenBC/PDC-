# PDC Decoder — Person 3 Implementation

## File Structure

```
Decoding/
├── decode.py          Core decode(y) → message function
├── decode_robust.py   Enhanced decoder with confidence scoring
├── channel_sim.py     Local channel simulator (for testing without server)
├── test_harness.py    Unit tests + Monte Carlo reliability evaluation
├── error_analysis.py  BER analysis, error classification, grade estimation
├── demo.py            Live demo runner for May 29th
└── README.md          This file
```

## Quick Start

```bash
# Install dependencies
pip install numpy scipy

# Run unit tests
python test_harness.py

# Monte Carlo reliability evaluation
python test_harness.py --mc --trials 1000 --ber

# BER / error analysis
python error_analysis.py

# Local end-to-end test (no server or VPN needed)
python demo.py test "Hello this is the test message ok."

# Full pipeline via EPFL server (must be on EPFL network or VPN)
python demo.py run "Your 40 character message goes here  "
```

## Demo Commands (May 29th)

| Command | What it does |
|---------|--------------|
| `python demo.py run "<message>"` | Full pipeline: encode → server → decode |
| `python demo.py transmit "<message>"` | Encode only, write `input.txt` |
| `python demo.py decode [output.txt]` | Decode only from an existing `output.txt` |
| `python demo.py test "<message>"` | Local simulation, no server needed |

Messages must be exactly 40 characters (auto-padded with spaces).  
Allowed characters: `a-z A-Z 0-9 space period`.  
Wait 30 s between server calls (rate limit).

## Signal Structure

The transmitted vector `x` has 500 elements:

```
x[0:16]    8 pilot pairs  (16 values, 8 non-zero at ±r)
x[16:500]  484 BPSK data  (484 non-zero), one coded bit per sample
```

Energy budget: `‖x‖² = 492 · r² = 1200`, so `r = √(1200/492) ≈ 1.562`.

## Integration with Encoder

The encoder (`Encoding/encode.py`) is fully integrated. `demo.py` imports it
directly — no stub needed. Both sides use the same alphabet order and the same
convolutional code parameters (K=3, G1=7, G2=5 octal).

## Design Choices

### 1. Why pilot symbols?
Theory Problem 3(a) proves that without pilots, if the code is symmetric
(−c is also a codeword), the error probability is ≥ 1/2. BPSK is symmetric,
so without knowing T_i we can't do better than random guessing. Eight pilot
pairs `(r, 0)` let the receiver identify T_i before touching the data.

### 2. Why `(r, 0)` as the pilot symbol?
When each T_i is applied, `(r, 0)` maps to a different axis:
- T1 → `( r,  0)`  positive real axis
- T2 → `( 0,  r)`  positive imaginary axis
- T3 → `(-r,  0)`  negative real axis
- T4 → `( 0, -r)`  negative imaginary axis

These four outcomes are orthogonal and maximally separated.

### 3. Why 8 pilot pairs?
Inter-candidate distance² = `16r² ≈ 39`. P(wrong T_i) = Q(d/2) ≈ Q(3.12) ≈ 0.09%.
With only 4 pairs the rate would be ~15× higher. 8 pairs is the sweet spot before
the marginal data-energy cost starts to dominate.

### 4. Why minimum squared distance (ML) for rotation detection?
The received pilot is `y_pilot = T_i(pilot) + Z`, `Z ~ N(0, I_8)`.
Maximising the likelihood over i is equivalent to minimising `‖y_pilot − candidate_i‖²`.
This is Bayes-optimal when all four T_i are equally likely.

### 5. Why a convolutional code instead of repetition?
The old rate-1/2 repetition scheme (send 240 bits twice) has minimum distance d=2
and gives BER ≈ 1.3%, so P(perfect 240-bit decode) ≈ 4%.

The K=3 convolutional code (generators G1=7, G2=5 octal) has free distance d_free=5.
Soft Viterbi at Eb/N0 ≈ 3.87 dB gives BER ≈ 2.4×10⁻⁴, pushing
P(perfect 240-bit decode) ≈ 94%. Same channel uses, same energy budget — purely
an algorithmic upgrade.

### 6. Why soft-decision Viterbi?
The decoder operates on post-rotation-inversion real values directly, without
thresholding to bits first. The branch metric is the correlation between the
received pair and the candidate transmitted pair. Soft decoding gives ~2 dB
extra coding gain over hard-decision Viterbi.

### 7. Why BPSK and not QPSK or higher-order modulation?
At the available SNR (s² ≈ 2.44 per component), BPSK paired with the
convolutional code achieves ~94% perfect-decode probability. Higher-order
modulation (QPSK, 16-QAM) would require more SNR for equivalent BER and
offer no advantage here given the energy constraint.

### 8. What can go wrong?
Two failure modes:
1. Wrong T_i detection (probability ~10⁻⁵): entire message is garbled (~20 wrong chars)
2. Convolutional decoder residual errors (probability ~6%): 0–3 wrong characters

The `confidence_gap` metric in `decode_with_confidence()` lets you detect case 1
before presenting the result. If gap < 4.0, the detection was borderline — retry.
