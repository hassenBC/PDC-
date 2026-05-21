# PDC Decoder — Person 3 Implementation

## File Structure

```
decoder/
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

pip install numpy scipy


python test_harness.py


python test_harness.py --mc --trials 1000 --ber


python error_analysis.py


python demo.py test "Hello this is the test message ok."
```

## Integration with Person 2 (Encoder)

The test harness uses a `stub_encode()` function until Person 2's real encoder
is ready. Once you have it:

1. Open `test_harness.py`
2. Replace:
   ```python
   from test_harness import stub_encode
   ```
   with:
   ```python
   from encode import encode as stub_encode
   ```
3. Verify the alphabet order matches: `'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789 .'`

## Design Choices

### 1. Why pilot symbols?
Theory Problem 3(a) proves that without pilots, if the code is symmetric
(−c is also a codeword), the error probability is ≥ 1/2. QPSK is symmetric,
so we can't do better than random guessing without knowing T_i. The pilots
(4 copies of (r,0)) solve this completely by letting the receiver identify T_i
before decoding the data.

### 2. Why (r, 0) as the pilot symbol?
When each T_i is applied, (r, 0) maps to a different axis:
- T1 → ( r,  0)   positive real axis
- T2 → ( 0,  r)   positive imaginary axis
- T3 → (-r,  0)   negative real axis
- T4 → ( 0, -r)   negative imaginary axis

These 4 outcomes are orthogonal and maximally separated.
The minimum distance between any two is 2r√4 ≈ 6.3, giving very reliable detection.

### 3. Why minimum squared distance (ML) for rotation detection?
The received pilot is T_i(pilot) + Z where Z ~ N(0, I_8).
The likelihood of observing y_pilot given hypothesis i is:
    p(y_pilot | i) ∝ exp(-‖y_pilot - candidate_i‖² / 2)
Maximising likelihood = minimising squared distance.
This is the Bayes-optimal rule when all 4 T_i are equally likely.

### 4. Why QPSK and not higher-order modulation (16-QAM etc.)?
At the available SNR (s² ≈ 2.48 per component), QPSK has BER ≈ 10⁻³.
16-QAM would require ~6 dB more SNR for the same BER — we don't have the energy budget.
QPSK is optimal here.

### 5. Why send the data twice?
Summing two independent noisy copies doubles the effective SNR (3 dB gain):
    signal: s + s = 2s (doubles)
    noise:  N(0,1) + N(0,1) = N(0,2)  (grows by √2)
    SNR gain: (2s)²/2 ÷ s²/1 = 2×
This is the maximum gain achievable from 2 independent copies.
The 488-element vector fits in the n ≤ 500 constraint: 8 (pilots) + 240×2 (data) = 488.

### 6. Why sum the copies rather than average them?
For sign-based QPSK decoding, sum and average give identical bit decisions.
We use sum because it's one less operation and matches the spec.

### 7. Why sign-based QPSK decoding?
QPSK encodes 2 bits per pair (s1, s2) where s1, s2 ∈ {+s, -s}.
With Gaussian noise, the ML decision for b1 is: b1 = 0 if y1 ≥ 0, else 1.
This is because p(y1 | s1=+s) > p(y1 | s1=-s) iff y1 > 0.
The two bits are decoded independently — no cross-component computation needed.

### 8. What can go wrong?
Two failure modes:
1. Wrong T_i detection (probability ~10⁻⁵): entire message is garbled (≈ 20 wrong chars)
2. QPSK noise (probability ~BER per bit): 0-5 wrong characters randomly scattered

The confidence_gap metric in decode_with_confidence() lets you detect case 1 before
presenting the result. If gap < 2.0, the detection was borderline — consider retrying.


