# PDC Transmitter — Person 2 (Encoder)

## Quick start

```bash
pip install numpy
python encode.py
python transmit.py "Hello this is a PDC test message ok.    "
```

## Official server client (`client.py` from Moodle)

Your `encode()` output is compatible with the provided client:

| `client.py` check | Your encoder |
|-------------------|--------------|
| `.txt`, `loadtxt` → 1D float vector | `write_channel_input()` / `np.savetxt` |
| `N_sample <= 500` | `n = 500` |
| `sum(x**2) <= 1200` | `‖x‖² = 1200` exactly |
| real-valued | BPSK + pilots |

**Demo workflow (Person 2):**

1. `python transmit.py "<40-char message>"` → writes `input.txt`
2. On EPFL network (or VPN), run **`../Python client/client.py`**
3. Give `output.txt` to Person 3 → `decode(loadtxt(output.txt))`

Notes from the client docstring:

- Max **1 M samples** on the wire (you use 500).
- **30 s** between connections — plan only two demo transmissions.
- Rate-limit / version errors come back as header `b'1'` / `b'2'` — use the Moodle `client.py` as-is.

## API

```python
from encode import encode, check_constraints, write_channel_input

message = "Hello this is a PDC test message ok."  # exactly 40 chars
x = encode(message)
ok, info = check_constraints(x)
write_channel_input(x, "input.txt")  # for EPFL client.py
```

## Frame design

| Section | Indices | Content |
|---------|---------|---------|
| Pilots | 0–15 | 8× `(r, 0)` |
| Data | 16–499 | 484 BPSK-coded bits (K=3 conv, rate 1/2) |

- `n = 500` (even, ≤ 500)
- `‖x‖² = 1200` with `r = √(1200/492)`

## Test with the decoder (without changing Decoding/)

`Decoding/test_harness.py` keeps its own `stub_encode` simulator for the receiver team.
You test your real encoder separately:

```bash
cd Encoding
python test_encode.py          # encoder-only
python test_with_decoder.py    # your encode + their decode + channel
python test_with_decoder.py --mc --trials 1000
```

Coordinate with the receiver on `ALPHABET`, frame layout, conv code, and BPSK signs.
