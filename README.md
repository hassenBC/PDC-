# PDC Project

| Folder | Owner | Contents |
|--------|-------|----------|
| `Encoding/` | Person 2 | `encode()`, tests |
| `Decoding/` | Person 3 | `decode()`, channel sim, tests |
| `Decoding/` | Person 3 | `decode()`, `demo.py`, `input.txt`, `output.txt` (demo I/O) |
| `Python client/` | **Shared** | Moodle `client.py` + `channel_helper.py` |

## Channel client (EPFL)

```text
encode → Decoding/input.txt → client.py → Decoding/output.txt → decode
```

Person 2: `python encode.py "..."` from `Encoding/` (writes `Decoding/input.txt`).

Person 3: `python demo.py decode` from `Decoding/` (after server step).

Requires EPFL network or VPN; wait 30 s between server calls.
