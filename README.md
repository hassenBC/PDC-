# PDC Project

| Folder | Owner | Contents |
|--------|-------|----------|
| `Encoding/` | Person 2 | `encode()`, `transmit.py`, tests |
| `Decoding/` | Person 3 | `decode()`, channel sim, demo |
| `Python client/` | **Shared** | Moodle `client.py` + `channel_helper.py` |

## Channel client (EPFL)

```text
encode (Encoding) → input.txt → client (Python client) → output.txt → decode (Decoding)
```

From `Encoding/` after `transmit.py`:

```bash
python "../Python client/client.py" --input_file=input.txt --output_file=output.txt --srv_hostname=iscsrv72.epfl.ch --srv_port=80
```

Requires EPFL network or VPN; wait 30 s between server calls.
