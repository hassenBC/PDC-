"""
Prepare input.txt for the official EPFL client.py (Person 2 demo step).

Usage
-----
  python transmit.py "Your forty character message here ok.     "
  python transmit.py "Your forty character message here ok.     " -o input.txt

Then run the shared client (../Python client/):

  python "../Python client/client.py" --input_file=input.txt --output_file=output.txt ^
      --srv_hostname=iscsrv72.epfl.ch --srv_port=80

Hand output.txt to the decoder team for decode(y).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from encode import MESSAGE_LEN, encode, validate_for_client, write_channel_input

_CLIENT_DIR = Path(__file__).resolve().parent.parent / "Python client"
_CLIENT = _CLIENT_DIR / "client.py"


def pad_message(message: str) -> str:
    if len(message) > MESSAGE_LEN:
        print(f"Warning: truncating to {MESSAGE_LEN} characters.")
    return (message + " " * MESSAGE_LEN)[:MESSAGE_LEN]


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode message -> input.txt for client.py")
    parser.add_argument("message", nargs="?", help=f"Text message (padded/truncated to {MESSAGE_LEN} chars)")
    parser.add_argument("-o", "--output", type=Path, default=Path("input.txt"), help="Output .txt path")
    args = parser.parse_args()

    if args.message is None:
        args.message = input(f"Enter message ({MESSAGE_LEN} chars): ").strip()

    message = pad_message(args.message)
    x = encode(message)
    validate_for_client(x)

    write_channel_input(x, str(args.output))

    energy = sum(v * v for v in x)
    print(f"Message:  {message!r}")
    print(f"Wrote:    {args.output.resolve()}")
    print(f"Samples:  {len(x)}")
    print(f"Energy:   {energy:.6f} / 1200")
    out_y = args.output.parent / "output.txt"
    print()
    if not _CLIENT.is_file():
        print(f"Put Moodle client.py + channel_helper.py in: {_CLIENT_DIR}")
    print("Next (EPFL network / VPN), run shared client:")
    print(
        f"  python {_CLIENT} --input_file={args.output.resolve()} "
        f"--output_file={out_y.resolve()} "
        "--srv_hostname=iscsrv72.epfl.ch --srv_port=80"
    )


if __name__ == "__main__":
    main()
