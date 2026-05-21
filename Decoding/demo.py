"""
demo.py — Live Demo Runner (May 29th)
======================================
This is what you run during the presentation.

USAGE
─────
Step 1: Transmit a message to the server
    python demo.py transmit "Your 40 char message here padded ok"

Step 2: Decode the server's response
    python demo.py decode output.txt

Or do both in one command (requires client.py in the same directory):
    python demo.py run "Your 40 char message here padded ok"

WORKFLOW ON THE DAY
───────────────────
1. Professor gives you the 40-character message
2. Run:  python demo.py run "<message>"
3. The script:
   a. Calls encode() → writes x to input.txt
   b. Calls client.py → sends x to server, gets y in output.txt
   c. Calls decode(y) → prints the recovered message
   d. Shows confidence diagnostics
4. If message is wrong, run again (you have 2 attempts)
"""

import sys
import os
import subprocess
import numpy as np

from decode       import decode, ALPHABET
from decode_robust import decode_with_confidence
from test_harness  import stub_encode   # Replace with Person 2's encode()

# ── Server configuration ──────────────────────────────────────────────────────
SERVER_HOST = 'iscsrv72.epfl.ch'
SERVER_PORT = 80


def pad_message(message):
    """Ensure message is exactly 40 characters."""
    msg = (message + ' ' * 40)[:40]
    if len(message) > 40:
        print(f"⚠ Message truncated to 40 characters: '{msg}'")
    return msg


def validate_message(message):
    """Check all characters are in the allowed alphabet."""
    invalid = [c for c in message if c not in ALPHABET]
    if invalid:
        print(f"Invalid characters: {invalid}")
        print(f"   Allowed: a-z, A-Z, 0-9, space, period")
        sys.exit(1)


def transmit(message):
    """
    Encode message and write to input.txt for client.py.
    Returns the encoded x vector.
    """
    message = pad_message(message)
    validate_message(message)

    print(f"Encoding message: '{message}'")

    # Use Person 2's encode() in production
    # Replace stub_encode with: from encode import encode
    x = stub_encode(message)

    energy = float(np.sum(x**2))
    print(f"   Vector length: {len(x)}")
    print(f"   Energy ‖x‖²:  {energy:.4f} / 1200.00")

    # Write to input.txt (format: one number per line)
    with open('input.txt', 'w') as f:
        for val in x:
            f.write(f"{val:.10f}\n")
    print(f"   Written to input.txt")

    return x, message


def call_server():
    """
    Call client.py to send input.txt to the server and receive output.txt.
    """
    print(f"\n📡 Sending to server ({SERVER_HOST}:{SERVER_PORT})...")
    cmd = [
        sys.executable, 'client.py',
        '--input_file',  'input.txt',
        '--output_file', 'output.txt',
        f'--srv_hostname={SERVER_HOST}',
        f'--srv_port={SERVER_PORT}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Server error:\n{result.stderr}")
        sys.exit(1)
    print(f"   Received output.txt")


def load_y(filepath='output.txt'):
    """Load the server's Y vector from a text file (one number per line)."""
    with open(filepath, 'r') as f:
        y = [float(line.strip()) for line in f if line.strip()]
    return y


def run_decode(y_filepath='output.txt', original_message=None):
    """
    Load Y from file, decode, and print results with diagnostics.
    """
    y = load_y(y_filepath)
    print(f"\n🔍 Decoding Y vector ({len(y)} values)...")

    result = decode_with_confidence(y)

    print(f"\n{'='*50}")
    print(f"  DECODED MESSAGE:")
    print(f"  '{result['message']}'")
    print(f"{'='*50}")
    print(f"  Detected rotation: T{result['i_hat']}")
    print(f"  Confidence gap:    {result['confidence']['gap']:.3f}")

    if result['warning']:
        print(f"\n  {result['warning']}")
    else:
        print(f" Rotation detection: high confidence")

    if original_message:
        original_message = pad_message(original_message)
        errors = sum(a != b for a, b in zip(original_message, result['message']))
        if errors == 0:
            print(f"\n  PERFECT DECODING — 0 character errors")
        else:
            print(f"\n  {errors} character error(s)")
            for i, (o, d) in enumerate(zip(original_message, result['message'])):
                if o != d:
                    print(f"     Position {i}: expected '{o}', got '{d}'")

    return result


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == 'transmit':
        if len(sys.argv) < 3:
            print("Usage: python demo.py transmit <message>")
            sys.exit(1)
        message = sys.argv[2]
        transmit(message)
        print("\nNow run client.py manually, then:")
        print("  python demo.py decode output.txt")

    elif command == 'decode':
        filepath = sys.argv[2] if len(sys.argv) > 2 else 'output.txt'
        original = sys.argv[3] if len(sys.argv) > 3 else None
        run_decode(filepath, original_message=original)

    elif command == 'run':
        # Full pipeline: encode → server → decode
        if len(sys.argv) < 3:
            print("Usage: python demo.py run <message>")
            sys.exit(1)
        message = sys.argv[2]
        x, padded = transmit(message)
        call_server()
        run_decode('output.txt', original_message=padded)

    elif command == 'test':
        # Quick local test using channel simulator (no server)
        message = sys.argv[2] if len(sys.argv) > 2 else 'Test message for local decode check.'
        from channel_sim import channel as local_channel

        message = pad_message(message)
        validate_message(message)
        x = stub_encode(message)

        print(f"\n🧪 Local test (no server)")
        print(f"   Message: '{message}'")
        y = local_channel(x)
        result = decode_with_confidence(y)

        print(f"   Decoded: '{result['message']}'")
        errors = sum(a != b for a, b in zip(message, result['message']))
        print(f"   Errors:  {errors}")
        print(f"   T_i:     T{result['i_hat']}")
        print(f"   Conf:    {result['confidence']['gap']:.3f}")
        if result['warning']:
            print(f"   {result['warning']}")

    else:
        print(f"Unknown command: {command}")
        print("Commands: transmit, decode, run, test")
        sys.exit(1)
