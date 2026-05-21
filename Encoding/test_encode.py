"""Standalone encoder tests (no dependency on Decoding/)."""

from encode import (
    ALPHABET,
    MESSAGE_LEN,
    VECTOR_LEN,
    MAX_ENERGY,
    SYMBOL_AMPLITUDE,
    encode,
    check_constraints,
    message_to_bits,
    bits_to_message,
)


def test_round_trip_bits():
    msg = "a" * MESSAGE_LEN
    assert bits_to_message(message_to_bits(msg)) == msg


def test_constraints():
    msg = ("Hello this is a PDC test message ok." + " " * MESSAGE_LEN)[:MESSAGE_LEN]
    x = encode(msg)
    ok, info = check_constraints(x)
    assert ok
    assert info["length"] == VECTOR_LEN
    assert abs(info["energy"] - MAX_ENERGY) < 1e-4


def test_alphabet():
    assert len(ALPHABET) == 64
    assert len(set(ALPHABET)) == 64


if __name__ == "__main__":
    test_alphabet()
    test_round_trip_bits()
    test_constraints()
    print("All encoder tests passed.")
    print(f"r = {SYMBOL_AMPLITUDE:.6f}")
