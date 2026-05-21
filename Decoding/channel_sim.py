"""
channel_sim.py — Local Channel Simulator
=========================================
Exact replica of the server-side channel function from the project spec.
Use this for all local testing. The server throttles to 1 request / 30 seconds
so you cannot run 1000-trial Monte Carlo tests against it.

WHY replicate locally?
──────────────────────
The server is a black box over the network. Local simulation lets you:
  - Run thousands of trials in seconds
  - Control the random seed for reproducible tests
  - Test edge cases (e.g. force a specific T_i)
  - Develop and debug without EPFL VPN access

The channel function below is copied VERBATIM from the project handout
(Handout 21, page 4). Do NOT modify it.
"""

import numpy as np


def channel(x):
    """
    Simulate the PDC project channel:  Y = T_i(x) + Z

    Picks T_i uniformly at random from {T1, T2, T3, T4} and applies it
    pair-wise to x, then adds i.i.d. standard Gaussian noise.

    Parameters
    ----------
    x : array-like of even length n ≤ 500
        Must satisfy ‖x‖² ≤ 1200 (the server enforces this; we don't here,
        but violating it means your transmitter is broken, not this function).

    Returns
    -------
    np.ndarray of length n
        The noisy, rotated channel output Y.
    """
    x = np.asarray(x, dtype=float)
    if x.size % 2 != 0:
        raise ValueError(f"Input length must be even, got {x.size}.")

    i = int(np.random.randint(1, 5))   # uniform on {1, 2, 3, 4}

    # Reshape into (n/2, 2) so each row is one complex symbol (a, b)
    pairs = x.reshape(-1, 2)
    a, b  = pairs[:, 0], pairs[:, 1]

    if   i == 1: Tx = np.stack([ a,  b], axis=1)
    elif i == 2: Tx = np.stack([-b,  a], axis=1)
    elif i == 3: Tx = np.stack([-a, -b], axis=1)
    else:        Tx = np.stack([ b, -a], axis=1)   # i == 4

    Tx = Tx.reshape(-1)
    z  = np.random.standard_normal(Tx.shape)
    return Tx + z


def channel_fixed_Ti(x, i):
    """
    Same as channel() but forces a specific T_i instead of random selection.

    WHY this variant?
    ─────────────────
    Useful for unit testing: you can check that your decoder correctly handles
    each of the 4 rotations individually before testing the random case.
    Also useful for debugging: if T2 always fails, you know the T2 inverse
    rotation is wrong.

    Parameters
    ----------
    x : array-like
    i : int in {1, 2, 3, 4}

    Returns
    -------
    np.ndarray
    """
    if i not in {1, 2, 3, 4}:
        raise ValueError(f"i must be in {{1,2,3,4}}, got {i}")

    x = np.asarray(x, dtype=float)
    pairs = x.reshape(-1, 2)
    a, b  = pairs[:, 0], pairs[:, 1]

    if   i == 1: Tx = np.stack([ a,  b], axis=1)
    elif i == 2: Tx = np.stack([-b,  a], axis=1)
    elif i == 3: Tx = np.stack([-a, -b], axis=1)
    else:        Tx = np.stack([ b, -a], axis=1)

    Tx = Tx.reshape(-1)
    z  = np.random.standard_normal(Tx.shape)
    return Tx + z


def channel_noiseless(x, i):
    """
    Apply T_i to x with NO noise added.

    WHY?
    ────
    Useful for verifying that:
    1. The rotation inversion in your decoder is correct (should recover x exactly)
    2. The pilot candidates in decode.py match what the channel actually produces

    If decode(channel_noiseless(x, i)) != original_message for any i, there's
    a bug in your rotation inversion logic.
    """
    x = np.asarray(x, dtype=float)
    pairs = x.reshape(-1, 2)
    a, b  = pairs[:, 0], pairs[:, 1]

    if   i == 1: Tx = np.stack([ a,  b], axis=1)
    elif i == 2: Tx = np.stack([-b,  a], axis=1)
    elif i == 3: Tx = np.stack([-a, -b], axis=1)
    else:        Tx = np.stack([ b, -a], axis=1)

    return Tx.reshape(-1)
