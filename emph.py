import numpy as np
from scipy import signal


def pre_emphasis(signal_batch, emph_coeff=0.95) -> np.array:
    """
    Pre-emphasis of higher frequencies given a batch of signal.

    Args:
        signal_batch(np.array): batch of signals, represented as numpy arrays
        emph_coeff(float): emphasis coefficient

    Returns:
        result: pre-emphasized signal batch
    """
    return signal.lfilter([1, -emph_coeff], [1], signal_batch)


def de_emphasis(signal_batch, emph_coeff=0.95) -> np.array:
    """
    De-emphasis operation given a batch of signal.
    Reverts the pre-emphasized signal.

    Args:
        signal_batch(np.array): batch of signals, represented as numpy arrays
        emph_coeff(float): emphasis coefficient

    Returns:
        result: de-emphasized signal batch
    """
    return signal.lfilter([1], [1, -emph_coeff], signal_batch)
