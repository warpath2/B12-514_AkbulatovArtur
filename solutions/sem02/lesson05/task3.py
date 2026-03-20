import numpy as np


class ShapeMismatchError(Exception):
    pass


def adaptive_filter(
    Vs: np.ndarray,
    Vj: np.ndarray,
    diag_A: np.ndarray,
) -> np.ndarray:
    M, N = Vs.shape
    M2, K = Vj.shape
    K2 = len(diag_A)

    if M != M2 or K != K2:
        raise ShapeMismatchError

    I = np.eye(M)
    A = np.diag(diag_A)
    Vj_H = Vj.conj().T
    inner = Vj_H @ Vj @ A
    I_K = np.eye(K)
    to_invert = I_K + inner
    inverted = np.linalg.inv(to_invert)
    R_inv = I - Vj @ inverted @ Vj_H
    y = R_inv @ Vs
    return y
