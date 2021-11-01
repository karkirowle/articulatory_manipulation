import numpy as np
from nnmnkwii import paramgen


windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0]))]


def static_delta_delta_to_static(static_delta_delta: np.ndarray):
    """
    Converts a normalised static+delta+delta feature to static using maximum likelihood parameter
    generation.

    The returned feature still has to be unnormalised, but it is suitable for the evaluation loop.

    :param static_delta_delta:
    :return:
    """
    #print(static_delta_delta.shape)
    variance_frames = np.ones_like(static_delta_delta)
    static_features = paramgen.mlpg(static_delta_delta, variance_frames, windows)

    #static_features = static_features * std[:60] + mean[:60]

    return static_features