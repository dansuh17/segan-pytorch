import numpy as np


def pre_emphasis(signal_batch, emph_coeff=0.9):
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            result[sample_idx][ch] = np.append(
                    channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])
    return result


def de_emphasis(signal_batch, emph_coeff=0.9):
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            result[sample_idx][ch] = np.append(
                    channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])
    return result
