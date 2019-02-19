import os
import subprocess
import librosa
import numpy as np
import time


"""
Audio data preprocessing for SEGAN training.

It provides:
    1. 16k downsampling (sox required)
    2. slicing and serializing
    3. verifying serialized data
"""


# specify the paths - modify the paths at your will
DATA_ROOT_DIR = '../data/segan'  # the base folder for dataset
CLEAN_TRAIN_DIR = 'clean_trainset_56spk_wav'  # where original clean train data exist
NOISY_TRAIN_DIR = 'noisy_trainset_56spk_wav'  # where original noisy train data exist
DST_CLEAN_TRAIN_DIR = 'clean_trainset_wav_16k'  # clean preprocessed data folder
DST_NOISY_TRAIN_DIR = 'noisy_trainset_wav_16k'  # noisy preprocessed data folder
SER_DATA_DIR = 'ser_data'  # serialized data folder
SER_DST_PATH = os.path.join(DATA_ROOT_DIR, SER_DATA_DIR)


def verify_data():
    """
    Verifies the length of each data after preprocessing.
    """
    for dirname, dirs, files in os.walk(SER_DST_PATH):
        for filename in files:
            data_pair = np.load(os.path.join(dirname, filename))
            if data_pair.shape[1] != 16384:
                print('Snippet length not 16384 : {} instead'.format(data_pair.shape[1]))
                break


def downsample_16k():
    """
    Convert all audio files to have sampling rate 16k.
    """
    # clean training sets
    dst_clean_dir = os.path.join(DATA_ROOT_DIR, DST_CLEAN_TRAIN_DIR)
    if not os.path.exists(dst_clean_dir):
        os.makedirs(dst_clean_dir)

    for dirname, dirs, files in os.walk(os.path.join(DATA_ROOT_DIR, CLEAN_TRAIN_DIR)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            out_filepath = os.path.join(dst_clean_dir, filename)
            # use sox to down-sample to 16k
            print('Downsampling : {}'.format(input_filepath))
            subprocess.run(
                    'sox {} -r 16k {}'
                    .format(input_filepath, out_filepath),
                    shell=True, check=True)

    # noisy training sets
    dst_noisy_dir = os.path.join(DATA_ROOT_DIR, DST_NOISY_TRAIN_DIR)
    if not os.path.exists(dst_noisy_dir):
        os.makedirs(dst_noisy_dir)

    for dirname, dirs, files in os.walk(os.path.join(DATA_ROOT_DIR, NOISY_TRAIN_DIR)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            out_filepath = os.path.join(dst_noisy_dir, filename)
            # use sox to down-sample to 16k
            print('Processing : {}'.format(input_filepath))
            subprocess.run(
                    'sox {} -r 16k {}'
                    .format(input_filepath, out_filepath),
                    shell=True, check=True)


def slice_signal(filepath, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size with [stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(filepath, sr=sample_rate)
    n_samples = wav.shape[0]  # contains simple amplitudes
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize():
    """
    Serialize the sliced signals and save on separate folder.
    """
    start_time = time.time()  # measure the time
    window_size = 2 ** 14  # about 1 second of samples
    sample_rate = 16000
    stride = 0.5

    if not os.path.exists(SER_DST_PATH):
        print('Creating new destination folder for new data')
        os.makedirs(SER_DST_PATH)

    # the path for source data (16k downsampled)
    clean_data_path = os.path.join(DATA_ROOT_DIR, DST_CLEAN_TRAIN_DIR)
    noisy_data_path = os.path.join(DATA_ROOT_DIR, DST_NOISY_TRAIN_DIR)

    # walk through the path, slice the audio file, and save the serialized result
    for dirname, dirs, files in os.walk(clean_data_path):
        if len(files) == 0:
            continue
        for filename in files:
            print('Splitting : {}'.format(filename))
            clean_filepath = os.path.join(clean_data_path, filename)
            noisy_filepath = os.path.join(noisy_data_path, filename)

            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_filepath, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_filepath, window_size, stride, sample_rate)

            # serialize - file format goes [original_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(SER_DST_PATH, '{}_{}'.format(filename, idx)), arr=pair)

    # measure the time it took to process
    end_time = time.time()
    print('Total elapsed time for preprocessing : {}'.format(end_time - start_time))


if __name__ == '__main__':
    """
    Uncomment each function call that suits your needs.
    """
    # downsample_16k()
    # process_and_serialize()  # WARNING - takes very long time
    # verify_data()
