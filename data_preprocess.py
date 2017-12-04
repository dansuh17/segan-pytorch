import os
import subprocess
import librosa
import numpy as np
import time


data_path = '../data/segan'
clean_train_foldername = 'clean_trainset_wav/clean_trainset_56spk_wav'
noisy_train_foldername = 'noisy_trainset_wav/noisy_trainset_56spk_wav'
out_clean_train_fdrnm = 'clean_trainset_wav_16k'
out_noisy_train_fdrnm = 'noisy_trainset_wav_16k'
ser_data_fdrnm = 'ser_data'

def data_verify():
    ser_data_path = os.path.join(data_path, ser_data_fdrnm)
    for dirname, dirs, files in os.walk(ser_data_path):
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
    if not os.path.exists(os.path.join(data_path, out_clean_train_fdrnm)):
        os.makedirs(os.path.join(data_path, out_clean_train_fdrnm))

    for dirname, dirs, files in os.walk(os.path.join(data_path, clean_train_foldername)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            output_folderpath = os.path.join(data_path, out_clean_train_fdrnm)
            # use sox to down-sample to 16k
            print('Downsampling : {}'.format(input_filepath))
            completed_process = subprocess.run(
                    'sox {} -r 16k {}'
                    .format(input_filepath, os.path.join(output_folderpath, filename)),
                    shell=True, check=True)

    # noisy training sets
    if not os.path.exists(os.path.join(data_path, out_noisy_train_fdrnm)):
        os.makedirs(os.path.join(data_path, out_noisy_train_fdrnm))

    for dirname, dirs, files in os.walk(os.path.join(data_path, noisy_train_foldername)):
        for filename in files:
            input_filepath = os.path.abspath(os.path.join(dirname, filename))
            output_folderpath = os.path.join(data_path, out_noisy_train_fdrnm)
            # use sox to down-sample to 16k
            print('Processing : {}'.format(input_filepath))
            completed_process = subprocess.run(
                    'sox {} -r 16k {}'
                    .format(input_filepath, os.path.join(output_folderpath, filename)),
                    shell=True, check=True)


def slice_signal(filepath, window_size, stride=0.5):
    wav, sr = librosa.load(filepath)
    n_samples = wav.shape[0]  # contains simple amplitudes
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize():
    start_time = time.time()  # measure the time
    window_size = 2 ** 14  # about 1 second of samples
    dst_folder = os.path.join(data_path, ser_data_fdrnm)

    if not os.path.exists(dst_folder):
        print('Creating new destination folder for new data')
        os.makedirs(dst_folder)

    clean_data_path = os.path.join(data_path, clean_train_foldername)
    noisy_data_path = os.path.join(data_path, noisy_train_foldername)

    for dirname, dirs, files in os.walk(clean_data_path):
        if len(files) == 0:
            continue
        for filename in files:
            print('Splitting : {}'.format(filename))
            clean_filepath = os.path.join(clean_data_path, filename)
            noisy_filepath = os.path.join(noisy_data_path, filename)

            clean_sliced = slice_signal(clean_filepath, window_size)
            noisy_sliced = slice_signal(noisy_filepath, window_size)

            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                pair = np.array([slice_tuple[0], slice_tuple[1]])
                np.save(os.path.join(dst_folder, '{}_{}'.format(filename, idx)), arr=pair)

    end_time = time.time()
    print('Total elapsed time for prerpocessing : {}'.format(end_time - start_time))



if __name__ == '__main__':
    # downsample_16k()
    # process_and_serialize()  #!! takes very long time - about 5000s
    data_verify()
