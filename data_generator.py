import torch
from torch.utils import data
import numpy as np
import os


class AudioSampleGenerator(data.Dataset):
    """
    Audio sample reader.
    Used alongside with DataLoader class to generate batches.
    see: http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset
    """
    SAMPLE_LENGTH = 16384

    def __init__(self, data_folder_path: str):
        if not os.path.exists(data_folder_path):
            raise FileNotFoundError

        # store full paths - not the actual files.
        # all files cannot be loaded up to memory due to its large size.
        # insted, we read from files upon fetching batches (see __getitem__() implementation)
        self.filepaths = [os.path.join(data_folder_path, filename)
                for filename in os.listdir(data_folder_path)]
        self.num_data = len(self.filepaths)

    def reference_batch(self, batch_size: int):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.

        Args:
            batch_size(int): batch size

        Returns:
            ref_batch: reference batch
        """
        ref_filenames = np.random.choice(self.filepaths, batch_size)
        ref_batch = torch.from_numpy(np.stack([np.load(f) for f in ref_filenames]))
        return ref_batch

    def fixed_test_audio(self, num_test_audio: int):
        """
        Randomly chosen batch for testing generated results.

        Args:
            num_test_audio(int): number of test audio.
                Must be same as batch size of training,
                otherwise it cannot go through the forward step of generator.
        """
        test_filenames = np.random.choice(self.filepaths, num_test_audio)
        # stack the data for all test audios
        test_audios = np.stack([np.load(f) for f in test_filenames])
        test_clean_set = test_audios[:, 0].reshape((num_test_audio, 1, self.SAMPLE_LENGTH))
        test_noisy_set = test_audios[:, 1].reshape((num_test_audio, 1, self.SAMPLE_LENGTH))
        # file names of test samples
        test_basenames = [os.path.basename(fpath) for fpath in test_filenames]
        return test_basenames, test_clean_set, test_noisy_set

    def __getitem__(self, idx):
        # get item for specified index
        pair = np.load(self.filepaths[idx])
        return pair

    def __len__(self):
        return self.num_data

