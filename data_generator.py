from torch.utils import data
import numpy as np
import os

class AudioSampleGenerator(data.Dataset):
    """
    Audio sample reader.
    """
    def __init__(self, data_folder_path):
        if not os.path.exists(data_folder_path):
            raise Error('The data folder does not exist!')

        # store full paths
        self.filepaths = [os.path.join(data_folder_path, filename)
                for filename in os.listdir(data_folder_path)]
        self.num_data = len(self.filepaths)

    def fixed_test_audio(self, num_test_audio):
        test_filenames = np.random.choice(self.filepaths, num_test_audio)
        test_noisy_set = [np.load(f)[1] for f in test_filenames]
        return test_filenames, np.array(test_noisy_set).reshape(num_test_audio, 1, 16384)

    def __getitem__(self, idx):
        pair = np.load(self.filepaths[idx])
        return pair

    def __len__(self):
        return self.num_data

