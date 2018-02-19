# Pytorch Implementation of SEGAN (Speech Enhancement GAN)
Implementation of [SEGAN](https://arxiv.org/abs/1703.09452) by Pascual et al. in 2017, using pytorch.
Original Tensorflow version can be found [here](https://github.com/santi-pdp/segan).

## Prerequisites

- python v3.5.2 or higher
- pytorch v0.3.0 (other versions not tested)
- CUDA preferred
- noisy speech dataset downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/2791)
- libraries specified in `requirements.txt`

### Installing Libraries

`pip install -r requirements.txt`

## Data Preprocessing

Use `data_preprocess.py` file to preprocess downloaded data. 
Adjust the file paths at the beginning of the file to properly locate the data files, output folder, etc.
Uncomment functions in `__main__` to perform desired preprocessing stage.

Data preprocessing consists of three main stages:
1. Downsampling - downsample original audio files (48k) to sampling rate of 16000.
2. Serialization - Splitting the audio files into 2^14-sample (about 1 second) snippets.
3. Verification - whether it contains proper number of samples.

Note that the second stage takes a fairly long time - more than an hour.

## Training

`python model.py`

Again, fix and adjust datapaths in `model.py` according to your needs.
Especially, provide accurate path to where serialized data are stored.
