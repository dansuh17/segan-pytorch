# Pytorch Implementation of SEGAN (Speech Enhancement GAN)
Implementation of [SEGAN](https://arxiv.org/abs/1703.09452) by Pascual et al., using pytorch.
Original Tensorflow version can be found [here](https://github.com/santi-pdp/segan).

## Prerequisites

- python v3.5.2 or higher
- pytorch v0.3.0
- CUDA preferred
- dataset downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/2791)
- libraries specified in `requirements.txt`

### Installing Libraries

`pip install -r requirements.txt`

## Data Preprocessing

Use `data_preprocess.py` file to preprocess downloaded data. Uncomment functions in `__main__` to perform desired preprocessing stage.

## Training

`python model.py`

Again, fix datapaths in `model.py` by your needs.
