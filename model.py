"""
Here we define the discriminator and generator for SEGAN,
from paper "Pascual et al. - SEGAN: Speech Enhancement Generative Adversarial Network."
After defining each module, the script also runs the training.

See: https://arxiv.org/abs/1703.09452
"""

import time
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from scipy.io import wavfile
from data_generator import AudioSampleGenerator
from vbnorm import VirtualBatchNorm1d
from tensorboardX import SummaryWriter
import emph

# device we're using
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define folders for output data
in_path = 'segan_data_in'
out_path_root = 'segan_data_out'
ser_data_fdr = 'ser_data'  # serialized data
gen_data_fdr = 'gen_data'  # folder for saving generated data
checkpoint_fdr = 'checkpoint'  # folder for saving models, optimizer states, etc.
tblog_fdr = 'logs'  # summary data for tensorboard
# time info is used to distinguish dfferent training sessions
run_time = time.strftime('%Y%m%d_%H%M', time.gmtime())  # 20180625_1742
# output path - all outputs (generated data, logs, model checkpoints) will be stored here
# the directory structure is as: "[curr_dir]/segan_data_out/[run_time]/"
out_path = os.path.join(os.getcwd(), out_path_root, run_time)
tblog_path = os.path.join(os.getcwd(), tblog_fdr, run_time)  # summary data for tensorboard


# create folder for generated data
gen_data_path = os.path.join(out_path, gen_data_fdr)
if not os.path.exists(gen_data_path):
    os.makedirs(gen_data_path)

# create folder for model checkpoints
checkpoint_path = os.path.join(out_path, checkpoint_fdr)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


class Discriminator(nn.Module):
    """
    Discriminator model of SEGAN.
    """
    def __init__(self, dropout_drop=0.5):
        super().__init__()
        # Define convolution operations.
        # (#input channel, #output channel, kernel_size, stride, padding)
        # in : 16384 x 2
        negative_slope = 0.03
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=31, stride=2, padding=15)   # out : 8192 x 32
        self.vbn1 = VirtualBatchNorm1d(32)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv1d(32, 64, 31, 2, 15)  # 4096 x 64
        self.vbn2 = VirtualBatchNorm1d(64)
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.conv3 = nn.Conv1d(64, 64, 31, 2, 15)  # 2048 x 64
        self.dropout1 = nn.Dropout(dropout_drop)
        self.vbn3 = VirtualBatchNorm1d(64)
        self.lrelu3 = nn.LeakyReLU(negative_slope)
        self.conv4 = nn.Conv1d(64, 128, 31, 2, 15)  # 1024 x 128
        self.vbn4 = VirtualBatchNorm1d(128)
        self.lrelu4 = nn.LeakyReLU(negative_slope)
        self.conv5 = nn.Conv1d(128, 128, 31, 2, 15)  # 512 x 128
        self.vbn5 = VirtualBatchNorm1d(128)
        self.lrelu5 = nn.LeakyReLU(negative_slope)
        self.conv6 = nn.Conv1d(128, 256, 31, 2, 15)  # 256 x 256
        self.dropout2 = nn.Dropout(dropout_drop)
        self.vbn6 = VirtualBatchNorm1d(256)
        self.lrelu6 = nn.LeakyReLU(negative_slope)
        self.conv7 = nn.Conv1d(256, 256, 31, 2, 15)  # 128 x 256
        self.vbn7 = VirtualBatchNorm1d(256)
        self.lrelu7 = nn.LeakyReLU(negative_slope)
        self.conv8 = nn.Conv1d(256, 512, 31, 2, 15)  # 64 x 512
        self.vbn8 = VirtualBatchNorm1d(512)
        self.lrelu8 = nn.LeakyReLU(negative_slope)
        self.conv9 = nn.Conv1d(512, 512, 31, 2, 15)  # 32 x 512
        self.dropout3 = nn.Dropout(dropout_drop)
        self.vbn9 = VirtualBatchNorm1d(512)
        self.lrelu9 = nn.LeakyReLU(negative_slope)
        self.conv10 = nn.Conv1d(512, 1024, 31, 2, 15)  # 16 x 1024
        self.vbn10 = VirtualBatchNorm1d(1024)
        self.lrelu10 = nn.LeakyReLU(negative_slope)
        self.conv11 = nn.Conv1d(1024, 2048, 31, 2, 15)  # 8 x 1024
        self.vbn11 = VirtualBatchNorm1d(2048)
        self.lrelu11 = nn.LeakyReLU(negative_slope)
        # 1x1 size kernel for dimension and parameter reduction
        self.conv_final = nn.Conv1d(2048, 1, kernel_size=1, stride=1)  # 8 x 1
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        self.fully_connected = nn.Linear(in_features=8, out_features=1)  # 1
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, ref_x):
        """
        Forward pass of discriminator.

        Args:
            x: batch
            ref_x: reference batch for virtual batch norm
        """
        # reference pass
        ref_x = self.conv1(ref_x)
        ref_x, mean1, meansq1 = self.vbn1(ref_x, None, None)
        ref_x = self.lrelu1(ref_x)
        ref_x = self.conv2(ref_x)
        ref_x, mean2, meansq2 = self.vbn2(ref_x, None, None)
        ref_x = self.lrelu2(ref_x)
        ref_x = self.conv3(ref_x)
        ref_x = self.dropout1(ref_x)
        ref_x, mean3, meansq3 = self.vbn3(ref_x, None, None)
        ref_x = self.lrelu3(ref_x)
        ref_x = self.conv4(ref_x)
        ref_x, mean4, meansq4 = self.vbn4(ref_x, None, None)
        ref_x = self.lrelu4(ref_x)
        ref_x = self.conv5(ref_x)
        ref_x, mean5, meansq5 = self.vbn5(ref_x, None, None)
        ref_x = self.lrelu5(ref_x)
        ref_x = self.conv6(ref_x)
        ref_x = self.dropout2(ref_x)
        ref_x, mean6, meansq6 = self.vbn6(ref_x, None, None)
        ref_x = self.lrelu6(ref_x)
        ref_x = self.conv7(ref_x)
        ref_x, mean7, meansq7 = self.vbn7(ref_x, None, None)
        ref_x = self.lrelu7(ref_x)
        ref_x = self.conv8(ref_x)
        ref_x, mean8, meansq8 = self.vbn8(ref_x, None, None)
        ref_x = self.lrelu8(ref_x)
        ref_x = self.conv9(ref_x)
        ref_x = self.dropout3(ref_x)
        ref_x, mean9, meansq9 = self.vbn9(ref_x, None, None)
        ref_x = self.lrelu9(ref_x)
        ref_x = self.conv10(ref_x)
        ref_x, mean10, meansq10 = self.vbn10(ref_x, None, None)
        ref_x = self.lrelu10(ref_x)
        ref_x = self.conv11(ref_x)
        ref_x, mean11, meansq11 = self.vbn11(ref_x, None, None)
        # further pass no longer needed

        # train pass
        x = self.conv1(x)
        x, _, _ = self.vbn1(x, mean1, meansq1)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x, _, _ = self.vbn2(x, mean2, meansq2)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        x, _, _ = self.vbn3(x, mean3, meansq3)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x, _, _ = self.vbn4(x, mean4, meansq4)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x, _, _ = self.vbn5(x, mean5, meansq5)
        x = self.lrelu5(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x, _, _ = self.vbn6(x, mean6, meansq6)
        x = self.lrelu6(x)
        x = self.conv7(x)
        x, _, _ = self.vbn7(x, mean7, meansq7)
        x = self.lrelu7(x)
        x = self.conv8(x)
        x, _, _ = self.vbn8(x, mean8, meansq8)
        x = self.lrelu8(x)
        x = self.conv9(x)
        x = self.dropout3(x)
        x, _, _ = self.vbn9(x, mean9, meansq9)
        x = self.lrelu9(x)
        x = self.conv10(x)
        x, _, _ = self.vbn10(x, mean10, meansq10)
        x = self.lrelu10(x)
        x = self.conv11(x)
        x, _, _ = self.vbn11(x, mean11, meansq11)
        x = self.lrelu11(x)
        x = self.conv_final(x)
        x = self.lrelu_final(x)
        # reduce down to a scalar value
        x = torch.squeeze(x)
        x = self.fully_connected(x)
        # return self.sigmoid(x)
        return x


class Generator(nn.Module):
    """
    Generator model of SEGAN.
    """
    def __init__(self):
        super().__init__()
        # size notations = [batch_size x feature_maps x width] (height omitted - 1D convolutions)
        # encoder gets a noisy signal as input
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2, padding=15)   # out : [B x 16 x 8192]
        self.enc1_nl = nn.PReLU()  # non-linear transformation after encoder layer 1
        self.enc2 = nn.Conv1d(16, 32, 32, 2, 15)  # [B x 32 x 4096]
        self.enc2_nl = nn.PReLU()
        self.enc3 = nn.Conv1d(32, 32, 32, 2, 15)  # [B x 32 x 2048]
        self.enc3_nl = nn.PReLU()
        self.enc4 = nn.Conv1d(32, 64, 32, 2, 15)  # [B x 64 x 1024]
        self.enc4_nl = nn.PReLU()
        self.enc5 = nn.Conv1d(64, 64, 32, 2, 15)  # [B x 64 x 512]
        self.enc5_nl = nn.PReLU()
        self.enc6 = nn.Conv1d(64, 128, 32, 2, 15)  # [B x 128 x 256]
        self.enc6_nl = nn.PReLU()
        self.enc7 = nn.Conv1d(128, 128, 32, 2, 15)  # [B x 128 x 128]
        self.enc7_nl = nn.PReLU()
        self.enc8 = nn.Conv1d(128, 256, 32, 2, 15)  # [B x 256 x 64]
        self.enc8_nl = nn.PReLU()
        self.enc9 = nn.Conv1d(256, 256, 32, 2, 15)  # [B x 256 x 32]
        self.enc9_nl = nn.PReLU()
        self.enc10 = nn.Conv1d(256, 512, 32, 2, 15)  # [B x 512 x 16]
        self.enc10_nl = nn.PReLU()
        self.enc11 = nn.Conv1d(512, 1024, 32, 2, 15)  # output : [B x 1024 x 8]
        self.enc11_nl = nn.PReLU()

        # decoder generates an enhanced signal
        # each decoder output are concatenated with homolgous encoder output,
        # so the feature map sizes are doubled
        self.dec10 = nn.ConvTranspose1d(in_channels=2048, out_channels=512, kernel_size=32, stride=2, padding=15)
        self.dec10_nl = nn.PReLU()  # out : [B x 512 x 16] -> (concat) [B x 1024 x 16]
        self.dec9 = nn.ConvTranspose1d(1024, 256, 32, 2, 15)  # [B x 256 x 32]
        self.dec9_nl = nn.PReLU()
        self.dec8 = nn.ConvTranspose1d(512, 256, 32, 2, 15)  # [B x 256 x 64]
        self.dec8_nl = nn.PReLU()
        self.dec7 = nn.ConvTranspose1d(512, 128, 32, 2, 15)  # [B x 128 x 128]
        self.dec7_nl = nn.PReLU()
        self.dec6 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # [B x 128 x 256]
        self.dec6_nl = nn.PReLU()
        self.dec5 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  # [B x 64 x 512]
        self.dec5_nl = nn.PReLU()
        self.dec4 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # [B x 64 x 1024]
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # [B x 32 x 2048]
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  # [B x 32 x 4096]
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # [B x 16 x 8192]
        self.dec1_nl = nn.PReLU()
        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # [B x 1 x 16384]
        self.dec_tanh = nn.Tanh()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, z):
        """
        Forward pass of generator.

        Args:
            x: input batch (signal)
            z: latent vector
        """
        ### encoding step
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))
        e4 = self.enc4(self.enc3_nl(e3))
        e5 = self.enc5(self.enc4_nl(e4))
        e6 = self.enc6(self.enc5_nl(e5))
        e7 = self.enc7(self.enc6_nl(e6))
        e8 = self.enc8(self.enc7_nl(e7))
        e9 = self.enc9(self.enc8_nl(e8))
        e10 = self.enc10(self.enc9_nl(e9))
        e11 = self.enc11(self.enc10_nl(e10))
        # c = compressed feature, the 'thought vector'
        c = self.enc11_nl(e11)

        # concatenate the thought vector with latent variable
        encoded = torch.cat((c, z), dim=1)

        ### decoding step
        d10 = self.dec10(encoded)
        # dx_c : concatenated with skip-connected layer's output & passed nonlinear layer
        d10_c = self.dec10_nl(torch.cat((d10, e10), dim=1))
        d9 = self.dec9(d10_c)
        d9_c = self.dec9_nl(torch.cat((d9, e9), dim=1))
        d8 = self.dec8(d9_c)
        d8_c = self.dec8_nl(torch.cat((d8, e8), dim=1))
        d7 = self.dec7(d8_c)
        d7_c = self.dec7_nl(torch.cat((d7, e7), dim=1))
        d6 = self.dec6(d7_c)
        d6_c = self.dec6_nl(torch.cat((d6, e6), dim=1))
        d5 = self.dec5(d6_c)
        d5_c = self.dec5_nl(torch.cat((d5, e5), dim=1))
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(torch.cat((d4, e4), dim=1))
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, e3), dim=1))
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(torch.cat((d2, e2), dim=1))
        d1 = self.dec1(d2_c)
        d1_c = self.dec1_nl(torch.cat((d1, e1), dim=1))
        out = self.dec_tanh(self.dec_final(d1_c))
        return out


def split_pair_to_vars(sample_batch_pair):
    """
    Splits the generated batch data and creates combination of pairs.
    Input argument sample_batch_pair consists of a batch_size number of
    [clean_signal, noisy_signal] pairs.

    This function creates three pytorch Variables - a clean_signal, noisy_signal pair,
    clean signal only, and noisy signal only.
    It goes through preemphasis preprocessing before converted into variable.

    Args:
        sample_batch_pair(torch.Tensor): batch of [clean_signal, noisy_signal] pairs
    Returns:
        batch_pairs_var(torch.Tensor): batch of pairs containing clean signal and noisy signal
        clean_batch_var(torch.Tensor): clean signal batch
        noisy_batch_var(torch.Tensor): noisy signal batch
    """
    # pre-emphasis
    sample_batch_pair = emph.pre_emphasis(sample_batch_pair.numpy(), emph_coeff=0.95)

    batch_pairs_var = torch.from_numpy(sample_batch_pair).type(torch.FloatTensor).to(device)  # [40 x 2 x 16384]

    clean_batch = np.stack([pair[0].reshape(1, -1) for pair in sample_batch_pair])
    clean_batch_var = torch.from_numpy(clean_batch).type(torch.FloatTensor).to(device)

    noisy_batch = np.stack([pair[1].reshape(1, -1) for pair in sample_batch_pair])
    noisy_batch_var = torch.from_numpy(noisy_batch).type(torch.FloatTensor).to(device)
    return batch_pairs_var, clean_batch_var, noisy_batch_var


def sample_latent():
    """
    Sample a latent vector - normal distribution

    Returns:
        z(torch.Tensor): random latent vector
    """
    return torch.randn((batch_size, 1024, 8)).to(device)


# SOME TRAINING PARAMETERS #
batch_size = 256
d_learning_rate = 0.0002
g_learning_rate = 0.0002
g_lambda = 0  # regularizer for generator
use_devices = [0, 1, 2, 3]
sample_rate = 16000
num_gen_examples = 10  # number of generated audio examples displayed per epoch
num_epochs = 86
train_g_iter = 5

# create D and G instances
discriminator = torch.nn.DataParallel(Discriminator().to(device), device_ids=use_devices)  # use GPU
print(discriminator)
print('Discriminator created')

generator = torch.nn.DataParallel(Generator().to(device), device_ids=use_devices)
print(generator)
print('Generator created')

# This is how you define a data loader
sample_generator = AudioSampleGenerator(os.path.join(in_path, ser_data_fdr))
random_data_loader = DataLoader(
        dataset=sample_generator,
        batch_size=batch_size,  # specified batch size here
        shuffle=True,
        num_workers=4,
        drop_last=True,  # drop the last batch that cannot be divided by batch_size
        pin_memory=True)
print('DataLoader created')

# generate reference batch
ref_batch_pairs = sample_generator.reference_batch(batch_size)
ref_batch_var, ref_clean_var, ref_noisy_var = split_pair_to_vars(ref_batch_pairs)

# optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))

# create tensorboard writer
# The logs will be stored NOT under the run_time, but under segan_data_out/'tblog_fdr'.
# This way, tensorboard can show graphs for each experiment in one board
tbwriter = SummaryWriter(log_dir=tblog_path)
print('TensorboardX summary writer created')

# test samples for generation
test_noise_filenames, fixed_test_clean, fixed_test_noise = \
    sample_generator.fixed_test_audio(batch_size)
fixed_test_clean = torch.from_numpy(fixed_test_clean)
fixed_test_noise = torch.from_numpy(fixed_test_noise)
print('Test samples loaded')

# record the fixed examples
for idx, fname in enumerate(test_noise_filenames[:num_gen_examples]):
    tbwriter.add_audio(
        'test_audio_clean/{}'.format(fname),
        fixed_test_clean.numpy()[idx].T,
        sample_rate=sample_rate)
    tbwriter.add_audio(
        'test_audio_noise/{}'.format(fname),
        fixed_test_noise.numpy()[idx].T,
        sample_rate=sample_rate)


### Train! ###
print('Starting Training...')
total_steps = 1
for epoch in range(num_epochs):
    g_lambda += 1
    # add epoch number with corresponding step number
    tbwriter.add_scalar('epoch', epoch, total_steps)
    for i, sample_batch_pairs in enumerate(random_data_loader):
        # using the sample batch pair, split into
        # batch of combined pairs, clean signals, and noisy signals
        batch_pairs_var, clean_batch_var, noisy_batch_var = split_pair_to_vars(sample_batch_pairs)

        # latent vector - normal distribution
        z = sample_latent()

        ##### TRAIN D #####
        # TRAIN D to recognize clean audio as clean
        # training batch pass
        outputs = discriminator(batch_pairs_var, ref_batch_var)  # out: [n_batch x 1]
        clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1

        # TRAIN D to recognize generated audio as noisy
        generated_outputs = generator(noisy_batch_var, z)
        disc_in_pair = torch.cat((generated_outputs.detach(), noisy_batch_var), dim=1)
        outputs = discriminator(disc_in_pair, ref_batch_var)
        noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
        d_loss = 0.5 * (clean_loss + noisy_loss)

        # back-propagate and update
        discriminator.zero_grad()
        d_loss.backward()
        d_optimizer.step()  # update parameters

        ##### TRAIN G #####
        # TRAIN G so that D recognizes G(z) as real
        for _ in range(train_g_iter):
            z = sample_latent()
            generated_outputs = generator(noisy_batch_var, z)
            gen_noise_pair = torch.cat((generated_outputs, noisy_batch_var), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch_var)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(clean_batch_var)))
            g_cond_loss = g_lambda * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss

            # back-propagate and update
            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # print message and store logs per 10 steps
        if (i + 1) % 20 == 0:
            print(
                'Epoch {}\t'
                'Step {}\t'
                'd_loss {:.5f}\t'
                'd_clean_loss {:.5f}\t'
                'd_noisy_loss {:.5f}\t'
                'g_loss {:.5f}\t'
                'g_loss_cond {:.5f}'
                .format(epoch + 1, i + 1, d_loss.item(), clean_loss.item(),
                        noisy_loss.item(), g_loss.item(), g_cond_loss.item()))

            ### Functions below print various information about the network. Uncomment to use.
            # print('Weight for latent variable z : {}'.format(z))
            # print('Generated Outputs : {}'.format(generated_outputs))
            # print('Encoding 8th layer weight: {}'.format(generator.module.enc8.weight))

            # record scalar data for tensorboard
            tbwriter.add_scalar('loss/d_loss', d_loss.item(), total_steps)
            tbwriter.add_scalar('loss/d_clean_loss', clean_loss.item(), total_steps)
            tbwriter.add_scalar('loss/d_noisy_loss', noisy_loss.item(), total_steps)
            tbwriter.add_scalar('loss/g_loss', g_loss.item(), total_steps)
            tbwriter.add_scalar('loss/g_conditional_loss', g_cond_loss.item(), total_steps)

        # save sampled audio at the beginning of each epoch
        if i == 0:
            z = sample_latent()
            fake_speech = generator(fixed_test_noise, z)
            fake_speech_data = fake_speech.data.cpu().numpy()  # convert to numpy array
            fake_speech_data = emph.de_emphasis(fake_speech_data, emph_coeff=0.95)

            for idx in range(num_gen_examples):
                generated_sample = fake_speech_data[idx]
                gen_fname = test_noise_filenames[idx]
                filepath = os.path.join(
                        gen_data_path, '{}_e{}.wav'.format(gen_fname, epoch))
                # write to file
                wavfile.write(filepath, sample_rate, generated_sample.T)
                # show on tensorboard log
                tbwriter.add_audio(
                    '{}/{}'.format(epoch, gen_fname),
                    generated_sample.T,
                    total_steps,
                    sample_rate)

        total_steps += 1

    # save various states
    state_path = os.path.join(checkpoint_path, 'state-{}.pkl'.format(epoch + 1))
    state = {
        'discriminator': discriminator.state_dict(),
        'generator': generator.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
    }
    torch.save(state, state_path)

    ### Can be loaded using, for example:
    # states = torch.load(state_path)
    # discriminator.load_state_dict(state['discriminator'])

tbwriter.close()
print('Finished Training!')
