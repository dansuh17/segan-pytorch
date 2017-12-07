import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from data_generator import AudioSampleGenerator
from scipy.io import wavfile
from vbnorm import VirtualBatchNorm1d

"""
Here we define the discriminator and generator for SEGAN.
After definition of each modules, run the training.
"""

# define folders for data
data_path = '../data/segan'
clean_train_foldername = 'clean_trainset_wav/clean_trainset_56spk_wav'
noisy_train_foldername = 'noisy_trainset_wav/noisy_trainset_56spk_wav'
out_clean_train_fdrnm = 'clean_trainset_wav_16k'
out_noisy_train_fdrnm = 'noisy_trainset_wav_16k'
ser_data_fdrnm = 'ser_data'  # serialized data
gen_data_fdrnm = 'gen_data_v2'  # folder for saving generated data
model_fdrnm = 'models'  # folder for saving models

# create folder for generated data
gen_data_path = os.path.join(os.getcwd(), gen_data_fdrnm)
if not os.path.exists(gen_data_path):
    os.makedirs(gen_data_path)

# create folder for model checkpoints
models_path = os.path.join(os.getcwd(), model_fdrnm)
if not os.path.exists(models_path):
    os.makedirs(models_path)


class Discriminator(nn.Module):
    """
    D
    """
    def __init__(self, dropout_drop=0.5):
        super().__init__()
        # Define convolution operations.
        # (#input channel, #output channel, kernel_size, stride, padding)
        # in : 16384 x 2
        negative_slope = 0.03
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=31, stride=2, padding=15)   # out : 8192 x 32
        self.vbn1 = nn.BatchNorm1d(32)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv1d(32, 64, 31, 2, 15)  # 4096 x 64
        self.vbn2 = nn.BatchNorm1d(64)
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.conv3 = nn.Conv1d(64, 64, 31, 2, 15)  # 2048 x 64
        self.dropout1 = nn.Dropout(dropout_drop)
        self.vbn3 = nn.BatchNorm1d(64)
        self.lrelu3 = nn.LeakyReLU(negative_slope)
        self.conv4 = nn.Conv1d(64, 128, 31, 2, 15) # 1024 x 128
        self.vbn4 = nn.BatchNorm1d(128)
        self.lrelu4 = nn.LeakyReLU(negative_slope)
        self.conv5 = nn.Conv1d(128, 128, 31, 2, 15)  # 512 x 128
        self.vbn5 = nn.BatchNorm1d(128)
        self.lrelu5 = nn.LeakyReLU(negative_slope)
        self.conv6 = nn.Conv1d(128, 256, 31, 2, 15)  # 256 x 256
        self.dropout2 = nn.Dropout(dropout_drop)
        self.vbn6 = nn.BatchNorm1d(256)
        self.lrelu6 = nn.LeakyReLU(negative_slope)
        self.conv7 = nn.Conv1d(256, 256, 31, 2, 15)  # 128 x 256
        self.vbn7 = nn.BatchNorm1d(256)
        self.lrelu7 = nn.LeakyReLU(negative_slope)
        self.conv8 = nn.Conv1d(256, 512, 31, 2, 15)  # 64 x 512
        self.vbn8 = nn.BatchNorm1d(512)
        self.lrelu8 = nn.LeakyReLU(negative_slope)
        self.conv9 = nn.Conv1d(512, 512, 31, 2, 15)  # 32 x 512
        self.dropout3 = nn.Dropout(dropout_drop)
        self.vbn9 = nn.BatchNorm1d(512)
        self.lrelu9 = nn.LeakyReLU(negative_slope)
        self.conv10 = nn.Conv1d(512, 1024, 31, 2, 15)  # 16 x 1024
        self.vbn10 = nn.BatchNorm1d(1024)
        self.lrelu10 = nn.LeakyReLU(negative_slope)
        self.conv11 = nn.Conv1d(1024, 2048, 31, 2, 15)  # 8 x 1024
        self.vbn11 = nn.BatchNorm1d(2048)
        self.lrelu11 = nn.LeakyReLU(negative_slope)
        # 1x1 size kernel for dimension and parameter reduction
        self.conv_final = nn.Conv1d(2048, 1, kernel_size=1, stride=1)  # 8 x 1
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.vbn1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.vbn2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        x = self.vbn3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.vbn4(x)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x = self.vbn5(x)
        x = self.lrelu5(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x = self.vbn6(x)
        x = self.lrelu6(x)
        x = self.conv7(x)
        x = self.vbn7(x)
        x = self.lrelu7(x)
        x = self.conv8(x)
        x = self.vbn8(x)
        x = self.lrelu8(x)
        x = self.conv9(x)
        x = self.dropout3(x)
        x = self.vbn9(x)
        x = self.lrelu9(x)
        x = self.conv10(x)
        x = self.vbn10(x)
        x = self.lrelu10(x)
        x = self.conv11(x)
        x = self.vbn11(x)
        x = self.lrelu11(x)
        x = self.conv_final(x)
        x = self.lrelu_final(x)
        return self.sigmoid(x)


class Generator(nn.Module):
    """
    G
    """
    def __init__(self, batch_size):
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
        self.dec10_nl = nn.PReLU()  # out : [B x 512 x 16]
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

    def forward(self, x, z):
        """
        Forward pass of generator.

        Args:
            x: input data (signal)
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
        out = self.dec_final(d1_c)
        return out


### SOME TRAINING PARAMETERS ###
# batch size
batch_size = 400
learning_rate = 0.0002
g_lambda = 100  # regularizer for generator


# create D and G instances
discriminator = torch.nn.DataParallel(Discriminator(), device_ids=[0]).cuda()  # use GPU
print(discriminator)
print('Discriminator created')

generator = torch.nn.DataParallel(Generator(batch_size), device_ids=[0]).cuda()
# latent variable for generator
z = Variable(torch.rand((batch_size, 1024, 8)).cuda(), requires_grad=True)
print(generator)
print('Generator created')

# This is how you define a data loader
sample_generator = AudioSampleGenerator(os.path.join(data_path, ser_data_fdrnm))
random_data_loader = DataLoader(
        dataset=sample_generator,
        batch_size=batch_size,  # specified batch size here
        shuffle=True,
        num_workers=4,
        drop_last=True,  # drop the last batch that cannot be divided by batch_size
        pin_memory=True)
print('DataLoader created')

# test samples for generation
test_noise_filenames, fixed_test_noise = sample_generator.fixed_test_audio(batch_size)
fixed_test_noise = Variable(torch.from_numpy(fixed_test_noise))
print('Test samples loaded')

# optimizers
g_optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=learning_rate)


### Train! ###
print('Starting Training...')
for epoch in range(40):
    for i, sample_batch_pairs in enumerate(random_data_loader):
        batch_pairs_var = Variable(sample_batch_pairs).cuda()  # [40 x 2 x 16384]
        clean_batch = np.stack([pair[0].numpy().reshape(1, -1) for pair in sample_batch_pairs])
        clean_batch_var = Variable(torch.from_numpy(clean_batch), requires_grad=False).cuda()
        noisy_batch = np.stack([pair[1].numpy().reshape(1, -1) for pair in sample_batch_pairs])
        noisy_batch_var = Variable(torch.from_numpy(noisy_batch), requires_grad=False).cuda()  # do not apply grad update for samples

        ##### TRAIN D #####
        ##### TRAIN D to recognize clean audio as clean
        # training batch pass
        outputs = discriminator(batch_pairs_var)  # output : [40 x 1 x 8]
        clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1

        ##### TRAIN D to recognize generated audio as noisy
        generated_outputs = generator(noisy_batch_var, z)
        disc_in_pair = torch.cat((generated_outputs, noisy_batch_var), dim=1)
        outputs = discriminator(disc_in_pair)

        noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
        # backprop + optimize
        d_loss = clean_loss + noisy_loss
        discriminator.zero_grad()
        generator.zero_grad()
        d_loss.backward()  # perform single backpropagation
        d_optimizer.step()  # update parameters

        ##### TRAIN G #####
        ##### TRAIN G so that D recognizes G(z) as real
        generated_outputs = generator(noisy_batch_var, z)
        gen_noise_pair = torch.cat((generated_outputs, noisy_batch_var), dim=1)
        outputs = discriminator(gen_noise_pair)

        g_loss = 0.5 * torch.mean((outputs - 1.0) ** 2)
        # L1 loss between generated output and clean sample
        g_gennoise_dist = g_lambda * torch.abs(torch.sum((torch.add(generated_outputs, torch.neg(clean_batch_var)))))
        g_loss = g_loss + g_gennoise_dist

        # backprop + optimize
        generator.zero_grad()
        discriminator.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # print message per 10 steps
        if (i + 1) % 10 == 0:
            print('Epoch {}, Step {}, d_clean_loss {}, d_noisy_loss {}, g_loss {}'
                    .format(epoch + 1, i + 1, clean_loss.data[0], noisy_loss.data[0], g_loss.data[0]))

        # save sampled audio at the beginning of each epoch
        if i == 0:
            fake_speech = generator(fixed_test_noise, z)
            fake_speech_data = fake_speech.data.cpu().numpy()  # convert to numpy array
            for idx in range(4):
                generated_sample = fake_speech_data[idx]
                filepath = os.path.join(gen_data_path, '{}_e{}.wav'.format(test_noise_filenames[idx], epoch + 1))
                wavfile.write(filepath, 16000, generated_sample.T)

    # save the model parameters for each epoch
    g_path = os.path.join(models_path, 'generator-{}.pkl'.format(epoch + 1))
    d_path = os.path.join(models_path, 'discriminator-{}.pkl'.format(epoch + 1))
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)
print('Finished Training!')
