
import pyworld as pw
import numpy as np
import librosa
#from nnmnkwii.baseline.gmm import MLPG
from nnmnkwii import paramgen
from nnmnkwii.preprocessing import delta_features
from data_utils import MFCCSource, ArticulatorySource, NanamiDataset
from torch.utils.data import DataLoader

from nnmnkwii.datasets import FileDataSource, FileSourceDataset, PaddedFileSourceDataset

import os
import matplotlib.pyplot as plt
from glob import glob

import sounddevice as sd
from tqdm import tqdm
fftlen = pw.get_cheaptrick_fft_size(16000)
#files = glob("/home/boomkin/ownCloud/mngu0_wav/mngu0_wav/train/*.wav")

#audio_path = test_y.collected_files[i]

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

def preprocessing():
    train_speech = FileSourceDataset(MFCCSource("partitions/trainfiles.txt",save=True))
    train_art = FileSourceDataset(ArticulatorySource("partitions/trainfiles.txt",save=True))
    train_nanami = NanamiDataset(train_speech, train_art, norm_calc=True)
    train = DataLoader(train_nanami, batch_size=1, num_workers=4)

def check_preprocessing():

    print("Checking preprocessing")
    input_mean = np.load("input_mean.npy")
    input_std = np.load("input_std.npy")
    output_mean = np.load("output_mean.npy")
    output_std = np.load("output_std.npy")

    print("Input mean: ", input_mean.shape)
    print("Input std: ", input_std.shape)
    print("Output mean: ", output_mean.shape)
    print("Output std: ", output_std.shape)


def load_normalisation():

    input_mean = np.load("input_mean.npy")
    input_std = np.load("input_std.npy")
    output_mean = np.load("output_mean.npy")
    output_std = np.load("output_std.npy")

    return input_mean, input_std, output_mean, output_std

#features = list()
#f0s = list()
#aps = list()
# for file in tqdm(files):
#     audio, _ = librosa.load(file, sr=16000)
#     audio = librosa.util.normalize(audio)
#     audio = audio.astype(np.float64)
#
#     f0, sp, ap = pw.wav2world(audio, fs=16000)
#     encoded_sp = pw.code_spectral_envelope(sp, 16000, 40)
#     delta_and_static = delta_features(encoded_sp, windows)
#     #plt.imshow(delta_and_static.T[:40,:], aspect='auto', origin='lower', interpolation='none')
#     #plt.title(file)
#     #plt.show()
#     features.append(delta_and_static)
#     f0s.append(f0)
#     aps.append(ap)
#
# features_ctd = np.concatenate(features, axis=0)
#
# #std_ctd = np.concat
# mean = np.mean(features_ctd,axis=0)
#
# std = np.std(features_ctd, axis=0)
# #unit_variance = paramgen.unit_variance_mlpg_matrix(windows, delta_and_static.shape[0])
#
# #g = paramgen.MLPG(encoded_sp, unit_variance, windows)
# #print(delta_and_static.shape)
# #print(unit_variance.shape)
#
#
#
# for feature, f0, ap in zip(features,f0s,aps):
#
#     normalised_feature = (feature - mean) / std
#     variance_frames = np.ones_like(normalised_feature)
#     static_features = paramgen.mlpg(normalised_feature, variance_frames, windows)
#
#
#     static_features = static_features * std[:40] + mean[:40]
#     #plt.imshow(static_features.T, aspect='auto', origin='lower', interpolation='none')
#     #plt.show()
#     decoded_sp = pw.decode_spectral_envelope(static_features, 16000, fftlen)
#     #plt.imshow(decoded_sp,aspect='auto')
#     #plt.show()
#
#     # pw.decode_spectral_envelope()
#
#     #os.makedirs("lightning_logs/version_3/test_audios", exist_ok=True)
#     y = pw.synthesize(f0, decoded_sp, ap, 16000)
#
#     y = librosa.util.normalize(y)
#
#     sd.play(y, 16000)
#     sd.wait()
#     #plt.plot(y)
#     #plt.show()

#paramgen = MLPG(gmm, windows=windows, diff=diffvc)

if __name__ == '__main__':
    preprocessing()
    print(check_preprocessing())