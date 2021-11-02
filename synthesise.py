import argparse

from nnmnkwii.datasets import FileDataSource, FileSourceDataset, PaddedFileSourceDataset
from data_utils import MFCCSource, ArticulatorySource, NanamiDataset
from torch.utils.data import DataLoader
import torch

from models.gonzalez_gru import GRU_Model

import matplotlib.pyplot as plt

import pyworld as pw
import os

import librosa
import numpy as np
import soundfile as sf

import argparse

from nnmnkwii.metrics import melcd
from preprocessing import load_normalisation
from utils.synthesis_utils import static_delta_delta_to_static

vocoder_dim = 60
fs = 16000
def synthesise_audio_from_test_set(test_set_file: str):
    """
    Synthesise the audio of all the test set files
    
    :param test_set_file: 
    :return: 
    """

    show_figure = False

    test_y = FileSourceDataset(MFCCSource("partitions/testfiles.txt"))
    test_x = FileSourceDataset(ArticulatorySource("partitions/testfiles.txt"))
    test_nanami = NanamiDataset(test_y, test_x, norm_calc=False)
    input_mean, input_std, output_mean, output_std = load_normalisation()
    input_meanstd = (input_mean, input_std)
    output_meanstd = (output_mean, output_std)
    test_nanami.input_meanstd = input_meanstd
    test_nanami.output_meanstd = output_meanstd

    test = DataLoader(test_nanami, batch_size=1, shuffle=False)


    model_file = torch.load("logs/gru_model_hidden_150_num_layer_4/version_86/checkpoints/epoch=14-step=1154.ckpt")
    #model_file = torch.load("lightning_logs/version_3/checkpoints/epoch=6-step=8581.ckpt")
    #print(model_file)

    parser = argparse.ArgumentParser()
    parser = GRU_Model.add_model_specific_args(parser)

    args = parser.parse_args()
    autoencoder = GRU_Model(input_dim=54, output_dim=180, args=args,
                            input_meanstd=input_meanstd, output_meanstd=output_meanstd, learning_rate=0.001)
    autoencoder.load_state_dict(model_file["state_dict"])

    i = 0
    for articulation, mfcc in test:
        audio_path = test_y.collected_files[i]

        result = autoencoder(articulation)

        if show_figure:
            plt.subplot(1, 3, 1)
            plt.plot(articulation.detach().numpy()[0])
            plt.subplot(1, 3, 2)
            plt.imshow(result[0].detach().numpy(),aspect='auto')
            plt.subplot(1, 3, 3)
            plt.imshow(mfcc[0].detach().numpy(),aspect='auto')
            plt.show()

        print("audio path in synthesise", audio_path)
        audio, _ = librosa.load(audio_path[0], sr=fs)
        audio = audio.astype(np.float64)
        f0, sp, ap = pw.wav2world(audio, fs=fs)
        encoded_sp = pw.code_spectral_envelope(sp, fs, vocoder_dim)
        predicted_sp_delta = result[0].detach().numpy().astype(np.float64)

        predicted_sp = static_delta_delta_to_static(predicted_sp_delta)
        predicted_sp = predicted_sp * output_meanstd[1][:vocoder_dim] + output_meanstd[0][:vocoder_dim]
        encoded_sp[:,1:] = predicted_sp[:,1:]

        fft_size = pw.get_cheaptrick_fft_size(fs)
        full_sp = pw.decode_spectral_envelope(encoded_sp, fs, fft_size)

        #pw.decode_spectral_envelope()

        os.makedirs("logs/gru_model_hidden_150_num_layer_4/version_86/test_audios",exist_ok=True)
        y = pw.synthesize(f0, full_sp, ap, fs)

        y = librosa.util.normalize(y)
        plt.plot(y)
        plt.show()
        sf.write("logs/gru_model_hidden_150_num_layer_4/version_86/test_audios/{}.wav".format(audio_path[0].split("/")[-1].split(".")[0]), y, fs)
        #librosa.output.write_wav("lightning_logs/version_3/test_audios/{}.wav".format(audio_path[0].split("/")[-1], y, 16000))
        i += 1


    #RU_Model.load_state_dict(autoencoder, model_file["state_dict"])
    #print(model_file(test_x[0]))


if __name__ == '__main__':

    synthesise_audio_from_test_set("partitions/testfiles.txt")