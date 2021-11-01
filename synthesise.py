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

def synthesise_audio_from_test_set(test_set_file: str):
    """
    Synthesise the audio of all the test set files
    
    :param test_set_file: 
    :return: 
    """

    show_figure = False

    test_y = FileSourceDataset(MFCCSource("partitions/testfiles.txt"))
    test_x = FileSourceDataset(ArticulatorySource("partitions/testfiles.txt"))
    test = DataLoader(NanamiDataset(test_y, test_x), batch_size=1, shuffle=False)


    model_file = torch.load("saved_logs/gru_model_hidden_50_num_layer_4/version_1/checkpoints/epoch=224-step=275849.ckpt")
    #model_file = torch.load("lightning_logs/version_3/checkpoints/epoch=6-step=8581.ckpt")
    #print(model_file)

    parser = argparse.ArgumentParser()
    parser = GRU_Model.add_model_specific_args(parser)

    args = parser.parse_args()

    autoencoder = GRU_Model(input_dim=12, output_dim=39, args=args)
    autoencoder.load_state_dict(model_file["state_dict"])

    i = 0
    mcd = 0
    for mfcc, articulation in test:
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
        audio, _ = librosa.load(audio_path[0], sr=16000)
        audio = audio.astype(np.float64)
        f0, sp, ap = pw.wav2world(audio, fs=16000)
        encoded_sp = pw.code_spectral_envelope(sp, 16000, 40)
        predicted_sp = result[0].detach().numpy().astype(np.float64)

        # MCD evaluation
        mcd += melcd(predicted_sp, encoded_sp[:,1:])

        encoded_sp[:,1:] = predicted_sp



        fft_size = pw.get_cheaptrick_fft_size(16000)
        full_sp = pw.decode_spectral_envelope(encoded_sp, 16000, fft_size)

        #pw.decode_spectral_envelope()

        os.makedirs("lightning_logs/version_3/test_audios",exist_ok=True)
        y = pw.synthesize(f0, full_sp, ap, 16000)

        y = librosa.util.normalize(y)
        plt.plot(y)
        plt.show()
        sf.write("lightning_logs/version_3/test_audios/{}.wav".format(audio_path[0].split("/")[-1].split(".")[0]), y, 16000)
        #librosa.output.write_wav("lightning_logs/version_3/test_audios/{}.wav".format(audio_path[0].split("/")[-1], y, 16000))
        i += 1

    mcd /= i
    print("MCD:", mcd)

    #RU_Model.load_state_dict(autoencoder, model_file["state_dict"])
    #print(model_file(test_x[0]))


if __name__ == '__main__':

    synthesise_audio_from_test_set("partitions/testfiles.txt")
    print("")