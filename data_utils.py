import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import scipy
import scipy.interpolate
from torch.nn.utils.rnn import pad_sequence
from nnmnkwii.datasets import FileDataSource, FileSourceDataset, PaddedFileSourceDataset
from nnmnkwii.preprocessing import meanstd
from nnmnkwii.preprocessing import delta_features

import librosa
import itertools
import pyworld as pw

import matplotlib.pyplot as plt

import os
windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0]))]



class MFCCSource(FileDataSource):
    def __init__(self,data_root,max_files=None, save=False, load=False):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None
        self.save = save
        self.load = load


    def collect_files(self):
        files = open(self.data_root).read().splitlines()
        # Because of a,b,c,d,e,f not being included, we need a more brute force globbing approach here
        files_wav = list(map(lambda x: "/home/boomkin/ownCloud/mngu0_wav/mngu0_wav/*/" + x + "*.wav", files))
        all_files = [glob(files) for files in files_wav]
        all_files_flattened = list(itertools.chain(*all_files))
        return all_files_flattened

    def collect_features(self, wav_path):

        if self.load:
            return np.load(os.path.join("preprocessed", "mfcc", os.path.basename(wav_path).split(".")[0] + ".npy"))

        x, fs = librosa.load(wav_path,sr=16000)
        frame_time = 10 / 1000
        hop_time = 5 / 1000
        # TODO: Here make sure you have the correct analysis window size
        #hop_length = int(hop_time * 16000)
        #frame_length = int(frame_time * 16000)
        #n_mels = 40
        x = x.astype(np.float64)
        f0, sp, ap = pw.wav2world(x, fs)
        coded_sp = pw.code_spectral_envelope(sp, fs, 60)
        mfcc = coded_sp
        mfcc_delta = delta_features(mfcc, windows).astype(np.float32)
        if self.save:
            np.save(os.path.join("preprocessed", "mfcc", os.path.basename(wav_path).replace(".wav", ".npy")), mfcc_delta)
        return mfcc_delta


class ArticulatorySource(FileDataSource):
    def __init__(self,data_root,max_files=None, save=False, load=False):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None
        self.save = save
        self.load = load

    def collect_files(self):
        files = open(self.data_root).read().splitlines()
        files_wav = list(map(lambda x: "/home/boomkin/ownCloud/mngu0_ema/*/" + x + "*.ema", files))
        all_files = [glob(files) for files in files_wav]
        all_files_flattened = list(itertools.chain(*all_files))
        return all_files_flattened

    def clean(self,s):
        """
        Strips the new line character from the buffer input
        Parameters:
        -----------
        s: Byte buffer
        Returns:
        --------
        p: string stripped from new-line character
        """
        s = str(s, "utf-8")
        return s.rstrip('\n').strip()

    def collect_features(self, ema_path):

        if self.load:
            return np.load(os.path.join("preprocessed","ema",os.path.basename(ema_path).split(".")[0] + ".npy"))
        columns = {}
        columns["time"] = 0
        columns["present"] = 1

        with open(ema_path, 'rb') as f:

            dummy_line = f.readline()  # EST File Track
            datatype = self.clean(f.readline()).split()[1]
            nframes = int(self.clean(f.readline()).split()[1])
            f.readline()  # Byte Order
            nchannels = int(self.clean(f.readline()).split()[1])

            while not 'CommentChar' in str(f.readline(), "utf-8"):
                pass
            f.readline()  # empty line
            line = self.clean(f.readline())

            while not "EST_Header_End" in line:
                channel_number = int(line.split()[0].split('_')[1]) + 2
                channel_name = line.split()[1]
                columns[channel_name] = channel_number
                line = self.clean(f.readline())
        # with open(ema_path, 'rb') as ema_annotation:
        #     column_names = [0] * 87
        #     for line in ema_annotation:
        #         line = line.decode('latin-1').strip("\n")
        #         if line == 'EST_Header_End':
        #             break
        #         elif line.startswith('NumFrames'):
        #             n_frames = int(line.rsplit(' ', 1)[-1])
        #         elif line.startswith('Channel_'):
        #             col_id, col_name = line.split(' ', 1)
        #             column_names[int(col_id.split('_', 1)[-1])] = col_name
        #
        #
        #     ema_data = np.fromfile(ema_annotation, "float32").reshape(n_frames, 87 + 2)
            ema_buffer = f.read()
            data = np.fromstring(ema_buffer, dtype='float32')
            data_ = np.reshape(data, (-1, len(columns)))

            # There is a list of columns here we can select from, but looking around github, mostly the below are used

            articulators = [
                'T1_py', 'T1_px', 'T1_pz',
                'T3_py', 'T3_px', 'T3_pz',
                'T2_py', 'T2_px', 'T3_pz',
                'jaw_py', 'jaw_px', 'jaw_pz',
                'upperlip_py', 'upperlip_px', 'upperlip_pz',
                'lowerlip_py', 'lowerlip_px', 'lowerlip_pz']
            articulator_idx = [columns[articulator] for articulator in articulators]

            data_out = data_[:, articulator_idx]

            if np.isnan(data_out).sum() != 0:
                # Build a cubic spline out of non-NaN values.
                spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(data_out).ravel()),
                                                  data_out[~np.isnan(data_out)], k=3)
                # Interpolate missing values and replace them.
                for j in np.argwhere(np.isnan(data_out)).ravel():
                    data_out[j] = scipy.interpolate.splev(j, spline)

        delta_ema = delta_features(data_out, windows)

        if self.save:
            np.save(os.path.join("preprocessed","ema",os.path.basename(ema_path).split(".")[0] + ".npy"), delta_ema)
        return delta_ema

class NanamiDataset(Dataset):
    """
    Generic wrapper around nnmnkwii datsets
    """
    def __init__(self,speech_padded_file_source,art_padded_file_source, norm_calc):
        self.speech = speech_padded_file_source
        self.art = art_padded_file_source

        self.input_meanstd = None
        self.output_meanstd = None

        if norm_calc:

            print("Performing articulatory feature normalization (input)...")
            art_lengths = [len(y) for y in self.art]
            self.input_meanstd = meanstd(self.art, art_lengths)
            np.save("input_mean.npy", self.input_meanstd[0])
            np.save("input_std.npy", self.input_meanstd[1])
            print("Performing speech feature normalization (output)...")
            speech_lengths = [len(y) for y in self.speech]
            self.output_meanstd = meanstd(self.speech, speech_lengths)
            np.save("output_mean.npy", self.output_meanstd[0])
            np.save("output_std.npy", self.output_meanstd[1])

            #np.save(self.output_meanstd[0], "output_mean.npy")
            #np.save(self.output_meanstd[1], "output_std.npy")

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_mean, input_std = self.input_meanstd
        output_mean, output_std = self.output_meanstd
        normalised_art = (self.art[idx] - input_mean) / input_std
        normalised_speech = (self.speech[idx] - output_mean) / output_std

        synced_art = scipy.signal.resample(normalised_art, num=self.speech[idx].shape[0])
        #art_temp = scipy.signal.resample(self.art[idx][0], num=self.speech[idx][0].shape[0])
        return torch.FloatTensor(synced_art), torch.FloatTensor(normalised_speech)

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    #mask = pad_sequence([torch.ones_like(y) for y in yy], batch_first=True, padding_value=0)
    return xx_pad,  yy_pad, x_lens, y_lens