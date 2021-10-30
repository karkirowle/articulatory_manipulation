import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import scipy
import scipy.interpolate
from torch.nn.utils.rnn import pad_sequence
from nnmnkwii.datasets import FileDataSource, FileSourceDataset, PaddedFileSourceDataset
import librosa
import itertools
import pyworld as pw

import matplotlib.pyplot as plt


class MFCCSource(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):
        files = open(self.data_root).read().splitlines()
        # Because of a,b,c,d,e,f not being included, we need a more brute force globbing approach here
        files_wav = list(map(lambda x: "/home/boomkin/ownCloud/mngu0_wav/mngu0_wav/*/" + x + "*.wav", files))
        all_files = [glob(files) for files in files_wav]
        all_files_flattened = list(itertools.chain(*all_files))
        return all_files_flattened

    def collect_features(self, wav_path):
        x, fs = librosa.load(wav_path,sr=16000)
        frame_time = 10 / 1000
        hop_time = 5 / 1000
        hop_length = int(hop_time * 16000)
        frame_length = int(frame_time * 16000)
        #n_mels = 40
        x = x.astype(np.float64)
        f0, sp, ap = pw.wav2world(x, fs)
        coded_sp = pw.code_spectral_envelope(sp, fs, 40)
        mfcc = coded_sp

        mfcc = mfcc[:,1:]



        return mfcc.astype(np.float32), wav_path


class ArticulatorySource(FileDataSource):
    def __init__(self,data_root,max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

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

            string = f.read()
            data = np.fromstring(string, dtype='float32')
            data_ = np.reshape(data, (-1, len(columns)))

            # There is a list of columns here we can select from, but looking around github, mostly the below are used

            articulators = [
                'T1_py', 'T1_px', 'T3_py', 'T3_px', 'T2_py', 'T2_px',
                'jaw_py', 'jaw_px', 'upperlip_py', 'upperlip_px',
                'lowerlip_py', 'lowerlip_px']
            articulator_idx = [columns[articulator] for articulator in articulators]

            data_out = data_[:, articulator_idx]

            if np.isnan(data_out).sum() != 0:
                # Build a cubic spline out of non-NaN values.
                spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(data_out).ravel()),
                                                  data_out[~np.isnan(data_out)], k=3)
                # Interpolate missing values and replace them.
                for j in np.argwhere(np.isnan(data_out)).ravel():
                    data_out[j] = scipy.interpolate.splev(j, spline)

        return data_out, ema_path

class NanamiDataset(Dataset):
    """
    Generic wrapper around nnmnkwii datsets
    """
    def __init__(self,speech_padded_file_source,art_padded_file_source):
        self.speech = speech_padded_file_source
        self.art = art_padded_file_source

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()



        art_temp = scipy.signal.resample(self.art[idx][0], num=self.speech[idx][0].shape[0])
        return torch.FloatTensor(self.speech[idx][0]), torch.FloatTensor(art_temp)
