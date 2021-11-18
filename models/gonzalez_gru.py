import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from nnmnkwii.metrics import melcd

from torch.autograd import Variable
#from torch.nn import Variable

from utils.synthesis_utils import static_delta_delta_to_static


from nnmnkwii import autograd
from fastdtw import fastdtw
import scipy

import numpy as np

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="sum")

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(input).to(device="cuda")
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask.sum()

class GRU_Model(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GRU_Model")
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--hidden_dim", type=int, default=150)
        return parent_parser

    def __init__(self, input_dim, output_dim,learning_rate, input_meanstd, output_meanstd, args):
        super().__init__()
        #self.noise = GaussianNoise()
        self.learning_rate = learning_rate
        self.input_meanstd = input_meanstd
        self.output_meanstd = output_meanstd
        # GRU layers share number of hidden layer parameter
        #self.mlpg = autograd.UnitVarianceMLPG()
        self.masked_mse = MaskedMSELoss()
        self.gru_1a = nn.GRU(input_dim, args.hidden_dim, num_layers=args.num_layers, bidirectional=True, batch_first=True)
        self.dense = nn.Linear(args.hidden_dim*2, output_dim)

    def forward(self, x):

        output, _ = self.gru_1a(x)

        #print("gru output shape", output.shape)

        output = self.dense(output)


        return output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y, x_len, y_len = batch
        x_hat = self.forward(x)

        #variance_frames = torch.ones_like(x_hat)
        #static_x_hat = self.mlpg(x_hat, variance_frames)
        #static_x_hat = static_delta_delta_to_static(x_hat.detach().cpu().numpy()[0,y_len[0],:])
        #   print(y_len[0], batch_idx)
        loss = self.masked_mse(x_hat, y, lengths=torch.LongTensor(x_len))
        #loss = F.mse_loss(x_hat[:,:y_len[0],:], y[:,:y_len[0],:])
        mcd_loss = melcd(x_hat[0,:y_len[0],:], y[0,:y_len[0],:])
        self.log("train_loss", loss)
        self.log("melcd", mcd_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, y)

        static_x_hat = static_delta_delta_to_static(x_hat.detach().cpu().numpy()[0,:,:])

        dim = 60
        denormalised_hat = (static_x_hat[:,:dim] * self.output_meanstd[1][:dim]) + self.output_meanstd[0][:dim]
        denormalised_y = (y.detach().cpu().numpy()[0,:,:dim] * self.output_meanstd[1][:dim]) + self.output_meanstd[0][:dim]

        # Alignment is used in some algorithms and I think it's guaranteed that perform can only improve with this
        _, path = fastdtw(
            denormalised_hat[:,1:dim],
            denormalised_y[:,1:dim],
            dist=scipy.spatial.distance.euclidean,
        )
        twf_pow = np.array(path).T

        mcd_loss = melcd(denormalised_hat[twf_pow[0],1:dim], denormalised_y[twf_pow[1],1:dim])

        #mcd_loss = melcd(x_hat[:,:,:60].cpu().numpy(), y[0,:,:60].cpu().numpy())
        #print("MCD", mcd_loss)
        self.log("val_loss", loss)
        self.log("val_melcd", mcd_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, y)
        mcd_loss = melcd(x_hat, y)
        self.log("test_loss", loss)
        self.log("test_melcd", mcd_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate))
        return optimizer
