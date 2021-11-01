import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from nnmnkwii.metrics import melcd

from utils.synthesis_utils import static_delta_delta_to_static


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to("cuda")

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 


class GRU_Model(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GRU_Model")
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--hidden_dim", type=int, default=256)
        return parent_parser

    def __init__(self, input_dim, output_dim,learning_rate, input_meanstd, output_meanstd, args):
        super().__init__()
        #self.noise = GaussianNoise()
        self.learning_rate = learning_rate
        self.input_meanstd = input_meanstd
        self.output_meanstd = output_meanstd
        # GRU layers share number of hidden layer parameter

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
        loss = F.mse_loss(x_hat[:,:y_len[0],:], y[:,:y_len[0],:])
        mcd_loss = melcd(x_hat[0,:y_len[0],:], y[0,:y_len[0],:])
        self.log("train_loss", loss)
        self.log("melcd", mcd_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, y)

        #static_x_hat = static_delta_delta_to_static(x_hat.detach().cpu().numpy()[0,:,:])

        denormalised_hat = (x_hat.detach().cpu().numpy()[:,:,:60] * self.output_meanstd[1][:60]) + self.output_meanstd[0][:60]
        denormalised_y = (y.detach().cpu().numpy()[0,:,:60] * self.output_meanstd[1][:60]) + self.output_meanstd[0][:60]

        mcd_loss = melcd(denormalised_hat, denormalised_y)
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
