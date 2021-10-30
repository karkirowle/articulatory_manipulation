import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from nnmnkwii.metrics import melcd

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
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--hidden_dim", type=int, default=150)
        return parent_parser

    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        #self.noise = GaussianNoise()

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
        x, y = batch
        x_hat = self.forward(x)
        #print("x_hat", x_hat.shape)
        #print("y", y.shape)
        loss = F.mse_loss(x_hat, y)
        mcd_loss = melcd(x_hat, y)
        self.log("train_loss", loss)
        self.log("melcd", mcd_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
