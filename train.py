from models.gonzalez_gru import GRU_Model
import numpy as np
from nnmnkwii.datasets import FileSourceDataset
from data_utils import MFCCSource, ArticulatorySource, NanamiDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from data_utils import pad_collate
from preprocessing import load_normalisation

def worker_init_fn(worker_id):
    # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(manual_seed + worker_id)

def train(parser):

    train_speech = FileSourceDataset(MFCCSource("partitions/trainfiles.txt", load=True))
    train_art = FileSourceDataset(ArticulatorySource("partitions/trainfiles.txt", load=True))
    # TODO: ONLY FOR REPRODUCTION PURPOSES
    val_speech = FileSourceDataset(MFCCSource("partitions/testfiles.txt"))
    val_art = FileSourceDataset(ArticulatorySource("partitions/testfiles.txt"))
    #test_y = FileSourceDataset(MFCCSource("partitions/testfiles.txt"))
    #test_x = FileSourceDataset(ArticulatorySource("partitions/testfiles.txt"))

    train_nanami = NanamiDataset(train_speech, train_art, norm_calc=False)

    input_mean,  input_std, output_mean, output_std = load_normalisation()
    #input_mean = np.zeros_like(input_mean)
    #input_std = np.ones_like(input_std)
    #output_mean = np.zeros_like(output_mean)
    #output_std = np.ones_like(output_std)
    train_nanami.input_meanstd = (input_mean, input_std)
    train_nanami.output_meanstd = (output_mean, output_std)

    train = DataLoader(train_nanami, batch_size=16, num_workers=4, shuffle=True, collate_fn=pad_collate)

    val_nanami = NanamiDataset(val_speech, val_art, norm_calc=False)
    val_nanami.input_meanstd = train_nanami.input_meanstd
    val_nanami.output_meanstd = train_nanami.output_meanstd

    val = DataLoader(val_nanami, batch_size=1, num_workers=4, collate_fn=pad_collate)


    parser = GRU_Model.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    trainer = pl.Trainer.from_argparse_args(parser,
                                            logger=TensorBoardLogger("logs",
                                                                     name='gru_model_hidden_{}_num_layer_{}'.format(args.hidden_dim, args.num_layers)),
                                            deterministic=True,
                                            max_epochs=20)

    autoencoder = GRU_Model(input_dim=54, output_dim=180, learning_rate=0.003,
                            input_meanstd=train_nanami.input_meanstd,
                            output_meanstd=train_nanami.output_meanstd,
                            args=args)
    #trainer.tune(autoencoder)
    trainer.fit(autoencoder, train, val)

if __name__ == '__main__':

    manual_seed = 0
    parser = ArgumentParser()
    train(parser)


