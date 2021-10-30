from models.gonzalez_gru import GRU_Model
import numpy as np
from nnmnkwii.datasets import FileSourceDataset
from data_utils import MFCCSource, ArticulatorySource, NanamiDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from argparse import ArgumentParser

def worker_init_fn(worker_id):
    # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(manual_seed + worker_id)

def train(parser):

    train_y = FileSourceDataset(MFCCSource("partitions/trainfiles.txt"))
    train_x = FileSourceDataset(ArticulatorySource("partitions/trainfiles.txt"))
    val_y = FileSourceDataset(MFCCSource("partitions/validationfiles.txt"))
    val_x = FileSourceDataset(ArticulatorySource("partitions/validationfiles.txt"))
    #test_y = FileSourceDataset(MFCCSource("partitions/testfiles.txt"))
    #test_x = FileSourceDataset(ArticulatorySource("partitions/testfiles.txt"))

    train = DataLoader(NanamiDataset(train_x, train_y), batch_size=1, num_workers=4)
    val = DataLoader(NanamiDataset(val_x, val_y), batch_size=1, num_workers=4)

    parser = GRU_Model.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    trainer = pl.Trainer.from_argparse_args(parser,
                                            logger=TensorBoardLogger("logs",
                                                                     name='gru_model_hidden_{}_num_layer_{}'.format(args.hidden_dim, args.num_layers)),
                                            deterministic=True)

    autoencoder = GRU_Model(input_dim=12, output_dim=39, args=args)
    trainer.fit(autoencoder, train, val)

if __name__ == '__main__':

    manual_seed = 0
    parser = ArgumentParser()
    train(parser)


