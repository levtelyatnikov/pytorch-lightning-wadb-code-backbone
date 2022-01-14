import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig

from dataset.dataset import CustomImageDataset

class PL_DataModule(pl.LightningDataModule):
    """
    
    This class creates train/validation datasets
    In usual case this class do not need to be changed
    all data manipulations are performed in dataset.py and 
    transformfactory.py files
    """
    def __init__(
        self,
        cfg: DictConfig, 
    ):

        super().__init__()
        self.dataset_types = {
            "CustomImageDataset": CustomImageDataset
            }

        self.data_dir = cfg.data_dir
        self.train_batch_size = cfg.train_batch_size
        self.val_batch_size = cfg.val_batch_size
        self.transforms = cfg.transforms
        self.num_workers = cfg.num_workers

        # Define dataset and train/val splits
        dataset = self.dataset_types.get(cfg.dataset_type)
        self.train_dataset = dataset(cfg['train']) 
        self.val_dataset = dataset(cfg['val'])
 
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )



