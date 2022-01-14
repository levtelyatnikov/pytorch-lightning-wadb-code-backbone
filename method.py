import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from models.resnet import resnetClassifier




class LitModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        pl.utilities.seed.seed_everything(0)
        # -----------------!!!----------------
        # If you want to substitute models from .yaml file
        # it is necessary to map models name: class
        self.model_types = {
            "resnetClassifier": resnetClassifier
            }
        # -----------------!!!----------------
        
        # save pytorch lightning parameters   
        # this row makes ur parameters be available with self.hparams name
        self.save_hyperparameters(cfg)

        # get model from .yaml file
        self.model = self.model_types.get(cfg.model_type)(cfg)

        # opt parameters
        self.learning_rate = cfg.opt.lr
        self.num_classes = cfg.num_classes

       
    # logic for a single training step
    def training_step(self, batch, batch_idx):
        train_loss = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        val_loss = self.model.loss_function(batch)
        return val_loss
        
    def validation_epoch_end(self, outputs):
        logs = {}
        keys = outputs[0].keys()
        for key in keys:
            logs["val_" + key] = torch.stack([x[key] for x in outputs]).mean()
               
        self.log_dict(logs, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer