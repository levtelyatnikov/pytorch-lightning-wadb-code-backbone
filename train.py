import os
# hydra
import hydra 
from omegaconf import DictConfig, OmegaConf

# pytorch-lightning related imports
from pytorch_lightning import Trainer
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

# own modules
from dataloader import PL_DataModule
from method import LitModel

def setup_cuda(cfg: DictConfig):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.trainer.cuda_number

def get_dataloader(cfg: DictConfig):
    return PL_DataModule(cfg.dataloader)
  
    
@hydra.main(config_path='./configs', config_name='defaults')
def main(cfg: DictConfig):
    setup_cuda(cfg)
    print(OmegaConf.to_yaml(cfg))

    # Configure weight and biases 
    logger = pl_loggers.WandbLogger(
        project=cfg.wadb.logger_project_name,
        name=cfg.wadb.logger_name, 
        entity=cfg.wadb.entity)
            
    # Configure trained
    trainer = Trainer(
        pus=cfg.trainer.gpus,
        logger=logger if cfg.trainer.is_logger_enabled else False,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps, 
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        max_epochs=cfg.model.opt.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[LearningRateMonitor("step")] if cfg.trainer.is_logger_enabled else [],)

    # Setup dataloader and model
    datamodule = get_dataloader(cfg)
    model = LitModel(cfg=cfg.model)

    # Train
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
