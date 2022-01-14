import sys
from torchvision.io import read_image
from torch.utils.data import Dataset
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import train_test_split
from dataset.transformfactory import *

class CustomImageDataset(Dataset):
    """Dataset example"""
    def __init__(self,  cfg: DictConfig):
        """Initialize

        cfg:
         :data_dir: data directory
         :transforms: TransformObject Class which is defined in the transformfactory.py file
         :target_transforms: TransformLabel Class which is defined in the transformfactory.py file

         :split: train/val split
         :val_size: validation size
         :seed: seed
        """
        
        self.data_dir = cfg.data_dir
        self.split = cfg.split
        self.val_size = cfg.val_size
        self.seed = cfg.seed

        # Setup transforms
        self.transforms = eval(cfg.transforms) 
        self.target_transforms = eval(cfg.target_transform)
        
        # Obtain list of (paths, labels)
        self.setup()

    def setup(self):
        img_paths, img_labels = # Get image paths and labels
        img_paths_labels = list(zip(img_paths, img_labels))

        # Split
        train, val = train_test_split(img_paths_labels,
                                test_size=self.val_size,
                                random_state=self.seed)

        if self.split == "train":
            self.img_paths_labels = train
        elif self.split == "val":
            self.img_paths_labels = val
        else:
            print("Specify dataset split correctly", file=sys.stderr)
            

    def __getitem__(self, idx):
        """Return image and label"""

        image_path, label = self.img_paths_labels[idx]
        image = read_image(image_path) 
        
        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            label = self.target_transforms(label)
        return image, label

    def __len__(self):
        return len(self.img_paths_labels)