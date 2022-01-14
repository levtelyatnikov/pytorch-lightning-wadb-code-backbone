
import torch
from torchvision import transforms

from omegaconf.dictconfig import DictConfig

class CustomImageDatasetTransforms():
    """Example Transform Object class 

    Define sequence of transforms in inita and pass it though yaml.
    Transform parameters can be easily manipulated trough yaml file
    """
    def __init__(self, cfg: DictConfig):
        self.transforms = transforms.Compose(
            [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)