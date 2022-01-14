import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from torchmetrics import Accuracy
from omegaconf import DictConfig



class resnetClassifier(nn.Module):
    """Model exaple"""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Init pretrained resnet
        self.feature_extractor = models.resnet18(pretrained=True)

        # Eval mode to switch batchnorm and dropout into eval mode
        self.feature_extractor.eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        n_sizes = self._get_conv_output()

        # Initialize classifies
        self.classifier = nn.Linear(n_sizes, self.cfg.num_classes)
        
        # Loss
        self.crossentropy = torch.nn.CrossEntropyLoss()
        
        # Metrics
        self.acc = Accuracy()

         
    def _forward_features(self, x):
        """Returns the feature tensor from the conv block"""
        x = self.feature_extractor(x)
        return x

    def _get_conv_output(self,):
        """Return size of the output tensor

        Return size of the output tensor which goes to linear layer 
        """
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *self.cfg.shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
    def forward(self, x):
        
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        
        return logits
    
    def loss_function(self, batch):
        """Loss fuction

        This function implements all logic during train step. In this way you
        model class is selfc ontained, hence there isn't need to change code
        in method.py when model is substituted with other one.
        """
        
        images, labels = batch
        logits = self.forward(images)  
        preds = torch.argmax(logits, dim=1)
        
        loss = self.crossentropy(logits, labels)

        acc = self.acc(preds, labels)

        return {
            "loss": loss,
            "accuracy": acc
            }
