'''
    Model implementation.
    We'll be using a "simple" ResNet-18 for image classification here.

    2022 Benjamin Kellenberger
'''

import torch.nn as nn
from torchvision.models import resnet
import torchvision.models as models
from collections import OrderedDict


class CustomResNet50(nn.Module):

    def __init__(self, num_classes):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(CustomResNet50, self).__init__()

        self.feature_extractor = resnet.resnet50(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet

        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        last_layer = self.feature_extractor.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        self.feature_extractor.fc = nn.Identity()                       # discard last layer...

        self.classifier = nn.Linear(in_features, num_classes)           # ...and create a new one
    

    def forward(self, x):
        '''
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-18 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-18 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return prediction


### is this the imagenet trained one??
class SimClrPytorchResNet50(nn.Module):

    def __init__(self, num_classes):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(SimClrPytorchResNet50, self).__init__()

        self.convnet = resnet.resnet50(models.resnet.Bottleneck, [3, 4, 6, 3])       # "pretrained": use weights pre-trained on ImageNet

        # replace the very last layer from the original to a new one that outputs num_classes
        last_layer = self.convnet.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        self.convnet.fc = nn.Identity()                       # discard last layer...

        self.classifier = nn.Linear(in_features, num_classes)           # ...and create a new one
    

    def forward(self, x):
        '''
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-50 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-50 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.convnet(x)    # features.size(): [B x 512 x W x H]
        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return prediction
    
