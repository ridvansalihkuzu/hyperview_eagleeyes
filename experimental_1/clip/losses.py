"""
This code is generated by Ridvan Salih KUZU
LAST EDITED:  20.06.2022
ABOUT SCRIPT:
IT INCLUDES CUSTOM LOSS FUNCTIONS
"""

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


class CustomMSE(nn.Module):

    def __init__(self,device, idx=[0,1,2,3]):
        super(CustomMSE, self).__init__()

        y_base_fact = np.array([121764.2 / 1731.0, 394876.1 / 1731.0, 275875.1 / 1731.0, 11747.67 / 1731.0]) / np.array([325.0, 625.0, 400.0, 7.8])
        y_base_fact=y_base_fact[idx]
        self.y_base= torch.from_numpy(y_base_fact).float().to(device)
        self.loss1 = nn.MSELoss(reduction='mean')

    def forward(self, pred, labels=None):
        loss_raw = torch.mean(torch.square(pred - labels),dim=0)
        loss_base = torch.mean(torch.square(labels - self.y_base),dim=0)
        loss=loss_raw/loss_base
        #images_loss = self.loss1(pred, labels)

        return loss.mean(), loss



class SymmetricCrossEntropyLoss(nn.Module):

        def __init__(self, temperature=1):
            super(SymmetricCrossEntropyLoss, self).__init__()
            self.temperature = temperature
            self.loss1 = nn.CrossEntropyLoss(reduction='mean')
            self.loss2 = nn.CrossEntropyLoss(reduction='mean')

        def forward(self, feature_tuples, labels=None):
            logits_per_image, logits_per_text, image_features, text_features = feature_tuples

            images_similarity = image_features @ image_features.T
            texts_similarity = text_features @ text_features.T

            targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
            # targets = torch.from_numpy(np.arange(0,len(logits_per_image)))
            # targets=F.one_hot(targets, num_classes=len(logits_per_image))

            # images_loss = self.standard_cross_entropy(logits_per_image, targets, reduction='none')
            # texts_loss = self.standard_cross_entropy(logits_per_text, targets, reduction='none')

            images_loss = self.loss1(logits_per_image, targets)
            texts_loss = self.loss2(logits_per_text, targets)
            loss = (images_loss + texts_loss) / 2 * self.temperature

            return loss.mean()


