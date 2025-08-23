import torch.nn as nn
import torch

class EuclideanContrastiveLoss(nn.Module):
    def __init__(self, margin,pos_weight,neg_weight):
        super(EuclideanContrastiveLoss, self).__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def forward(self,reference_embedding,query_embedding,label):
        distances = torch.sqrt(torch.sum((query_embedding-reference_embedding)**2,dim=1)+1e-9)
        
        positive_loss = label*distances**2
        negative_loss = (1-label)*torch.clamp(self.margin-distances,min=0)**2
        return torch.mean(self.pos_weight*positive_loss + self.neg_weight*negative_loss)
    
