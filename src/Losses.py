import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, num_classes=6, class_weights=None):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        
        # class_weights deve essere un torch.Tensor o None
        if class_weights is not None:
            self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        targets_onehot = torch.clamp(targets_onehot, min=1e-4, max=1.0)  # evita log(0)

        rce_loss = (-probs * torch.log(targets_onehot)).sum(dim=1).mean()

        return self.alpha * ce_loss + self.beta * rce_loss

class DynamicGCLoss(nn.Module):
    def __init__(self, trainset_size, device, q=0.3, k=0.7):
        super(DynamicGCLoss, self).__init__()
        self.q = q
        self.k = k
        self.trainset_size = trainset_size
        self.weight = torch.nn.Parameter(
            data=torch.ones(trainset_size, 1), requires_grad=False).to(device)

    def forward(self, logits, targets, indexes):

        
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, targets.unsqueeze(1))

        const_term = ((1 - self.k**self.q) / self.q)
        loss = ((1 - Yg**self.q) / self.q - const_term) * self.weight[indexes]
        return torch.mean(loss)

    def update_weight(self, logits, targets, indexes):
        
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, targets.unsqueeze(1))
        Lq = (1 - Yg**self.q) / self.q

        # Crea un tensore delle dimensioni di Lq e lo riempe di un lavore costante Lq(k) la cosidetta soglia
        Lqk = torch.full_like(Lq, fill_value=(1 - self.k**self.q) / self.q) 

        # Aggiorna i pesi solo dove Lq < Lqk, azzeriamo la loss quando Lq>Lq(k) esempi più incerti 
        condition = Lq < Lqk
        self.weight[indexes] = condition.float()
        print("Active weights this batch:", condition.sum().item(), "/", condition.numel())
    def update_q(self, new_q):
        self.q = new_q

    def update_k(self, new_k):
        self.k = new_k
        

