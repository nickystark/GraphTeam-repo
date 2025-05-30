import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=6, class_weights=None):
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
    def __init__(self, trainset_size, device, q=0.1, k=0.3):
        super(DynamicGCLoss, self).__init__()
        self.q = q
        self.k = k
        self.trainset_size = trainset_size
        self.weight = torch.nn.Parameter(
            data=torch.ones(trainset_size, 1), requires_grad=False).to(device)

    def forward(self, logits, targets, indexes):

        
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, targets.unsqueeze(1))
        Yg = torch.clamp(Yg, min=1e-6, max=1.0)

        #const_term = ((1 - self.k**self.q) / self.q)
        loss = ((1 - Yg**self.q) / self.q ) * self.weight[indexes]
        return torch.mean(loss)

    '''def update_weight(self, logits, targets, indexes):
        
        p  = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, targets.unsqueeze(1))
        Yg = torch.clamp(Yg, min=1e-6, max=1.0)
        
        Lq = (1 - Yg**self.q) / self.q
        
        # Crea un tensore delle dimensioni di Lq e lo riempe di un lavore costante Lq(k) la cosidetta soglia
        Lqk_value = (1 - self.k**self.q) / self.q
        Lqk = torch.full_like(Lq, fill_value=Lqk_value) 

        # Aggiorna i pesi solo dove Lq < Lqk, azzeriamo la loss quando Lq>Lq(k) esempi più incerti 
        condition = Lq < Lqk
        self.weight[indexes] = condition.float()
        print("Yg stats -> mean:", Yg.mean().item(), "max:", Yg.max().item())
        print("Lq stats -> mean:", Lq.mean().item(), "min:", Lq.min().item(), "max:", Lq.max().item())
        print("Lqk value:", Lqk_value)
        print("Active weights this batch:", condition.sum().item(), "/", condition.numel())'''
    
    def update_q(self, new_q):
        self.q = new_q

    def update_k(self, new_k):
        self.k = new_k

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, targets.unsqueeze(1))
        Yg = torch.clamp(Yg, min=1e-6, max=1.0)
        Lq = (1 - Yg**self.q) / self.q

        # Calcola la soglia Lq(k)
        Lqk_value = (1 - self.k**self.q) / self.q

        # Ad esempio, usa una sigmoid per mappare Lq a un peso tra 0 e 1
        # Maggiore è Lq, minore dovrebbe essere il peso.
        # La funzione sigmoid può essere parametrizzata in modo da centrare la transizione attorno a Lqk_value.
        scale = 10.0  # Un parametro da sperimentare.
        new_weights = torch.sigmoid(-scale * (Lq - Lqk_value))

        self.weight[indexes] = new_weights
        #print("Average weight this batch:", new_weights.mean().item())


''' def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, targets.unsqueeze(1))
        dynamic_loss = ((1 - Yg**self.q) / self.q) * self.weight[indexes]
        dynamic_loss = torch.mean(dynamic_loss)
    
        ce_loss = F.cross_entropy(logits, targets)
    
        # Peso per combinare le perdite, da sperimentare ad es. 0.5 e 0.5 o altri pesi
        return 0.5 * ce_loss + 0.5 * dynamic_loss
'''

class SCELossWithMAE(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, gamma=1.0, num_classes=6, smoothing=0.1, class_weights=None):
        super(SCELossWithMAE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.smoothing = smoothing
        
        if class_weights is not None:
            self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce = torch.nn.CrossEntropyLoss()

    def smooth_one_hot(self, targets):
        confidence = 1.0 - self.smoothing
        smoothing_value = self.smoothing / (self.num_classes - 1)
        one_hot = torch.full((targets.size(0), self.num_classes), smoothing_value).to(targets.device)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)
        return one_hot

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        
        # Soft label
        soft_targets = self.smooth_one_hot(targets)

        # RCE: usa soft target invece che hard one-hot
        rce_loss = (-probs * torch.log(torch.clamp(soft_targets, min=1e-4))).sum(dim=1).mean()

        # MAE: distanza tra predizione e soft label
        mae = torch.nn.functional.l1_loss(probs, soft_targets)

        return self.alpha * ce_loss + self.beta * rce_loss + self.gamma * mae

    def update_alfa(self, alfa):
        self.alpha = alfa

    def update_beta(self, beta):
        self.beta = beta

    def update_beta(self, gamma):
        self.gamma = gamma
