import torch
import torch.nn.functional as F

class GCODLoss(torch.nn.Module):
    def __init__(self, gamma=0.2):
        super(GCODLoss, self).__init__()
        self.gamma = gamma 

    def forward(self, logits, labels):
        # Cross Entropy potremmo provare anche ad utilizzare SCE qua
        # IL reduction='none' è fondamentale restituisce la loss sull'intero batch e non la somma delle loss del batch
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (batch_size,)

        probs = F.softmax(logits, dim=1)
        true_probs = probs[range(len(labels)), labels]  #restituse la probabilità delle classi corrette del batch

        # Peso GCOD: vogliamo pesare di più i modelli con bassa confidenza
        weight = (true_probs.detach() ** self.gamma)

        loss = weight * ce_loss
        return loss.mean()

class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, num_classes=6):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        targets_onehot = torch.clamp(targets_onehot, min=1e-4, max=1.0)  # evita log(0)

        rce_loss = (-probs * torch.log(targets_onehot)).sum(dim=1).mean()

        return self.alpha * ce_loss + self.beta * rce_loss
