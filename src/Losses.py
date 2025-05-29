import torch 
import torch.nn as nn
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

class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)

