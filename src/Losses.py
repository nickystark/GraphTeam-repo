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

class realGCODLoss(nn.Module):
    def __init__(self, num_samples, num_classes, gamma=0.2, lambda_u=1.0, lambda_kl=0.1):
        super(GCODLoss, self).__init__()
        self.gamma = gamma
        self.lambda_u = lambda_u
        self.lambda_kl = lambda_kl

        # Learnable confidence parameter u for each sample, initialized randomly
        self.u = nn.Parameter(torch.rand(num_samples))  # constrained to [0,1] via sigmoid
        self.num_classes = num_classes

    def forward(self, logits, labels, indices, acc_train=1.0):
        """
        Args:
            logits: shape (B, C)
            labels: shape (B,)
            indices: indices of samples in the full dataset (for indexing u)
            acc_train: scalar float, current accuracy of the model
        """
        probs = F.softmax(logits, dim=1)  # (B, C)
        log_probs = F.log_softmax(logits, dim=1)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        u_batch = torch.sigmoid(self.u[indices])  # (B,)
        u_batch_exp = u_batch.unsqueeze(1)  # shape (B, 1)

        # L1: weighted cross entropy
        L1 = -(u_batch_exp * one_hot * log_probs).sum(dim=1).mean()

        # L2: MSE between probs and labels
        L2 = ((probs - one_hot) ** 2).sum(dim=1).mean()

        # L3: KL divergence between u and the model's estimated reliability
        target_u = torch.full_like(u_batch, acc_train)
        L3 = F.kl_div(u_batch.log(), target_u, reduction='batchmean')

        total_loss = L1 + self.lambda_u * L2 + self.lambda_kl * L3
        return total_loss

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
