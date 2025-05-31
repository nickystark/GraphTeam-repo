import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from src.models import GNN  # Usa l’encoder che già utilizzi o una sua versione semplificata

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class DGI(nn.Module):
    def __init__(self, encoder, hidden_channels):
        super(DGI, self).__init__()
        self.encoder = encoder
        self.sigmoid = nn.Sigmoid()
        self.disc = nn.Linear(hidden_channels, hidden_channels)
    
    def readout(self, z):
        return torch.sigmoid(torch.mean(z, dim=0))
    
    def forward(self, x, edge_index):
        pos_z = self.encoder(x, edge_index)
        summary = self.readout(pos_z)
        perm = torch.randperm(x.size(0))
        neg_z = self.encoder(x[perm], edge_index)
        pos_logits = self.disc(pos_z * summary)
        neg_logits = self.disc(neg_z * summary)
        return pos_logits, neg_logits

# Inside the DGI model class, modify the loss_fn method:
        # Inside the DGI model class, modify the loss_fn method:

    def loss_fn(self, pos_logits, neg_logits):
        # Ensure pos_logits and neg_logits are 1D tensors
        # This assumes the relevant score is in the first column (index 0) of the last dimension
        # If your model outputs a single score per sample, the squeeze() operation might be sufficient.
        # You might need to adjust the slicing [:, 0] or use squeeze() depending on your model's output.
        if pos_logits.ndim > 1:
            # Assuming the last dimension is the one to remove and the relevant score is at index 0
            # Adjust this if your model outputs a different structure
            if pos_logits.shape[-1] == 1:
                 pos_logits = pos_logits.squeeze(-1) # Remove the last dimension if it's 1
            else:
                 # If the last dimension is > 1, you need to decide which value to use.
                 # For standard binary classification, it should be a single score per sample.
                 # Let's assume the first value in the last dimension is the intended score.
                 pos_logits = pos_logits[:, 0]
    
        if neg_logits.ndim > 1:
             if neg_logits.shape[-1] == 1:
                 neg_logits = neg_logits.squeeze(-1)
             else:
                 neg_logits = neg_logits[:, 0]
    
    
        pos_labels = torch.ones(pos_logits.size(0)).to(pos_logits.device)
        neg_labels = torch.zeros(neg_logits.size(0)).to(neg_logits.device)
    
        loss_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
        loss_neg = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
        return loss_pos + loss_neg








def pretrain_dgi(data_loader, in_channels, hidden_channels, epochs=20, device='gpu'):
    encoder = Encoder(in_channels, hidden_channels).to(device)
    model = DGI(encoder, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pos_logits, neg_logits = model(data.x, data.edge_index)
            loss = model.loss_fn(pos_logits, neg_logits)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Pretrain Epoch {epoch}, Loss: {total_loss / len(data_loader):.4f}")
    return encoder

if __name__ == '__main__':
    # Carica il dataset (quello dei grafi) dalla tua repository
    # Assicurati di utilizzare il DataLoader corretto
    from data_loader import get_data  # Supponendo tu abbia una funzione per caricare i dati
    dataset = get_data(train=True)  # Separa i dati per pretraining, non servono le etichette
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "gpu")
    # Parametri da adattare ai tuoi dati: in_channels ad esempio  ? , hidden_channels ad es. 300
    pretrain_encoder = pretrain_dgi(data_loader, in_channels=7, hidden_channels=300, epochs=20, device=device)
    
    # Salva l’encoder pretrainato in un checkpoint per il fine-tuning
    torch.save(pretrain_encoder.state_dict(), "checkpoints/encoder_pretrained.pth")
    
