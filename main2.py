import argparse
import os
import torch
import sklearn

from torch_geometric.loader import DataLoader
from src.loadData import GraphDataset
from src.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

from sklearn.metrics import f1_score

from src.models import GNN 
from src.gin_normal import GNN_Costume
from src.Losses import SCELoss
from src.Losses import DynamicGCLoss
from src.Losses import SCELossWithMAE
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
import math

# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, criterion, scheduler, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
    
    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader) ,  correct / total

def f1(val_loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1).cpu().numpy()
            label = batch.y.cpu().numpy()

            all_preds.extend(pred)
            all_labels.extend(label)

    return f1_score(all_labels, all_preds, average='macro')


def evaluate(data_loader, model, device,criterion, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
            else:
                predictions.extend(pred.cpu().numpy())
    if calculate_accuracy:
        accuracy = correct / total
        return  total_loss / len(data_loader),accuracy
    return predictions


def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()




def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3

    if args.residual == 1: 
        res=True
    else:
        res=False
    
    

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, JK = args.JK, residual= res, graph_pooling=args.readout).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True , JK = args.JK, residual= res, graph_pooling=args.readout).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, JK = args.JK, residual= res, graph_pooling=args.readout).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, JK = args.JK, residual= res, graph_pooling=args.readout).to(device)
    elif args.gnn == 'simple_gin':
        model = GNN_Costume(  num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio , JK = args.JK, residual= res, graph_pooling=args.readout).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_d)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, verbose=True
     )
   

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well


    # Define checkpoint path relative to the script's directory
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

 
    
    # If train_path is provided, train the model
    if args.train_path:
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(args.val_test * len(full_dataset))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        #weight = tensor([1.3773, 0.9781, 0.5685, 0.9495, 0.9572, 2.5337])

      
        criterion = SCELoss( alpha=0.1, beta=1.0)
   

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
   
        # Training loop
        num_epochs = args.epochs
        best_val_accuracy = 0.0
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # Calculate intervals for saving checkpoints
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        for epoch in range(num_epochs):

            total_epochs = num_epochs
            initial_alpha = 0.3
            final_alpha = 0.01

            initial_beta = 0.7
            final_beta = 2.0

            if epoch >= 20:
                progress = (epoch - 20) / (total_epochs - 20)
                progress = min(max(progress, 0.0), 1.0)  # clamp tra 0 e 1

                new_alpha = initial_alpha * (1 - progress) + final_alpha * progress
                new_beta = initial_beta * (1 - progress) + final_beta * progress
            else:
                new_alpha = initial_alpha
                new_beta = initial_beta

            print(f"Epoch {epoch}: alpha={new_alpha:.4f}, beta={new_beta:.4f}")
            criterion.update_alfa(new_alpha)
            criterion.update_beta(new_beta)
            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, scheduler, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )
            val_loss,val_acc = evaluate(val_loader, model, device, criterion, calculate_accuracy=True)
            val_f1 = f1(val_loader, model, device)

            scheduler.step(val_f1)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f},val_f1_score: {val_f1:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f},val_f1_score: {val_f1:.4f}")

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            patience=30

            if val_f1 > best_val_accuracy:
                best_val_accuracy = val_f1
                early_stopping_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")
            else :  
                early_stopping_counter += 1
                print(f"EarlyStopping counter: {early_stopping_counter}/{patience}")
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    break

        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
        plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plotsVal"))

    # Generate predictions for the test set using the best model
    model.load_state_dict(torch.load(checkpoint_path))
    predictions = evaluate(test_loader, model, device, criterion, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)
    del model, optimizer, scheduler, train_loader, val_loader, criterion
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--JK', type=str, default='last', help="Choose the JK setting ('last' or scale parse 'sum') (default: last")
    parser.add_argument('--residual', type=int, default=0, help='Set 0 to disable residual connections, 1 to enable (default: 0)')
    parser.add_argument('--readout', type=str, default="sum", choices=["sum", "mean", "max", "attention", "set2set"],
    help="Type of readout function to use: 'sum', 'mean', 'max', 'attention', or 'set2set'."
    )
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--baseline_mode', type=int, default=1, help='1 for SIMMETRYC loss 2 for GCOD loss (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.2, help='gamma only for GCODloss (default: 0.2)')
    parser.add_argument('--alfa', type=float, default=0.7, help='alfa only for SCEloss (default: 0.7)')
    parser.add_argument('--beta', type=float, default=0.3, help='beta only for SCEloss (default: 0.3)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--w_d', type=float, default=0.00001, help='weight decay (default: 0.00001)')
    parser.add_argument('--val_test', type=float, default=0.2, help='learning rate (default: 0.2)')
    parser.add_argument('--weight', type=int, default=0, help='loss with weight (default: 0)')
    
    

    args = parser.parse_args()
    
  
    main(args)
