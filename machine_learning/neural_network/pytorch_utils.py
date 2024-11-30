import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_eval(
    model: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    device: str = "cpu",
    verbose: bool = False,
) -> None:
    """
    Trains and evaluates the model with tqdm progress bars.
    
    Args:
        model (nn.Module): The PyTorch model to train and evaluate.
        train_dataloader: DataLoader for the training set.
        valid_dataloader: DataLoader for the validation set.
        loss_fn: Loss function for optimization.
        optimizer: Optimizer for training.
        epochs (int): Number of training epochs.
        device (str): Device for computation ('cpu' or 'cuda').
        verbose (bool): If True, prints epoch-wise results.
    """
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device, epoch)
        
        # Validation phase
        val_loss = validation(model, valid_dataloader, loss_fn, device, epoch)
        
        # Verbose logging
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Valid Loss: {val_loss:.4f}")


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cpu",
    epoch: int = 0,
) -> float:
    """
    Performs a single epoch of training with a tqdm progress bar.

    Args:
        model (nn.Module): The PyTorch model to train.
        data_loader: DataLoader for the training set.
        loss_fn: Loss function for optimization.
        optimizer: Optimizer for training.
        device (str): Device for computation ('cpu' or 'cuda').
        epoch (int): Current epoch (used for tqdm description).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    
    with tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False) as tbar:
        for inputs_batch, outputs_batch in tbar:
            inputs_batch, outputs_batch = inputs_batch.to(device), outputs_batch.to(device)
            
            # Forward pass
            predictions = model(inputs_batch)
            loss = loss_fn(predictions, outputs_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            tbar.set_postfix(loss=loss.item())
    
    # Return average loss
    return total_loss / len(data_loader)


def validation(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: str = "cpu",
    epoch: int = 0,
) -> float:
    """
    Evaluates the model on the validation set with a tqdm progress bar.

    Args:
        model (nn.Module): The PyTorch model to validate.
        data_loader: DataLoader for the validation set.
        loss_fn: Loss function for evaluation.
        device (str): Device for computation ('cpu' or 'cuda').
        epoch (int): Current epoch (used for tqdm description).

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.inference_mode():
        with tqdm(data_loader, desc=f"Validation Epoch {epoch + 1}", leave=False) as tbar:
            for inputs_batch, outputs_batch in tbar:
                inputs_batch, outputs_batch = inputs_batch.to(device), outputs_batch.to(device)
                
                # Forward pass
                predictions = model(inputs_batch)
                loss = loss_fn(predictions, outputs_batch)
                
                total_loss += loss.item()
                tbar.set_postfix(loss=loss.item())
    
    # Return average loss
    return total_loss / len(data_loader)



if __name__ == '__main__':
    # Dummy dataset and model
    X_train = torch.rand(100, 10)
    y_train = torch.rand(100, 1)
    X_valid = torch.rand(20, 10)
    y_valid = torch.rand(20, 1)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)

    model = nn.Sequential(nn.Linear(10, 1))
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    train_eval(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True
    )
