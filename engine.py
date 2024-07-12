"""
Constains functions for training and testing a Pytorch model.
"""
import time
import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """

    # set model to training mode
    model.train()

    # initialize variables to keep track of loss and accuracy
    train_loss, train_acc = 0, 0

    # iterate over the training data
    for batch, (X, y) in enumerate(dataloader):
        # send data to device
        X, y = X.to(device), y.to(device)

        # 1. forward pass
        y_pred = model(X)

        # 2. calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero gradient
        optimizer.zero_grad()

        # 4. backward pass
        loss.backward()

        # 5. optimizer step
        optimizer.step()

        # calculate and accumulate accuracy metric across batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    # Adjust metrics to get average loss and accuracy per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """vals a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a valing dataset.

    Args:
    model: A PyTorch model to be valed.
    dataloader: A DataLoader instance for the model to be valed on.
    loss_fn: A PyTorch loss function to calculate loss on the val data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of valing loss and valing accuracy metrics.
    In the form (val_loss, val_accuracy). For example:

    (0.0223, 0.8985)
    """
    # put model into eval mode
    model.eval()

    # initialize variables to keep track of loss and accuracy
    val_loss, val_acc = 0, 0

    # turn on inference mode
    with torch.inference_mode():
        # iterate over the val dataloader
        for batch, (X, y) in enumerate(dataloader):
            # send data to device
            X, y = X.to(device), y.to(device)

            # 1. forward pass
            val_pred_logits = model(X)

            # 2. calculate and accumulate loss
            val_loss += loss_fn(val_pred_logits, y).item()

            # calculate and accumulate accuracy metric across batches
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += (val_pred_labels == y).sum().item()/len(val_pred_labels)

    # Adjust metrics to get average loss and accuracy per batch
    val_loss /= len(dataloader)
    val_acc /= len(dataloader)

    return val_loss, val_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              val_loss: [...],
              val_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              val_loss: [1.2641, 1.5706],
              val_acc: [0.3400, 0.2973]}
    """
    # initialize dictionary to store metrics
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
        "training_time": ""
    }

    # make sure model on target device and get start time
    model.to(device)
    start_time = time.time()

    # iterate over the number of epochs
    for epoch in tqdm(range(epochs)):
        # train the network
        train_loss, train_acc = train_step(model=model, 
                                           dataloader=train_dataloader, 
                                           loss_fn=loss_fn, 
                                           optimizer=optimizer, 
                                           device=device)
        val_loss, val_acc = val_step(model=model, 
                                     dataloader=val_dataloader, 
                                     loss_fn=loss_fn, 
                                     device=device)
        
        # print out what's happening
        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # store metrics in results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    end_time = time.time()
    training_time = end_time - start_time
    minutes, seconds = divmod(int(training_time), 60)
    results["training_time"] = f"Training time: {minutes}m | {seconds}s"
    print(f"Training time: {minutes}m | {seconds}s")

    # test the model
    model.eval()

    # initialize variables to keep track of loss and accuracy
    test_loss, test_acc = 0, 0

    # turn on inference mode
    with torch.inference_mode():
        # iterate over the test dataloader
        for batch, (X, y) in enumerate(test_dataloader):
            # send data to device
            X, y = X.to(device), y.to(device)

            # 1. forward pass
            test_pred_logits = model(X)

            # 2. calculate and accumulate loss
            test_loss += loss_fn(test_pred_logits, y).item()

            # calculate and accumulate accuracy metric across batches
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)  

    # Adjust metrics to get average loss and accuracy per batch
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")
    return results