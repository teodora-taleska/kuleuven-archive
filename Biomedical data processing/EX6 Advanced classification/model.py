import numpy as np
import skorch
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar


def get_simple_eegnet(
    X_val: np.array,
    y_val: np.array,
    verbose: int = 2,
    max_epochs: int = 20,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "./checkpoints/",
) -> skorch.NeuralNetClassifier:
    """
    Create a simple EEGNet model for classification of sleep stages from EEG signals.
    The model expects input signals of shape (n_samples, 3000) where 3000 is the number of time points. (30 seconds of 100 Hz sampling rate).

    The validation dataset needs to be scaled to the same range as the training dataset.

    Parameters:
        X_val (np.array): Validation features of shape (n_samples, 3000).
        y_val (np.array): Validation labels of shape (n_samples,).
        max_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        device (str): Device to run the model on ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.
    Returns:
        skorch.NeuralNetClassifier: Configured neural network classifier ready for training.
    """

    monitor = skorch.callbacks.Checkpoint(
        monitor="valid_acc_best", load_best=True, dirname=checkpoint_dir
    )

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_val), y=y_val
    )

    train_val_split = TrainValSplit(X_val, y_val)

    return NeuralNetClassifier(
        SimpleEEGNet(),
        max_epochs=max_epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        criterion=nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
        ),
        iterator_train__shuffle=True,
        train_split=train_val_split,
        callbacks=[monitor],
        verbose = verbose,
        callbacks__print_log__sink=print,  # Output su una riga
        callbacks__print_log__tablefmt='plain',  # Formato più semplice
    )


class TrainValSplit:
    def __init__(self, X_val, y_val):
        self.val_dataset = skorch.dataset.Dataset(X_val, y_val)

    def __call__(self, train_dataset, y, **fit_params):
        return train_dataset, self.val_dataset


class SimpleEEGNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(32, 64, kernel_size=10, stride=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(64, 128, kernel_size=10, stride=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Flatten(),
        )

        out_size = self.conv(torch.zeros(1, 1, 3000)).shape[1]
        self.clf = nn.Linear(out_size, 5)

    def forward(self, signal):
        features = self.extract_features(signal)
        return self.clf(features)
    
    def extract_features(self, signal):
        signal = signal.unsqueeze(1)
        features = self.conv(signal)
        return features
