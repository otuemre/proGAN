import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    A Discriminator model for a GAN (Generative Adversarial Network).

    This model takes in a feature vector (typically the flattened image data)
    and outputs a single value between 0 and 1, where:
    - 1 represents the classification of "real" data.
    - 0 represents the classification of "fake" (generated) data.

    Parameters:
    -----------
    in_features : int
        The number of input features, which should match the dimensionality
        of the input data (e.g., for 28x28 images, in_features would be 784).

    Attributes:
    -----------
    disc : nn.Sequential
        A sequential model defining the architecture of the discriminator,
        consisting of a fully connected hidden layer, an activation function,
        and an output layer with a sigmoid function.
    """

    def __init__(self, img_dim):
        super().__init__()

        # Define the discriminator model architecture
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),    # First fully connected layer
            nn.LeakyReLU(0.1),                          # Leaky ReLU activation function
            nn.Linear(128, 1),    # Second fully connected layer (output layer)
            nn.Sigmoid()                                # Sigmoid activation for binary classification
        )

    def forward(self, X):
        """
        Forward pass through the Discriminator.

        Parameters:
        -----------
        X : torch.Tensor
            Input tensor representing either real or generated data, with shape (batch_size, in_features).

        Returns:
        --------
        torch.Tensor
            A tensor of shape (batch_size, 1) with values between 0 and 1,
            representing the probability that the input data is "real".
        """
        return self.disc(X)