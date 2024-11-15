import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    A Generator model for a GAN (Generative Adversarial Network).

    This model takes in a latent vector (random noise) and generates an image (or data)
    representation that resembles real data when successful.

    Parameters:
    -----------
    z_dim : int
        The dimensionality of the input noise vector.
    img_dim : int
        The dimensionality of the generated output data, typically matching the
        dimensionality of the target data (e.g., for 28x28 images, img_dim would be 784).

    Attributes:
    -----------
    gen : nn.Sequential
        A sequential model defining the architecture of the generator,
        consisting of a fully connected hidden layer, an activation function,
        and an output layer with a Tanh function to normalize output data.
    """

    def __init__(self, z_dim, img_dim):
        super().__init__()

        # Define the generator model architecture
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),  # First fully connected layer
            nn.LeakyReLU(0.1),  # Leaky ReLU activation function
            nn.Linear(256, img_dim),  # Second fully connected layer (output layer)
            nn.Tanh()  # Tanh activation to scale output to [-1, 1]
        )

    def forward(self, X):
        """
        Forward pass through the Generator.

        Parameters:
        -----------
        X : torch.Tensor
            Input tensor representing the noise vector, with shape (batch_size, z_dim).

        Returns:
        --------
        torch.Tensor
            A tensor of shape (batch_size, img_dim), representing the generated data
            (e.g., a generated image flattened into a vector).
        """
        return self.gen(X)