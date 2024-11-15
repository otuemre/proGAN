import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from src.models.discriminator import Discriminator
from src.models.generator import Generator

# Set device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
lr = 3e-4  # Learning rate
z_dim = 64  # Dimensionality of the noise vector
image_dim = 28 * 28 * 1  # Flattened image size for MNIST (28x28 grayscale)
batch_size = 32  # Number of samples per batch
num_epochs = 100  # Number of training epochs

# Initialize models, optimizers, and loss function
disc = Discriminator(image_dim).to(device)  # Discriminator model
gen = Generator(z_dim, image_dim).to(device)  # Generator model
fixed_noise = torch.randn((batch_size, z_dim)).to(device)  # Fixed noise for consistent fake image visualization
criterion = nn.BCELoss()  # Binary cross-entropy loss for real/fake classification

# Data preprocessing and loading
transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1] (for Tanh activation in generator)
])
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)  # MNIST dataset
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # DataLoader for batch processing

# Optimizers for discriminator and generator
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# TensorBoard writers for logging real and fake images
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

# Log model device for debugging
print("Discriminator on:", next(disc.parameters()).device)
print("Generator on:", next(gen.parameters()).device)

# Training loop
step = 0
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        # Prepare real data and noise for fake data
        real = real.view(-1, 784).to(device)  # Flatten real images and move to device
        batch_size = real.size(0)
        noise = torch.randn((batch_size, z_dim)).to(device)  # Generate random noise for generator

        # Generate fake data using the generator
        fake = gen(noise)

        # ---------------------
        # Train Discriminator
        # ---------------------
        # Loss for real data: maximize log(D(real))
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        # Loss for fake data: maximize log(1 - D(fake))
        disc_fake = disc(fake.detach()).view(-1)  # Detach fake to avoid gradients flowing to generator
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Combined discriminator loss and update
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # ---------------------
        # Train Generator
        # ---------------------
        # Loss for generator: maximize log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # ---------------------
        # Logging and Visualization
        # ---------------------
        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")

            with torch.no_grad():
                # Generate and log fixed fake images for consistent visualization
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)

                step += 1