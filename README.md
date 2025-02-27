# Simple GAN for MNIST Dataset

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.12%2B-orange)](https://pytorch.org/)
[![TensorBoard](https://img.shields.io/badge/tensorboard-enabled-yellow)](https://www.tensorflow.org/tensorboard/)

This project implements a simple Generative Adversarial Network (GAN) to generate MNIST-like images using PyTorch. It includes:
- A `Discriminator` class to distinguish between real and fake images.
- A `Generator` class to generate new images from random noise.
- A `main.py` script to train the GAN and log results to TensorBoard.

> **Code Source**: This implementation is inspired by [this YouTube tutorial](https://www.youtube.com/watch?v=OljTVUVzPpM) by the channel.

## Features

- Utilizes PyTorch for model implementation and training.
- Trains on the MNIST dataset with normalized input data.
- Logs loss values and visualizes generated images in TensorBoard.
- Supports GPU training for faster performance.

## Project Structure

```plaintext
.
├── src/
│   ├── models/
│   │   ├── discriminator.py  # Defines the Discriminator class
│   │   ├── generator.py      # Defines the Generator class
├── main.py                   # Main script for training the GAN
├── dataset/                  # Directory where MNIST data is downloaded
├── runs/                     # TensorBoard logs for real and fake images
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

## Requirements

All dependencies are listed in the `requirements.txt` file. Install them using the following command:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/simpleGAN.git
   cd simpleGAN
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate   # For Windows
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Training the GAN

1. Run the `main.py` script to start training:
   ```bash
   python main.py
   ```

2. View training progress and generated images using TensorBoard:
   ```bash
   tensorboard --logdir=runs
   ```

   Open the URL provided by TensorBoard to visualize logs and images.

## Outputs

- **Loss Values**: Discriminator and generator loss values printed in the console during training.
- **Generated Images**: Fake images generated at each epoch, logged in TensorBoard.

## Customization

- Modify hyperparameters such as learning rate, batch size, or the number of epochs in `main.py`.
- Adjust the architecture of the `Discriminator` or `Generator` in their respective files under `src/models/`.

## Acknowledgments

This project is inspired by [this YouTube tutorial](https://www.youtube.com/watch?v=OljTVUVzPpM).

## License

This project is licensed under the MIT License. See the [LICENSE.md](./LICENSE.md) file for details.

## Badges

- **Python Version**: 3.8+
- **PyTorch Version**: 1.12+
- **TensorBoard**: Enabled
