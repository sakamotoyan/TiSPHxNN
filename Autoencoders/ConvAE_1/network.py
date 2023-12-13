import torch.nn as nn

class ConvAutoencoder_1(nn.Module):
    """
    Convolutional autoencoder with 4 convolutional layers in the encoder and 4 transposed
    INPUT:  2-channel image (u, v) with size [256, 256]
    OUTPUT: 2-channel image (u, v) with size [256, 256]
    """
    
    def __init__(self, feature_vector_size):
        super(ConvAutoencoder_1, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # First convolutional layer taking input with 2 channels and producing 16 channels,
            # kernel_size=3 (3x3 kernel), stride=2 (reduces dimension by half), and padding=1.
            # Output tensor size: [batch_size, 16, 128, 128]
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  # Activation function

            # Second convolutional layer increasing channels from 16 to 32.
            # Output tensor size: [batch_size, 32, 64, 64]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # Activation function

            # Third convolutional layer increasing channels from 32 to 64.
            # Output tensor size: [batch_size, 64, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # Activation function

            # Fourth convolutional layer increasing channels from 64 to 128.
            # Output tensor size: [batch_size, 128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # Activation function

            nn.Flatten(),  # Flatten the output for feeding into a fully connected layer
            nn.Linear(128 * 16 * 16, feature_vector_size),  # Fully connected layer to create the bottleneck
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Expanding the flattened feature vector back to tensor shape [128, 16, 16]
            nn.Linear(feature_vector_size, 128 * 16 * 16),
            nn.Unflatten(1, (128, 16, 16)),  # Reshape to 3D tensor

            # Transposed convolution layers to upsample the image back to original dimensions
            # First layer upsamples from [128, 16, 16] to [64, 32, 32]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # Activation function

            # Second layer upsamples from [64, 32, 32] to [32, 64, 64]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # Activation function

            # Third layer upsamples from [32, 64, 64] to [16, 128, 128]
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # Activation function

            # Final layer that outputs the reconstructed image with 2 channels,
            # size [2, 256, 256]
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),  # Activation function for [-1,1] output range
        )

    def forward(self, x):
        # Forward pass: Input is encoded and then decoded
        encoded = self.encoder(x)  # Encoding input to a feature vector
        decoded = self.decoder(encoded)  # Decoding feature vector to reconstruct the image
        return decoded