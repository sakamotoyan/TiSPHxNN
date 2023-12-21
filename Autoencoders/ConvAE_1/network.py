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
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), 

            nn.ReLU(), 
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, feature_vector_size), 
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_vector_size, 64 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),  
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), 

            nn.ConvTranspose2d(8, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded