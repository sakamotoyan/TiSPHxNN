import torch.nn as nn

dropout_probability = 0.2
leakiness = 0.01
class ConvAutoencoder_1(nn.Module):
    
    def __init__(self):
        super(ConvAutoencoder_1, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
        )

        self.middle = nn.Sequential(   
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(8, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        middle = self.middle(encoded)
        decoded = self.decoder(encoded)
        return decoded