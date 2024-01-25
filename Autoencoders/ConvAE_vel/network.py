import torch.nn as nn

leakiness = 0.01

input_res = 256
feature_vector_size = 256

L0 = 2
L1 = 8
L2 = L1 * 2
L3 = L2 * 2
L4 = L3 * 2
L5 = L4 * 2
L6 = L5 * 2

LM=L5

feature_size = input_res // 2**5

class ConvAutoencoder(nn.Module):
    
    def __init__(self, type='train'):
        super(ConvAutoencoder, self).__init__()

        if type == 'train':
            dropout_probability = 0.2
        elif type == 'test':
            dropout_probability = 0.0

        self.feature_maps = {}

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(L0, L1, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(L1, L2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(L2, L3, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(L3, L4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(L4, L5, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            # nn.Conv2d(L5, L6, kernel_size=3, stride=2, padding=1),
            # nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
        )

        self.flatten = nn.Sequential(
            nn.Flatten(),
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(LM * feature_size * feature_size, feature_vector_size), 
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
        )

        self.unflatten = nn.Sequential(
            nn.Linear(feature_vector_size, LM * feature_size * feature_size),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
            nn.Unflatten(1, (LM, feature_size, feature_size)),
        )

        # Decoder
        self.decoder = nn.Sequential(
            
            # nn.ConvTranspose2d(L6, L5, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L5, L4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L4, L3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L3, L2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L2, L1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L1, L0, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, input, strategy='whole'):
        if strategy == 'whole':
            encoded = self.encoder(input)
            flatten = self.flatten(encoded)
            bottleneck = self.bottleneck(flatten)
            unflatten = self.unflatten(bottleneck)
            decoded = self.decoder(unflatten)
            return decoded
        if strategy == 'skip_bottleneck':
            encoded = self.encoder(input)
            decoded = self.decoder(encoded)
            return decoded
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.feature_maps[name] = output.detach()
        return hook