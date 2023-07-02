from torchview import draw_graph

from ConvAE import ConvAutoencoder
from settings import *

autoencoder = ConvAutoencoder()
autoencoder.to(PLATFORM)
model_graph = draw_graph(autoencoder, input_size=(batch_size, 3, 128, 128), save_graph=True, device=PLATFORM, directory="NN_models/")
model_graph.visual_graph.render()
# model_graph.draw_graph()