import torch

cosine_loss = torch.nn.CosineEmbeddingLoss()
input1 = torch.randn(100, 128, 128)
input2 = torch.randn(100, 128, 128)

print(input1.shape[0])
# loss = cosine_loss(input1.view(100,-1), input2.view(100,-1), torch.ones(100))
# print(loss)