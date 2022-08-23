from torch import optim
import torchvision.models as models
model = models.resnet18().cuda()
batch_size = 32 
optimizer = optim.Adam(model.parameters(), lr = 0.001)
epochs = 20
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')