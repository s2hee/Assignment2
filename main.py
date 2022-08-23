import model
from train import train
from test import test
from dataset import trainloader, testloader
from model import epochs, optimizer

for epoch in range(1, epochs + 1):
    train(model, trainloader, optimizer, epoch)
    test_loss, test_accuracy = test(model, testloader)
    print("[{}] Test Loss: {:.4f}, accuracy: {:.2f}%\n".format(epoch, test_loss, test_accuracy))