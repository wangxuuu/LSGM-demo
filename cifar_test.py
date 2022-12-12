import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from itertools import chain
from models.scorenetwork import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid 
import os
import argparse

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.Conv2d(48, 96, 4, stride=2, padding=0),           # [batch, 96, 2, 2]
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=0),  # [batch, 48, 4, 4]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def encode(self, x):
        encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return encoded
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--diff", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train for.")
    args = parser.parse_args()

    # Create model
    autoencoder = create_model()
    # Create score network
    sigma =  5.0#@param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='../../Data/cifar10/', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../../Data/cifar10/', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load("./checkpoints/autoencoder.pkl"))
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = Variable(images.cuda())

        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())
    z_dim = 48*4*4
    sn = SN_Model(marginal_prob_std=marginal_prob_std_fn, embed_dim=20, dim=z_dim, drop_p=0.3, num_classes=10, device=device)
    sn_optim = torch.optim.Adam(sn.parameters(), lr = 0.005)
    # joint_optim = torch.optim.Adam(params=chain(autoencoder.parameters(), sn.parameters()))
    for epoch in range(args.epochs):
        running_loss = 0.0
        
        for i, (inputs, y) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)
            y = get_torch_vars(y)
            # ============ Forward ============
            encoded = autoencoder.encode(inputs)
            outputs = autoencoder.decode(encoded)
            loss = criterion(outputs, inputs)
            
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.diff:
                z = encoded.view(inputs.shape[0], -1)
                loss_sn = sn.loss(z.detach(), y)
                sn_optim.zero_grad()
                loss_sn.backward()
                sn_optim.step()
            

            # ============ Logging ============
            running_loss += loss.data
            if (i+1) % 200 == 0:
                print('Epoch [%d, %5d] Step [%d, %5d] loss: %.3f ' %
                      (epoch + 1, args.epochs, i+1, len(trainloader), running_loss / 200))
                running_loss = 0.0
        
        with torch.no_grad():
            # Save the reconstructed images
            x_concat = torch.cat([inputs, outputs], dim=3)
            save_image(x_concat, os.path.join('./results/cifar_test/', 'reconst-{}.png'.format(epoch+1)), nrow=4)
            # Save the diffused images
            if args.diff:
                sample = Euler_Maruyama_sampler(sn, marginal_prob_std_fn, diffusion_coeff_fn, dim=z_dim, batch_size=inputs.shape[0], num_steps=500, device=device, y=y)
                sample = sample.view(-1, 48, 4, 4)
                outputs = autoencoder.decode(sample)
                x_concat = torch.cat([inputs, outputs], dim=3)
                save_image(x_concat, os.path.join('./results/cifar_test/', 'diffused-{}.png'.format(epoch+1)), nrow=4)

    print('Finished Training')
    print('Saving Model...')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    torch.save(autoencoder.state_dict(), "./checkpoints/autoencoder.pkl")


if __name__ == '__main__':
    main()