import os
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
from scipy.misc import imsave
from collections import OrderedDict


class VAE(torch.nn.Module):
    def __init__(self, sample_size, latent_size):
        super(VAE, self).__init__()
        self.sample_size = sample_size
        self.latent_size = latent_size
        self.eh_dim = 128
        self.dh_dim = 128
        # encoder
        self.encoder_cnn_seq = nn.Sequential(
        torch.nn.Conv2d(1, 32, 5, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 5),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 5),
        torch.nn.ReLU()
        )
        self.encoder_linear_mu = torch.nn.Linear(128*16*16, self.latent_size)
        self.encoder_linear_sigma = torch.nn.Linear(128*16*16, self.latent_size)
        # decoder
        self.decoder_seq = nn.Sequential(
            torch.nn.ConvTranspose2d(1, 128, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 5),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 5, 2),
            torch.nn.Sigmoid()
        )
        self.decoder_linear = torch.nn.Linear(32*215*17, self.sample_size)
        self.decoder_sigmod = torch.nn.Sigmoid()


    def encoder(self, input):
        x = self.encoder_cnn_seq(input)
        return x.view(-1, 128*16*16)

    def sample_z(self, input):
        self.mu = self.encoder_linear_mu(input)
        log_sigma = self.encoder_linear_sigma(input)
        self.sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=self.sigma.size())).float().cuda()
        return self.mu + self.sigma * Variable(std_z, requires_grad=False).cuda()

    def decoder(self, input):
        net = self.decoder_seq(input)
        net = net.view(-1, 32*215*17)
        net = self.decoder_linear(net)
        return self.decoder_sigmod(net)

    def forward(self, input):
        net_encode = self.encoder(input)
        z = self.sample_z(net_encode)
        z = z.view(-1, 1, self.latent_size, 1)
        return self.decoder(z)


if __name__ == '__main__':
    sample_size = 784 #28 * 28
    mb_size = 32
    z_dim = 100

    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=mb_size, shuffle=True, num_workers=2)

    imgs_folder = './out'
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)
    
    vae = VAE(sample_size, z_dim)
    vae = vae.cuda()
    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    #criterion = nn.MSELoss()
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            x_mb, _ = data
            x_mb = Variable(x_mb.resize_(mb_size, 1, 28, 28), requires_grad=False).cuda()
            optimizer.zero_grad()
            #logits, dec = vae(x_mb)
            dec = vae(x_mb)
            kl_loss = 0.5 * torch.mean(vae.mu*vae.mu + vae.sigma*vae.sigma - torch.log(vae.sigma*vae.sigma) -1)
            #recon_loss = criterion(dec, x_mb)
            raw = Variable(x_mb.view(-1, 784).data, requires_grad=False).cuda()
            recon_loss = F.binary_cross_entropy(dec, raw)
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            l = loss.data[0]
            if i % 1000 == 0:
                print 'Epoch:%d, iteration:%d, loss:%f' % (epoch, i, l)
                z = Variable(torch.from_numpy(np.random.normal(0, 1, size=[10, z_dim])).float().cuda())
                z = z.view(-1, 1, z_dim, 1)
                imgs = vae.decoder(z).cpu().data.numpy()
                imsave(os.path.join(imgs_folder, '%d_%d.png') % (epoch, i), imgs[0].reshape(28, 28))
