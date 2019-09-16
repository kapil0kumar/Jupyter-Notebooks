import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
import torch.optim as optim
from torch import nn
import json

from loss import GeneratorLoss
from model import Generator, Discriminator
from dataloader_utils import CustomDataset, ToTensor

try:
    os.mkdir("model_checkpoint")
except:
    pass

epochs = 100
batch_size = 4

checkpoint_path = "./model_checkpoint/"

gen = Generator()
dis = Discriminator()

data = CustomDataset(inp_dir="data/train/",
                     transform=transforms.Compose([ToTensor()]))

dataloader = DataLoader(data, batch_size=batch_size, num_workers=4)

generator_criterion = GeneratorLoss()
discriminator_criterion = torch.nn.BCELoss()

optimizerG = optim.Adam(gen.parameters())
optimizerD = optim.Adam(dis.parameters())

if torch.cuda.is_available():
    gen.cuda()
    dis.cuda()
    generator_criterion.cuda()
    discriminator_criterion.cuda()


for epoch in range(1, epochs + 1):

    running_results = {'batch_sizes': 0, 'd_loss': 0,
                       'g_loss': 0, 'epoch':epoch}

    gen.train()
    dis.train()
    for data in tqdm(dataloader):
        g_update_first = True
        running_results['batch_sizes'] += batch_size
        inp = data["input"].float()
        out = data["output"].float()

        real_img = Variable(out)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(inp)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = gen(z)

        ####################Training the discriminator###############

        dis.zero_grad()
        real_out = dis(real_img)
        fake_out = dis(fake_img)
        real_labels = Variable(torch.ones(batch_size,1))
        fake_labels = Variable(torch.zeros(batch_size,1))
        if torch.cuda.is_available():
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
        real_loss = discriminator_criterion(real_out,real_labels)
        fake_loss = discriminator_criterion(real_out,fake_labels)
    
        
        d_loss = 0.5*(real_loss + fake_loss)
        d_loss.backward(retain_graph=True)
        optimizerD.step()
        
        ###################Training the generator################

        gen.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        running_results['g_loss'] += g_loss.data * batch_size
        running_results['d_loss'] += d_loss.data * batch_size
        # break

    print("Epoch: {}/{},  Loss_D: {},  Loss_G: {}".format(epoch,
                                                          epochs,
                                                          running_results["d_loss"] / running_results["batch_sizes"],
                                                          running_results["g_loss"] / running_results["batch_sizes"]))
  
	
    torch.save(gen.state_dict(), '{}New2_gen_epoch_{}.pth'.format(checkpoint_path, epoch))
    # torch.save(dis.state_dict(), '{}dis_epoch_{}.pth'.format(checkpoint_path, epoch))
