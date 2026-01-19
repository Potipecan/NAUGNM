from collections import namedtuple

import torch
import torch.nn as nn
import torchvision.io
from torch import Tensor
import os
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.mps.is_available() else
                      'cpu')

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,  padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,  padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2,  padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.data_dir = dir
        self.images = os.listdir(dir)

    # Defining the length of the dataset
    def __len__(self):
        return len(self.images)

    # Defining the method to get an item from the dataset
    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.images[index])
        image = torchvision.io.decode_image(image_path)
        return image

HyperParams = namedtuple('HyperParams', [
    'gen_learning_rate',
    'dis_learning_rate',
    'epochs',
])

def create_example(gen: Gen, epoch_i, output_folder):
    gen.eval()
    with torch.no_grad():
        seed = torch.randn((64, 100, 1, 1), device=device)
        image_data = gen(seed)

        c, h, w = image_data.shape[1:]
        # Reshape
        x = image_data.view(8, 8, c, h, w)  # (8, 8, c, h, w)
        x = x.permute(2, 0, 3, 1, 4)  # (c, 8, h, 8, w)
        x = x.reshape(c, 8 * h, 8 * w)  # (c, 8h, 8w)

    img = to_pil_image(x.cpu()).resize((256, 256))
    img.save(os.path.join(output_folder, f'epoch_{str(epoch_i)}.png'))

    gen.train()

def train(gen: Gen, dis: Discriminator, h: HyperParams, real_data: torch.utils.data.DataLoader, image_output_folder=None):
    gen = gen.to(device)
    dis = dis.to(device)

    gen.train()
    dis.train()

    gen_loss = []
    dis_loss = []

    loss_f = torch.nn.BCELoss()

    optimizer_gen = torch.optim.Adam(gen.parameters(), h.gen_learning_rate)
    optimizer_dis = torch.optim.Adam(dis.parameters(), h.dis_learning_rate)

    for epoch in range(h.epochs):
        dis_c_loss = 0
        gen_c_loss = 0
        total = 0

        for batch_idx, data in enumerate(real_data):
            data = data.to(device=device) / 255.0
            pred_real = dis(data)
            n = data.size(0)
            total += n

            optimizer_dis.zero_grad()
            optimizer_gen.zero_grad()
            gen_seed = torch.randn((n, 100, 1, 1), device=device, dtype=data.dtype)
            gen_data = gen(gen_seed)

            pred_gen = dis(gen_data)

            loss_real_data = loss_f(pred_real, torch.ones_like(pred_real))
            loss_gen_data = loss_f(pred_gen, torch.zeros_like(pred_gen))
            loss_real_data.backward()
            loss_gen_data.backward()
            # loss_discriminator = loss_real_data + loss_gen_data
            loss_generator = loss_f(pred_gen, torch.ones_like(pred_gen))

            # loss_discriminator.backward(retain_graph=True)
            loss_generator.backward()
            # loss_discriminator.detach_()

            dis_c_loss += (loss_real_data.item() + loss_gen_data.item) * n
            gen_c_loss += loss_generator.item() * n

            optimizer_dis.step()
            optimizer_gen.step()

            del data
            del gen_data
            del pred_real
            del pred_gen
            del gen_seed
            torch.cuda.empty_cache()


        dis_c_loss /= total
        gen_c_loss /= total

        dis_loss.append(dis_c_loss)
        gen_loss.append(gen_c_loss)

        print(f'Epoch {epoch + 1:03}: gen loss: {gen_c_loss:.4f}, dis loss: {dis_c_loss:.4f}')

        if image_output_folder is not None:
            create_example(gen, epoch + 1, image_output_folder)

    return pd.DataFrame({
        'generator_loss': gen_loss,
        'discriminator_loss': dis_loss,
    })