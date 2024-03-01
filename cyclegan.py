import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import itertools
import random
from tqdm import tqdm

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels, num_residual_blocks=9): # 9
        super(Generator, self).__init__()

        model = [nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)]

        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features//2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        model += [nn.Conv2d(64, input_channels, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a buffer of negative size"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class WatermarkDataset(Dataset):
    def __init__(self, marked_dir, unmarked_dir, transform=None):
        self.marked_dir = marked_dir
        self.unmarked_dir = unmarked_dir
        self.transform = transform

        self.marked_images = os.listdir(marked_dir)
        self.unmarked_images = os.listdir(unmarked_dir)
        self.length_dataset = max(len(self.marked_images), len(self.unmarked_images)) 

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        marked_img = self.marked_images[index % len(self.marked_images)]
        unmarked_img = self.unmarked_images[index % len(self.unmarked_images)]

        marked_path = os.path.join(self.marked_dir, marked_img)
        unmarked_path = os.path.join(self.unmarked_dir, unmarked_img)

        marked_img = Image.open(marked_path).convert("RGB")
        unmarked_img = Image.open(unmarked_path).convert("RGB")
        if self.transform is not None:
            marked_img = self.transform(marked_img)
            unmarked_img = self.transform(unmarked_img)

        return marked_img, unmarked_img

class CycleGAN(nn.Module):
    def __init__(self, input_channels, device):
        super(CycleGAN, self).__init__()
        self.G_AB = Generator(input_channels).to(device)
        self.G_BA = Generator(input_channels).to(device) 
        self.D_A = Discriminator(input_channels).to(device)  
        self.D_B = Discriminator(input_channels).to(device)

        self.optimizer_G = optim.Adam(list(self.G_AB.parameters()) + list(self.G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        self.device = device

    def save_images(self, epoch, batch, marked, unmarked, fake_unmarked1, fake_unmarked2, directory="./saved_images", n_images=5):
        os.makedirs(directory, exist_ok=True)
        images = torch.cat((marked[:n_images], unmarked[:n_images], fake_unmarked1[:n_images], fake_unmarked2[:n_images]), 0)
        save_image(images, f"{directory}/{epoch}-{batch}.png", nrow=n_images)
        
    def save_model_checkpoint(self, epoch, directory="./model_checkpoints"):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.G_AB.state_dict(), f"{directory}/G_AB_epoch_{epoch}.pth")
        torch.save(self.G_BA.state_dict(), f"{directory}/G_BA_epoch_{epoch}.pth")

    def train(self, dataloader, n_epochs=100):
        for epoch in range(n_epochs):
            for i, batch in enumerate(tqdm(dataloader)):
                real_A, real_B = batch
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                
                # Generate a batch of valid labels
                valid = torch.ones((real_A.size(0), *self.D_A(real_A).shape[1:]), requires_grad=False).to(self.device)
                fake = torch.zeros((real_A.size(0), *self.D_A(real_A).shape[1:]), requires_grad=False).to(self.device)
                # ------------------
                #  Train Generators
                # ------------------
                self.optimizer_G.zero_grad()

                # Identity loss
                same_B = self.G_AB(real_B)
                loss_identity_B = self.criterion_identity(same_B, real_B)
                same_A = self.G_BA(real_A)
                loss_identity_A = self.criterion_identity(same_A, real_A)

                # GAN loss
                fake_B = self.G_AB(real_A)
                loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
                fake_A = self.G_BA(real_B)
                loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)

                # Cycle loss
                recovered_A = self.G_BA(fake_B)
                loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)
                recovered_B = self.G_AB(fake_A)
                loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B)

                total_cycle_loss = loss_cycle_ABA + loss_cycle_BAB

                total_loss_G_AB = total_cycle_loss + loss_identity_B + loss_GAN_AB
                total_loss_G_BA = total_cycle_loss + loss_identity_A + loss_GAN_BA 
                
                total_loss_G = total_loss_G_AB + total_loss_G_BA
                total_loss_G.backward()
                self.optimizer_G.step()

                # -----------------------
                #  Train Discriminator A
                # -----------------------
                self.optimizer_D_A.zero_grad()

                # Real loss
                loss_real_A = self.criterion_GAN(self.D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
                loss_fake_A = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)

                # Total loss
                loss_D_A = (loss_real_A + loss_fake_A) / 2
                loss_D_A.backward()
                self.optimizer_D_A.step()

                # -----------------------
                #  Train Discriminator B
                # -----------------------
                self.optimizer_D_B.zero_grad()

                # Real loss
                loss_real_B = self.criterion_GAN(self.D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
                loss_fake_B = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)

                # Total loss
                loss_D_B = (loss_real_B + loss_fake_B) / 2
                loss_D_B.backward()
                self.optimizer_D_B.step()
                
                # Progress logging
                if i % 50 == 0:
                    print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}] [D_A loss: {loss_D_A.item()}, D_B loss: {loss_D_B.item()}] [G_AB loss: {total_loss_G_AB.item()}, G_BA loss: {total_loss_G_BA.item()}]")
                    self.save_images(epoch, i, real_A.cpu(), real_B.cpu(), fake_A.cpu(), fake_B.cpu())
                    
            self.save_model_checkpoint(epoch)
# Main script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = 3

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    marked_dir = './B'
    unmarked_dir = './A'
    dataset = WatermarkDataset(marked_dir, unmarked_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CycleGAN(input_channels, device)
    model.train(dataloader)

if __name__ == "__main__":
    main()
