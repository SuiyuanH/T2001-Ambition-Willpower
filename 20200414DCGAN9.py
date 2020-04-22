# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms
import time
import os
import shutil
import winsound
import matplotlib.pyplot as plt
import cv2

random_nums = 15
graph_size = 32
new_graph_size = 4
save_path = './nameimg/epoch_{}.jpg'

def to_img(x):
    x = x.clamp(0, 1)
    return x
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
class SimpleFC(nn.Module):
    def __init__(self, dim_in, dim_out):
        nn.Module.__init__(self)
        self.fc = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)
        self.af = nn.ReLU()
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.af(x)
        return x
class discriminator(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) # 32 - 16
        self.l1_bn = nn.BatchNorm2d(16)
        self.l1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l1_af = nn.LeakyReLU(0.2, inplace=True)
        self.l2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 16 - 8
        self.l2_bn = nn.BatchNorm2d(32)
        self.l2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l2_af = nn.LeakyReLU(0.2, inplace=True)
        self.l3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 8 - 4
        self.l3_bn = nn.BatchNorm2d(64)
        self.l3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l3_af = nn.LeakyReLU(0.2, inplace=True)
        self.l4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 4 - 2
        self.l4_bn = nn.BatchNorm2d(128)
        self.l4_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.l4_af = nn.LeakyReLU(0.2, inplace=True)
        self.l5 = nn.Linear(128*2*2, 11)
    def forward(self, x):
        x = self.l1(x)
        x = self.l1_bn(x)
        x = self.l1_pool(x)
        x = self.l1_af(x)
        x = self.l2(x)
        x = self.l2_bn(x)
        x = self.l2_pool(x)
        x = self.l2_af(x)
        x = self.l3(x)
        x = self.l3_bn(x)
        x = self.l3_pool(x)
        x = self.l3_af(x)
        x = self.l4(x)
        x = self.l4_bn(x)
        x = self.l4_pool(x)
        x = self.l4_af(x)
        x = x.view(-1, 128*2*2)
        x = self.l5(x)
        x, y = x[:, 0], x[:, 1:]
        return x, y
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class generator(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.FC1 = nn.Linear(random_nums, 32*32)
        self.FC1_bn = nn.BatchNorm1d(32*32)
        self.FC1_af = nn.ReLU(inplace=True)
        self.l1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.l1_bn = nn.BatchNorm2d(16)
        self.l1_af = nn.ReLU(inplace=True)
        self.l2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.l2_bn = nn.BatchNorm2d(64)
        self.l2_af = nn.ReLU(inplace=True)
        self.l3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.l3_bn = nn.BatchNorm2d(256)
        self.l3_af = nn.ReLU(inplace=True)
        self.l4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.l4_af = nn.Sigmoid()
    def forward(self, x):
        x = self.FC1(x)
        x = self.FC1_bn(x)
        x = self.FC1_af(x)
        x = x.view(-1, 1, 32, 32)
        x = self.l1(x)
        x = self.l1_bn(x)
        x = self.l1_af(x)
        x = self.l2(x)
        x = self.l2_bn(x)
        x = self.l2_af(x)
        x = self.l3(x)
        x = self.l3_bn(x)
        x = self.l3_af(x)
        x = self.l4(x)
        x = self.l4_af(x)
        return x
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

if os.path.exists('./nameimg'):
    shutil.rmtree('./nameimg')
    time.sleep(0.5)
os.mkdir('./nameimg')

D = discriminator()
G = generator()
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()
D.weight_init(mean=0.0, std=0.02)
G.weight_init(mean=0.0, std=0.02)

d_batch_size = 16
g_batch_size = 2
data_tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
train_dataset = datasets.MNIST(root='../../dataset', train=True, transform=data_tf)
train_loader = iter(DataLoader(train_dataset, batch_size=d_batch_size, shuffle=True))

lr1 = 0.0003
wd1 = 0.000000001
lr2 = 0.003
wd2 = 0.000000001
optimizer1 = optim.Adam(D.parameters(), lr=lr1, weight_decay = wd1, betas=(0.5, 0.9))
optimizer2 = optim.Adam(G.parameters(), lr=lr2, weight_decay = wd2, betas=(0.5, 0.9))

time0 = time.time()

num_epochs = 10000
clamper = 0.05
print_num = 10
d_sub_epochs = 1
g_sub_epochs = 1
d_CE = 0.001
g_CE = 0.001

real_list = []
fake_list = []
cheat_list = []

epoch = 0
to_continue = ''
while to_continue != 'exit':
    for p in D.parameters():
        p.requires_grad_(True)
    for sub_epoch in range(d_sub_epochs):
        # 录入数据
        try:
            img, label = next(train_loader)
            real_img = img.type(torch.float)
        except StopIteration:
            train_loader = iter(DataLoader(train_dataset, batch_size=d_batch_size, shuffle=True))
            img, label = next(train_loader)
            real_img = img.type(torch.float)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
            label = label.cuda()
        # 假图生成
        random_fake = torch.rand((d_batch_size, random_nums))
        if torch.cuda.is_available():
            random_fake = random_fake.cuda()
        fake_img = G(random_fake)
        # D前馈
        real_out, real_label = D(real_img)
        fake_out, fake_label = D(fake_img)
        loss_d = - real_out.mean() + fake_out.mean() + torch.log(torch.tensor(epoch + 1, dtype=torch.float32).cuda()) * nn.CrossEntropyLoss()(real_label, label)
        # D反馈
        optimizer1.zero_grad()
        loss_d.backward()
        optimizer1.step()
        for p in D.parameters():
            p.data.clamp_(- clamper, clamper)
    for p in D.parameters():
        p.requires_grad_(False)
    for p in G.parameters():
        p.requires_grad_(True)
    for sub_epoch in range(g_sub_epochs):     
        # 随机扰动
        random_label = torch.randint(10, (g_batch_size,))
        random_cheat = torch.zeros((g_batch_size, 10))
        for i in range(g_batch_size):
            random_cheat[i, random_label[i]] = 1
        random_cheat = torch.cat((torch.rand((g_batch_size, random_nums - 10)), random_cheat), 1)
        if torch.cuda.is_available():
            random_cheat = random_cheat.cuda()
            random_label = random_label.cuda()
        # G前馈
        cheat_img = G(random_cheat)
        cheat_out, cheat_label = D(cheat_img)
        loss_g = - cheat_out.mean() + torch.log(torch.tensor(epoch + 1, dtype=torch.float32).cuda()) * nn.CrossEntropyLoss()(cheat_label, random_label)
        # G反馈
        optimizer2.zero_grad()
        loss_g.backward()
        optimizer2.step()
    for p in G.parameters():
        p.requires_grad_(False)
    # 观测
    if (epoch+1) % print_num == 0:
        test_label = torch.arange(0, 10)
        random_test = torch.zeros((100, 10))
        for i in range(100):
            random_test[i, i % 10] = 1
        random_test = torch.cat((torch.rand((100, random_nums - 10)), random_test), 1)
        if torch.cuda.is_available():
            random_test = random_test.cuda()
        test_img = G(random_test)
        test_images = to_img(test_img.cpu())
        save_image(test_images, save_path.format(epoch+1), nrow=10)
        real_value, fake_value, cheat_value = real_out.detach().mean().item(), fake_out.detach().mean().item(), cheat_out.detach().mean().item()
        print('*****epoch {}:\nreal:{}\nfake:{}\ncheat:{}\ntimecost = {}\n'.format(epoch+1, real_value, fake_value, cheat_value, time.time()-time0))
        real_list.append(real_value)
        fake_list.append(fake_value)
        cheat_list.append(cheat_value)
    
    epoch += 1
    if epoch % num_epochs == 0:
        winsound.Beep(600,1200)
        to_continue = input('Input \'exit\' to stop training!\n\n')

time_version = time.strftime('%Y%m%d%H%M', time.localtime())
os.mkdir('log/{}'.format(time_version))
plt.figure(1)
xx = range(len(real_list))
plt.scatter(xx, real_list, color='coral', label='real', s=2)
plt.scatter(xx, fake_list, color='violet', label='fake', s=2)
plt.scatter(xx, cheat_list, color='lime', label='cheat', s=2)
plt.title('DG9_{}'.format(time_version, epoch))
plt.legend()
plt.savefig('log/{}/DG9_{}.jpg'.format(time_version, epoch))
plt.show()
plt.close(1)
plt.figure(1)
plt.plot(xx, real_list, color='coral', label='real', linewidth=2)
plt.title('DG9_{}_real'.format(time_version, epoch))
plt.legend()
plt.savefig('log/{}/DG9_{}_real.jpg'.format(time_version, epoch))
plt.show()
plt.close(1)
plt.figure(1)
plt.plot(xx, fake_list, color='violet', label='fake', linewidth=2)
plt.title('DG9_{}_fake'.format(time_version, epoch))
plt.legend()
plt.savefig('log/{}/DG9_{}_fake.jpg'.format(time_version, epoch))
plt.show()
plt.close(1)
plt.figure(1)
plt.plot(xx, cheat_list, color='lime', label='cheat', linewidth=1)
plt.title('DG9_{}_cheat'.format(time_version, epoch))
plt.legend()
plt.savefig('log/{}/DG9_{}_cheat.jpg'.format(time_version, epoch))
plt.show()
plt.close(1)
log = open('log/{}/DG9_log.csv'.format(time_version),'w')
log.write('{},{},{},{}\n'.format('epoch', 'real', 'fake', 'cheat'))
for i in xx:
    log.write('{},{},{},{}\n'.format(i+1, real_list[i], fake_list[i], cheat_list[i]))
log.close()
size = (342, 342)
fps = 60
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
video = cv2.VideoWriter('log/{}/A_Falure_on_DCGAN9_{}_fps{}.MP4'.format(time_version, epoch, fps), fourcc, fps, size)
cover = cv2.imread('log/{}/DG9_{}.jpg'.format(time_version, epoch))
for i in range(2 * fps):
    video.write(cover)
for i in range(1,epoch):
    img = cv2.imread('nameimg/epoch_{}.jpg'.format(i))
    video.write(img)
    if i%100 == 0:
        print('.', end='')
video.release()