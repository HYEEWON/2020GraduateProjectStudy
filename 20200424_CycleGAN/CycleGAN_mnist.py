import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.autograd import Variable
#from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
from IPython import display

transform = transforms.Compose([
  transforms.ToTensor(), # 데이터를 파이토치의 Tensor 형식으로바꾼다.
  transforms.Normalize(mean=(0.5,), std=(0.5,)) # 픽셀값 0 ~ 1 -> -1 ~ 1
])

# MNIST 데이터셋을 불러온다. 지정한 폴더에 없을 경우 자동으로 다운로드한다.
mnist =datasets.MNIST(root='data', download=True, transform=transform)

# 데이터를 한번에 batch_size만큼만 가져오는 dataloader를 만든다.
dataloader =DataLoader(mnist, batch_size=60, shuffle=True)


# 생성자는 랜덤 벡터 z를 입력으로 받아 가짜 이미지를 출력한다.
class Generator(nn.Module):

    # 네트워크 구조
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=1024, out_features=28 * 28),
            nn.Tanh())

    # (batch_size x 100) 크기의 랜덤 벡터를 받아
    # 이미지를 (batch_size x 1 x 28 x 28) 크기로 출력한다.
    def forward(self, inputs):
        return self.main(inputs).view(-1, 1, 28, 28)


# 구분자는 이미지를 입력으로 받아 이미지가 진짜인지 가짜인지 출력한다.
class Discriminator(nn.Module):

    # 네트워크 구조
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid())

        # (batch_size x 1 x 28 x 28) 크기의 이미지를 받아
        # 이미지가 진짜일 확률을 0~1 사이로 출력한다.

    def forward(self, inputs):
        inputs = inputs.view(-1, 28 * 28)
        return self.main(inputs)

G = Generator()
D = Discriminator()
#model = models.vgg13()
#summary(model, (1, 28, 28))
# 구분자의 출력값: 이미지가 진짜일 확률 -> 이 확률이 얼마나 정답과 가까운지 측정
# Binary Cross Entropy loss
criterion = nn.BCELoss()
# 손실 함수의 값은 구분자가 출력한 확률이 정답에 가까우면 낮아지고 정답에 멀면 높아짐
# 이 함수의 값을 낮추는 것이 모델의 학습 목표

# 생성자의 매개 변수를 최적화하는 Adam optimizer
G_optimizer = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
# 구분자의 매개 변수를 최적화하는 Adam optimizer
D_optimizer = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_test_samples = 16
testnoise = Variable(torch.randn(num_test_samples, 100))
num_fig = 0
size_figure_grid = int(math.sqrt(num_test_samples))
fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    ax[i,j].get_xaxis().set_visible(False)
    ax[i,j].get_yaxis().set_visible(False)

tracking_dict = {}
tracking_dict["d_loss"] = []
tracking_dict["g_loss"] = []
tracking_dict["real_score"] = []
tracking_dict["fake_score"] = []

num_epochs = 10
num_train = 1000
for epoch in range(num_epochs):
    torch.autograd.set_detect_anomaly(True)
    # 한번에 batch_size(60)만큼 데이터를 가져온다.
    # 60000만개 -> 1 epoch = 1000 학습
    idx = 1
    for real_data, _ in dataloader:
        batch_size = real_data.size(0)

        # 데이터를 파이토치의 변수로 변환한다.
        real_data = Variable(real_data)

        #구분자 학습
        # 구분자의 손실 함수: 진짜 이미지를 입력했을 때의 출력값과 1과의 차이,
        #                   가짜 이미지를 입력했을 때의 출력값과 0과의 차이의 합

        # 이미지가 진짜일 때 정답 값은 1이고 가짜일 때는 0이다.
        # 정답지에 해당하는 변수를 만든다.
        target_real = Variable(torch.ones(batch_size, 1))
        target_fake = Variable(torch.zeros(batch_size, 1))

        # 진짜 이미지를 구분자에 넣는다.
        D_result_from_real = D(real_data)
        # 구분자의 출력값이 정답지인 1에서 멀수록 loss가 높아진다.
        D_loss_real = criterion(D_result_from_real, target_real)
        # 생성자에 입력으로 줄 랜덤 벡터 z를 만든다.
        z = Variable(torch.randn((batch_size, 100)))
        # 생성자로 가짜 이미지를 생성한다.
        fake_data = G(z)
        # 생성자가 만든 가짜 이미지를 구분자에 넣는다.
        D_result_from_fake = D(fake_data)
        # 구분자의 출력값이 정답지인 0에서 멀수록 loss가 높아진다.
        D_loss_fake = criterion(D_result_from_fake, target_fake)
        # 구분자의 loss는 두 문제에서 계산된 loss의 합이다.
        D_loss = D_loss_real + D_loss_fake
        # 구분자의 매개 변수의 미분값을 0으로 초기화한다.
        D.zero_grad()
        # 역전파를 통해 매개 변수의 loss에 대한 미분값을 계산한다.
        D_loss.backward()
        # 최적화 기법을 이용해 구분자의 매개 변수를 업데이트한다.
        D_optimizer.step()


        # 생성자 학습
        # 생성한 이미지를 구분자에 넣었을 때, 출력 값이 1에 가깝게 나오도록 함
        # 생성자의 손실 함수: 출력 값이 1에서 떨어진 정도 -> 최소가 되도록 생성자 학습습
        # 생성자에 력으로 줄 랜덤 벡터 z를 만든다.
        z = Variable(torch.randn((batch_size, 100)))
        #z = z.cuda()

        # 생성자로 가짜 이미지를 생성한다.
        fake_data = G(z)
        # 생성자가 만든 가짜 이미지를 구분자에 넣는다.
        D_result_from_fake = D(fake_data)
        # 생성자의 입장에서 구분자의 출력값이 1에서 멀수록 loss가 높아진다.
        G_loss = criterion(D_result_from_fake, target_real)
        # 생성자의 매개 변수의 미분값을 0으로 초기화한다.
        G.zero_grad()
        # 역전파를 통해 매개 변수의 loss에 대한 미분값을 계산한다.
        G_loss.backward()
        # 최적화 기법을 이용해 생성자의 매개 변수를 업데이트한다.
        a = G_optimizer.step()

        print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
              'D(x): %.2f, D(G(z)): %.2f'
              % (epoch + 1, num_epochs, idx + 1, num_train, D_loss.data, G_loss.data,
                 D_loss_real.data.mean(), D_loss_fake.data.mean()))
        idx += 1
        if epoch % 2 == 0 and idx == 1000:
            test_images = G(testnoise)

            # 이미지를 쥬피터 노트북에 띄웁니다.

            for k in range(num_test_samples):
                i = k // 4
                j = k % 4
                ax[i, j].cla()
                ax[i, j].imshow(test_images[k, :].data.cpu().numpy().reshape(28, 28), cmap='Greys')
            display.clear_output(wait=True)
            display.display(plt.gcf())

            plt.savefig('results/mnist-gan-%03d.png' % num_fig)
            num_fig += 1
            tracking_dict["d_loss"].append(D_loss.data)
            tracking_dict["g_loss"].append(G_loss.data)
            tracking_dict["real_score"].append(D_loss_real.data.mean())
            tracking_dict["fake_score"].append(D_loss_fake.data.mean())