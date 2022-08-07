import torch 
import torch.nn as nn
import numpy as np

class ConvNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32,kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.lstm = nn.LSTM(32256,100,1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(32976, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        
        out1 = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = out1.unsqueeze(0)
        #print(out1.shape)
        out, hid = self.lstm(out)

        out = out.squeeze(0)
        out2 = out.detach().numpy().reshape(10,10)
        out2 = torch.from_numpy(out2)
        out2 = out2.unsqueeze(0)
        out2 = out2.unsqueeze(0)
        out3 = self.layer3(out2)
        out = np.concatenate((out3.detach().numpy(),out1.detach().numpy()),axis=None)
        out = torch.from_numpy(out)
        out = out.unsqueeze(0)
        out = self.fc(out)
        out = self.fc2(out)

        return out
