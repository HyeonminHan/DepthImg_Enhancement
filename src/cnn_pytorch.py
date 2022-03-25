import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim     
from torch.utils.data import DataLoader, Dataset 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.len





class CNN_1layer(nn.Module):
    def __init__(self):
        super(CNN_1layer, self).__init__()
        self.main = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(960, 540), stride=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(960, 540), stride=1, bias=0.1),
            nn.ReLU(True)
    )

    def forward(self, input):
        output = self.main(input)
        return output

  

class CNN_4layer(nn.Module):

    def __init__(self):
        super(CNN_1layer, self).__init__()
        self.main = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(True)
    )

    def forward(self, input):
        output = self.main(input)
        return output
  

def cal_loss(input, gt):

    loss = torch.zeros([gt.shape], dtype=torch.float32)

    # gt 이미지에서 0인 픽셀값을 input영상 또한 0으로 처리
    input[gt==0] = 0 ### input 변수 실제 메모리 주소로 접근되는지 확인하기 (접근되면 안됨)

    criterion = nn.MSELoss()
    loss = criterion(input, gt)

    return loss
        

cnn_1layer = CNN_1layer()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

trainsets = TensorData(X_train, Y_train)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)

testsets = TensorData(X_test, Y_test)
testloader = torch.utils.data.DataLoader(testsets, batch_size=32, shuffle=False)

losses = [] 
n = len(trainloader)
epochs = 400

optimizer = optim.Adam(cnn_1layer.parameters(), lr=0.001, weight_decay=1e-7)

for epoch in range(epochs):

    running_loss = 0.0 
    for i, data in enumerate(trainloader, 0):
        inputs, values = data 

        optimizer.zero_grad() # 최적화 초기화.

        outputs = cnn_1layer(inputs) 
        loss = cal_loss(outputs, values) 
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item() 
  
    losses.append(running_loss/n) # MSE(Mean Squared Error) 계산

plt.plot(losses)
plt.title('Loss')
plt.xlabel('epoch')
plt.show()

##########################################################
