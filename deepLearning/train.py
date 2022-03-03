import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from custom_data import bpdata_test, bpdata_train, bpdata_val
from model import ConvNet




bp_train_dataset = bpdata_train(csv_file='./data/cleaned/bp_train_new.csv',
                                    root_dir='./data/cleaned/cleaned/train')

bp_test_dataset = bpdata_test(csv_file='./data/cleaned/bp_test_new.csv',
                                    root_dir='./data/cleaned/cleaned/test')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 20
num_classes = 1
batch_size = 1
learning_rate = 0.001


train_loader = torch.utils.data.DataLoader(dataset=bp_train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=bp_test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


model = ConvNet(num_classes).to(device)
min_loss = 1000
# Loss and optimizer
criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
#print(model)
for epoch in range(num_epochs):
    for i,(data, label, _) in enumerate(train_loader):
        label = label.float()
        #print(label)
        data = torch.tensor(data).float()
        data = data.unsqueeze(0)
        #print(data.shape)
        outputs = model(data)
        outputs = outputs[0]
        #print(outputs)
        loss = criterion(outputs, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            torch.save(model.state_dict(), 'modelb%d.ckpt'%epoch)
            if loss<min_loss:
                min_loss = loss
                torch.save(model.state_dict(), 'modelb.ckpt')
            output_list = list()
            label_list = list()    
            for i,(data,label,_) in enumerate(test_loader):
                label = label.float()
                #print(data,label)

                data = torch.tensor(data).float()
                data = data.unsqueeze(0)
                #print(data)
                #print(data.shape)
                outputs = model(data)
                outputs = outputs[0]
                #printoutputs,label)

                outputs = outputs.detach().numpy()
                label = label.numpy()

                output_list.append(outputs)
                label_list.append(label)



            print('Testing MAE in epoch {}: {} '.format(epoch,metrics.mean_absolute_error(label_list, output_list)))