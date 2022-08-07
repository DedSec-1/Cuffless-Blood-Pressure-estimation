import torch
from custom_data import bpdata_test
from sklearn import metrics   
from model import ConvNet

bp_test_dataset = bpdata_test(csv_file='./data/cleaned/bp_test_new.csv',
                                    root_dir='./data/cleaned/cleaned/test')

test_loader = torch.utils.data.DataLoader(dataset=bp_test_dataset,
                                          batch_size=1, 
                                          shuffle=False)

output_list = []
label_list = []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ConvNet(1).to(device)
model.load_state_dict(torch.load('deep_learning/sbp/modelb.ckpt'))

for i,(data,label,_) in enumerate(test_loader):
    label = label.float()
    # print(label)
    data = torch.tensor(data).float()
    data = data.unsqueeze(0)
    outputs = model(data)
    outputs = outputs[0]

    outputs = outputs.detach().numpy()
    label = label.numpy()

    output_list.append(outputs)
    label_list.append(label)



print('Testing MAE in epoch {}: {} '.format(1,metrics.mean_absolute_error(label_list, output_list)))
error = 0
sum = 0
for output, label in zip(output_list, label_list):
    error += abs(output-label)
    sum += label
    # print(output, label)

print(error/sum * 100)
    