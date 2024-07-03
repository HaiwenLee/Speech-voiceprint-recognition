from python_speech_features import logfbank
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(0)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model import LSTMModel

train_path = 'D:\\DL-codes\\practice\\natural language processing\\Speech voiceprint recognition\\dataset\\train'

def split_array(arr, T, cata):
    sparr = arr.copy()
    i = 1
    split_arr = []
    cata_list = []
    while(i * T < arr.shape[0]):
        split_arr.append(torch.from_numpy(sparr[(i-1) * 200: i * 200, :]))
        cata_list.append(cata)
        i += 1
    #print(len(split_arr))
    return split_arr, cata_list

def onehotencoder(data_list, catagory_list):
    data_list2 = catagory_list#list(set(data_list))
    catagories = len(data_list2)
    print("datalist2",data_list2)
    def getlist(n, i):
        result = [0.0] * n
        result[i] = 1.0
        return result
    encoded_list = []
    for elements in data_list:
        indeces = data_list2.index(elements)
        encoded_list.append(getlist(catagories, indeces))
    return encoded_list

train_dataset = []
train_label = []
ins = 0
catagory_list = ['Trump', 'Gumu', 'Li Mu', 'Luo Xiang', 'Tom']
for subfile in catagory_list:#os.listdir(train_path):
    subfile_path = train_path + "\\" + str(subfile)
    for filename in os.listdir(subfile_path):
        (rate, sig) = wav.read(os.path.join(subfile_path, filename))
        
        mfcc_feat = mfcc(sig, rate, nfft=2048)
        ins += 1
        print("reading......now ", ins)
        kk, ll = split_array(mfcc_feat, 200, cata = subfile)
        train_dataset.extend(kk)
        train_label.extend(ll)

train_label = onehotencoder(train_label, catagory_list)
      
train_dataset_tensor = torch.stack(train_dataset, dim = 0)
train_dataset_tensor = train_dataset_tensor.to(torch.float32)
train_label_tensor = torch.tensor(train_label).to(torch.float32)
print("dataset: ", train_dataset_tensor.shape)
print("label: ", train_label_tensor.shape)


def addbatch(data_train,data_test,batchsize):
    data = TensorDataset(data_train,data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=True)
    return data_loader
traindata = addbatch(train_dataset_tensor,train_label_tensor,200)

input_dim = 13 
hidden_dim = 16 
num_layers = 2 
num_classes = 5 
EPOCH = 8000

model = LSTMModel(input_dim, hidden_dim, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.00001)

loss_contain = []

for epoch in range(EPOCH):
    for step, data in enumerate(traindata):
        inputs, labels = data
        output = model(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%20 == 0:
            print(f"epoch: {epoch}: loss: {loss.item()}")
            loss_contain.append(loss.item())

PATH = 'lstm.pth'
torch.save(model, PATH)



