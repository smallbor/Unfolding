# %% imports
# libraries
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
# local imports
import MNIST_dataloader
import Fast_MRI_dataloader
from Fast_MRI_utility import *
# Model
from Unfolding import *
# %% set torches random seed
torch.random.manual_seed(0)
# %% Using host or device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Device: ",device)

# %% preperations
# mu
mu =1 
# hyperparamter for lambda
shrinkage = 0.01
# iteration number
K = 3
# epoch
no_epochs = 10
# get dataloader
data_loc = 'Fast_MRI_Knee' #change the datalocation to something that works for you
batch_size = 2

train_loader, test_loader = Fast_MRI_dataloader.create_dataloaders(data_loc, batch_size)
# %%
# take a random batch from dataloader
kp, m, gt_data= next(iter(train_loader))
gt_kspace = kspace(gt_data)
gt_partial_kspace = partial_kspace(gt_kspace,m)
recreate_data = recreate_MRI(gt_partial_kspace)
# %%
# Visualize ground true and kspace
plt.figure(figsize=(10,10))
for i in range(len(gt_data)):
    plt.subplot(len(gt_data),2 ,i*2+1)
    plt.imshow(gt_data[i,:,:], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Ground truth',fontsize=20)

    plt.subplot(len(gt_kspace),2 ,i*2+2)
    plt.imshow(gt_kspace[i,:,:].abs().log(), vmin=-2.3, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('K-space',fontsize=20)
plt.savefig("Exercise_Result/Exercise3_a")
# %%
# Visualize partial kspace and accerlated MRI
plt.figure(figsize=(10,10))
for i in range(batch_size):
    plt.subplot(batch_size,2 ,i*2+1)
    plt.imshow((gt_partial_kspace[i,:,:].abs()+0.0001).log(), vmin=-2.3, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Ground truth',fontsize=20)

    plt.subplot(batch_size,2 ,i*2+2)
    plt.imshow(recreate_data[i,:,:].abs(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Accerlerated MRI',fontsize=20)
plt.savefig("Exercise_Result/Exercise3_c")
# %%
# Visualize the process of accerlerated MRI
plt.figure(figsize = (25,10))
for i in range(len(gt_data)):
    plt.subplot(len(gt_data),5 ,i*5+1)
    plt.imshow(gt_data[i,:,:], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Ground truth',fontsize=20)

    plt.subplot(len(gt_kspace),5 ,i*5+2)
    plt.imshow(gt_kspace[i,:,:].abs().log(), vmin=-2.3, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('2D Fourier Transform',fontsize=20)

    plt.subplot(len(m),5 ,i*5+3)
    plt.imshow(m[i,:,:], interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Mask',fontsize=20)

    plt.subplot(len(gt_partial_kspace),5, i*5+4)
    plt.imshow((gt_partial_kspace[i,:,:].abs()+0.0001).log(), vmin=-2.3, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Partial kspace',fontsize=20)

    plt.subplot(len(recreate_data),5,i*5+5)
    plt.imshow(recreate_data[i,:,:].abs(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Recreate MRI image',fontsize=20)
plt.savefig("Exercise_Result/Exercise3_d")
plt.show()
# %%
# MRI ISTA
shrinkage = 0.1
MRI_ISTA = ISTA_MRI(kp, m, mu, K=1000,shrinkage= shrinkage)

plt.figure(figsize=(10,10))
for i in range(len(gt_data)):
    plt.subplot(batch_size,3 ,i*3+1)
    plt.imshow(gt_data[i,:,:], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Ground truth',fontsize=20)

    plt.subplot(batch_size,3 ,i*3+2)
    plt.imshow(recreate_data[i,:,:].abs(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Acc MRI',fontsize=20)    

    plt.subplot(batch_size,3 ,i*3+3)
    plt.imshow(MRI_ISTA[i,:,:].abs(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('ISTA MRI',fontsize=20)
plt.savefig("Exercise_Result/Exercise4")
# %%
# ConvNet
def Evaluation(model:ConvNet, dataloader) -> float:
    mse = torch.nn.MSELoss()
    total_loss = 0
    print("Evaluating the model with test set...")
    for batch_idx,(ksp,_,gt) in enumerate(tqdm(dataloader)):
        re_MRI = recreate_MRI(ksp).abs().unsqueeze(1)
        result = model(re_MRI)
        loss = mse(result,gt)
        total_loss += loss.data
    print("Test set MSE loss:{:.4f}".format(total_loss/len(dataloader)))
    return total_loss/len(dataloader)

convnet = ConvNet()
optimizer = torch.optim.Adam(convnet.parameters(), lr=0.01)
mse = torch.nn.MSELoss()
# train
trian_loss_list = []
# test
test_loss_list = []
for epoch in range(no_epochs):
    total_train_loss = 0
    test_loss_list.append(Evaluation(convnet,test_loader))
    for batch_idx,(ksp,mask,gt) in enumerate(tqdm(train_loader)):
        # recreate MRI
        re_MRI = recreate_MRI(ksp).abs().unsqueeze(1)
        optimizer.zero_grad()
        # ===forward===
        re_MRI = re_MRI.to(device=device, dtype=torch.float32)
        output = convnet(re_MRI)
        loss = mse(output,gt)
        # ===backward===
        loss.backward()
        optimizer.step()
        total_train_loss += loss.data
    print("epoch [{}/{}],MSE loss:{:.4f}".format(epoch+1,no_epochs,total_train_loss/len(train_loader)))
    trian_loss_list.append(total_train_loss/len(train_loader))
torch.save(convnet,"./ConvNet.pth")
# %%
# Visualize result of ConvNet
x = np.arange(0,no_epochs,1)
y_train = np.array(trian_loss_list)
y_test = np.array(test_loss_list)
plt.plot(x, y_train,label=r'Training set')
plt.plot(x, y_test,'-.r',label=r'Test set')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0,no_epochs,1))
plt.title("Training and test Loss")
plt.grid()
plt.legend()

plt.savefig("Exercise_Result/Exercise5")
plt.show()
# %%
# Evalution Convnet with the previous random batch
convnet = torch.load("./ConvNet.pth")
result = convnet(recreate_data.abs().unsqueeze(1)).squeeze()
plt.figure(figsize = (30,15))
with torch.no_grad():
    for i in range(len(result)):
        plt.subplot(len(result),3 ,i*3+1)
        plt.imshow(gt_data[i,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(len(result),3 ,i*3+2)
        plt.imshow(recreate_data[i,:,:].abs(),cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(len(result),3 ,i*3+3)
        plt.imshow(result[i,:,:].abs(),cmap='gray')
        plt.xticks([])
        plt.yticks([])
plt.savefig("Exercise_Result/Exercise5_evaluation")
# %%
# Neural proximal gradient descent


