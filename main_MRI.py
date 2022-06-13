# %% imports
# libraries
from doctest import OutputChecker
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
# local imports
import MNIST_dataloader
import Fast_MRI_dataloader
# Model
import Unfolding
# %% set torches random seed
torch.random.manual_seed(0)
# %% Using host or device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Device: ",device)

# %%
# %% preperations
# define parameters
data_loc = '5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64
# hyperparameter for learning rate
mu = 1
# hyperparamter for lambda
shrinkage = 2e-5
# iteration number
K = 3
# epoch
no_epochs = 10
# get dataloader
data_loc = 'Fast_MRI_Knee' #change the datalocation to something that works for you
batch_size = 2

train_loader, test_loader = Fast_MRI_dataloader.create_dataloaders(data_loc, batch_size)
# %%
def kspace(img):
    """
    create kspace by using FT
    """
    # shift (0,0) (2pi,0) (0,2pi) (2pi,2pi) to (1pi,1pi)
    #output =  torch.fft.fftshift(torch.fft.fft2(img))
    output = img.clone()
    output = torch.fft.fft2(output)
    print(len(output))
    for index in range(len(output)):
        output[index] = torch.fft.fftshift(output[index])
    return output

def partial_kspace(img, mask):
    """
    element-wise multiplication
    """
    output = torch.mul(img,mask)
    return output

def recreate_MRI(img):
    #output = torch.fft.ifft2(torch.fft.ifftshift(img))
    output = img.clone()
    for index in range(len(output)):
        output[index] = torch.fft.ifft2(torch.fft.ifftshift(output[index]))
    return output
# %%
kp, m, gt_data= next(iter(train_loader))
gt_kspace = kspace(gt_data)
gt_partial_kspace = partial_kspace(gt_kspace,m)
recreate_data = recreate_MRI(gt_partial_kspace)
# %%
plt.figure(figsize = (25,10))
for i in range(len(gt_data)):
    plt.subplot(len(gt_data),5 ,i*5+1)
    plt.imshow(gt_data[i,:,:], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Ground truth')

    plt.subplot(len(gt_kspace),5 ,i*5+2)
    plt.imshow(gt_kspace[i,:,:].abs().log(), vmin=-2.3, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('2D Fourier Transform')

    plt.subplot(len(m),5 ,i*5+3)
    plt.imshow(m[i,:,:], interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Mask')

    plt.subplot(len(gt_partial_kspace),5, i*5+4)
    plt.imshow((gt_partial_kspace[i,:,:].abs()+0.0001).log(), vmin=-2.3, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Partial kspace')

    plt.subplot(len(recreate_data),5,i*5+5)
    plt.imshow(recreate_data[i,:,:].abs(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if(i==0):
        plt.title('Recreate MRI image')
plt.show()
# %%

# %%
