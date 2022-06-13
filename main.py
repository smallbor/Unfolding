# %% imports
# libraries
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
# local imports
import MNIST_dataloader
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
# %% preperations
# define parameters
data_loc = '5LSL0-Datasets' #change the data location to something that works for you
batch_size = 64
# hyperparameter for learning rate
mu = 1
# hyperparamter for lambda
shrinkage = 1
# iteration number
K = 3
# epoch
no_epochs = 10
# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)
# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels
x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels
# use these 10 examples as representations for all digits
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10]
# %% ISTA
def softthreshold(x,shrinkage):
    return torch.sgn(x)*torch.relu(torch.abs(x)-shrinkage)

def ISTA(mu,shrinkage,K,y):
    A = torch.eye(y.size(1))
    x = torch.zeros(y.size(0),y.size(1))
    cost =[]

    for i in range(K):
        y_=y@(mu*A)
        x_=x@(torch.eye(A.shape[1])-mu*A.T@A)
        x=softthreshold(y_+x_,shrinkage)
    return x
# %%
# Visualize ISTA
y=x_noisy_example.flatten(1)
x_denoised=ISTA(mu,shrinkage,K,y+1)
x_denoised = x_denoised.reshape(-1,32,32)
plt.figure(figsize=(15,5))
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(x_noisy_example[i][0],cmap='gray')
    if i == 4:
        plt.title('Noisy', fontsize=17)
    plt.axis(False)
    
    plt.subplot(3,10,i+11)
    plt.imshow(x_denoised[i].numpy()-1,cmap='gray')
    if i == 4:
        plt.title('Denoised', fontsize=17)
    plt.axis(False)

    plt.subplot(3,10,i+21)
    plt.imshow(x_clean_example[i][0],cmap='gray')
    if i == 4:
        plt.title('Clean', fontsize=17)
    plt.axis(False)
    

# %%
# LISTA 
lista = Unfolding.LISTA(K).to(device=device, dtype=torch.float32)
optimizer = torch.optim.Adam(lista.parameters(), lr=mu)
mse = torch.nn.MSELoss()
data_loss = []
# training 
for epoch in range(no_epochs):
    total_loss = 0
    for batch_idx,(x_clean,x_noisy,label) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        # ===forward===
        x_noisy = x_noisy.to(device=device, dtype=torch.float32)
        output = lista(x_noisy)
        loss = mse(output,x_clean)
        # ===backward===
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    print("epoch [{}/{}],MSE loss:{:.4f}".format(epoch+1,no_epochs,total_loss/len(train_loader)))
    data_loss.append(total_loss/len(train_loader))

# %%
# plot LISTA training loss
x = range(1, no_epochs+1)
plt.plot(x, data_loss, 'b-')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss of LISTA")
plt.tight_layout()
plt.savefig("LISTA_training_loss.png",dpi=300,bbox_inches='tight')
plt.show()
lista = lista.to(torch.device('cpu'))
torch.save(lista, './LISTA.pth')
# %%
# showing LISTA denoise result
lista = torch.load("LISTA.pth")
output_data = lista(x_noisy_example)
plt.figure(figsize=(15,5))
for i in range(10):
    plt.subplot(3,10,i+1)
    if(i==4):
            plt.title(label="Noisy MNIST data",fontsize=25)
    plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(3,10,i+11)
    if(i==4):
            plt.title(label="Output of LISTA",fontsize=25)
    plt.imshow(output_data[i].detach().numpy().reshape(32,32),cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,10,i+21)
    if(i==4):
            plt.title(label="Clean MNIST data",fontsize=25)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("LISTA.png",dpi=300,bbox_inches='tight')
plt.show()
# %%
