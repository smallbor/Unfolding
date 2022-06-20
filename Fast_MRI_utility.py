import torch

def kspace(img):
    """
    Create kspace by using FT
    """
    # shift (0,0) (2pi,0) (0,2pi) (2pi,2pi) to (1pi,1pi)
    output = img.clone()
    output = torch.fft.fft2(output)
    for index in range(len(output)):
        output[index] = torch.fft.fftshift(output[index])
    return output

def partial_kspace(img, mask):
    """
    Element-wise multiplication with kspace and mask
    """
    output = torch.mul(img,mask)
    return output

def recreate_MRI(img):
    """
    Reconstruct MRI image by using inverse FT
    """
    #output = torch.fft.ifft2(torch.fft.ifftshift(img))
    output = img.clone()
    for index in range(len(output)):
        output[index] = torch.fft.ifft2(torch.fft.ifftshift(output[index]))
    return output