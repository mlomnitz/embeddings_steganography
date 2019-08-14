import PIL
import torchvision.transforms
from torch.autograd.variable import Variable
import numpy as np
import matplotlib.pyplot as plt
#
basic_resize = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(620),
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor()
            ])


def image_loader(image_name):
    """ Basic image loader function
    Parameters
    ----------
    image_name : str
        Path and name to the image file

    Returns
    -------    
    image : torch.tensor
        Loaded image tensor on the gpu
    """   
    image = PIL.Image.open(image_name)
    image = image.convert('RGB')
    image = basic_resize(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image.cuda()

def imshow(img):
    """ Function to display tensor as image - assumes image has been detached
    and on cpu
    Parameters
    ----------
    img : torch.tensor
        Torch tensor of image

    Returns
    -------    
    """   
    view_img = img.permute(1, 2, 0)
    view_img = np.array(view_img)
    plt.imshow(view_img)
