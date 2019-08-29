"""
WARNING - This document and related “CodeNitz” project material may contain technical information, export of which may be restricted by the Export Administration Regulations (EAR).  This project is subject to a license exception, which permits transfer of technical information to certain foreign entities, including IQT’s UK and Australian affiliates, and New Zealand Contractor Josh Bailey and his company OSDL, exclusively for “internal development or production of new products”.  Beyond this, IQT personnel must consult with IQT Legal prior to disclosing any project-related information publicly or to foreign persons.  IQT Legal will continue to review the classification of this technology as it is developed, and will update this marking accordingly.
"""


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
