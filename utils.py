import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage.measure import compare_psnr
from PIL import Image, ImageSequence
from apng import APNG

def imshow(img):
    view_img = img.permute(1, 2, 0)
    view_img = np.array(view_img)
    plt.imshow(view_img)


def save_frame(frame, output):
    frame = frame.permute(1, 2, 0).detach().cpu().numpy()
    img = Image.fromarray(np.uint8(255*(frame)), 'RGB')
    img.save(output)
    
    
def make_fig(original, covered, hidden, recovered, pix_error, psnr, full_text,
             scale=10):
    fig = plt.figure(figsize=(25, 15))
    gs = gridspec.GridSpec(3, 5)
    #
    ax = plt.subplot(gs[0, 0])
    imshow(original)
    plt.title('Cover')
    plt.axis('off')
    #
    ax = plt.subplot(gs[0, 1])
    imshow(covered)
    plt.title('Container')
    plt.axis('off')
    #
    ax = plt.subplot(gs[0, 2])
    imshow(scale*(original-covered))
    plt.title('{} x residual'.format(scale))
    plt.axis('off')
    #
    ax = plt.subplot(gs[1, 0])
    imshow(hidden)
    plt.title('Secret')
    plt.axis('off')
    #
    ax = plt.subplot(gs[1, 1])
    imshow(recovered)
    plt.title('Revealed')
    plt.axis('off')
    #
    ax = plt.subplot(gs[1, 2])
    imshow(scale*(hidden-recovered))
    plt.title('{} x residual'.format(scale))
    plt.axis('off')
    #
    ax = plt.subplot(gs[2, 0])
    hist, bins_ = np.histogram(pix_error)
    freq = hist/np.sum(hist)
    plt.bar(bins_[:-1], freq, align="edge", width=np.diff(bins_))
    plt.yscale('log')
    plt.title('Pixel error')
    #
    ax = plt.subplot(gs[2, 1])
    hist, bins_ = np.histogram(psnr)
    freq = hist/np.sum(hist)
    plt.bar(bins_[:-1], freq, align="edge", width=np.diff(bins_))
    plt.title('PSRN')
    #
    ax=plt.subplot(gs[ :, 3:4])
    plt.text(0., 0., full_text, size=10)
    plt.axis('off')    
    #
    plt.close()
    return fig


def image_metrics(image1, image2):
    np1 = image1.numpy()
    np2 = image2.numpy()
    diff = np.abs(np1-np2)/255
    #
    return diff, np.array([compare_psnr(np1, np2)])


def cos_matrix_multiplication(matrix, vector):
    """
    Calculating pairwise cosine distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors


def gif_to_frames(gif='', save_path='GIFsource/frames'):
    im = APNG.open(gif)
    for i, (png, control) in enumerate(im.frames):
        png.save('./{}/frame{}.png'.format(save_path, i))
    

def frames_to_gif(source_dir='./Example', save_path='./message.gif'):
    n_frames = len(os.listdir(source_dir))
    im = APNG()
    for idx in range(n_frames):
        im.append_file('./{}/frame{}.png'.format(source_dir, idx), delay=50)
    im.save(save_path)
