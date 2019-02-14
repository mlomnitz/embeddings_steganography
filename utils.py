import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def imshow(img):
    view_img = img.permute(1, 2, 0)
    view_img = np.array(view_img)
    plt.imshow(view_img)


def make_fig(original, covered, full_text, scale = 10):
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])
    #
    ax = plt.subplot(gs[0])
    imshow(original)
    plt.axis('off')
    #
    ax = plt.subplot(gs[1])
    imshow(covered)
    plt.axis('off')
    #
    ax = plt.subplot(gs[2])
    imshow(scale*(original-covered))
    plt.axis('off')
    #
    ax=plt.subplot(gs[1, :])
    plt.text(0.1, 0.1, full_text, size=10)
    plt.axis('off')    
    #
    plt.close()
    return fig


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
