'''
Created on Nov 17, 2016 by Hojjat @iPAL
'''

import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
CHANNELS = 1

def concat_images(imga, imgb, pad=1):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])+2*pad
    total_width = wa+wb+2*pad
    #new_img = np.zeros(shape=(max_height, total_width, CHANNELS), dtype=np.uint8)
    new_img = np.zeros(shape=(max_height, total_width, CHANNELS), dtype=np.uint8)
    new_img[pad:ha+pad,pad:wa+pad]=imga
    new_img[pad:hb+pad,wa+2*pad:wa+wb+2*pad]=imgb
    return new_img

def concat_n_images_row(image_cube,pad=10):
    """
    Combines N color images from a list of image paths.
    image sizes should be equal
    image cube structure: [-1,  W, H , DEPTH]
    """
    [_, h , w , num_img] = image_cube.shape
    #num_img = 4
    #img = plt.imread(image_path_list[0])[:,:,:CHANNELS]
    #h,w= img.shape[:2]
    total_width = w*num_img
    output =  np.ones(shape=(h+2*pad, total_width+pad*(num_img+1) ,CHANNELS), dtype=np.float32)
    for i in range(num_img): #img_path in enumerate(image_path_list):
        img = image_cube[0,:,:,i].reshape(h,w,1)
        output[pad:h+pad , (pad+w)*i+pad:(pad+w)*(i+1)]=img
 
    return output

def concat_n_images_col(image_cube,pad=10):
    """
    Combines N color images from a list of image paths.
    image sizes should be equal
    
    """
    [_, h , w , num_img] = image_cube.shape
    #num_img = 4
    #img = plt.imread(image_path_list[0])[:,:,:CHANNELS]
    #h,w= img.shape[:2]
    total_height = h*num_img
    output =  np.ones(shape=(total_height+pad*(num_img+1), w+2*pad ,CHANNELS), dtype=np.float32)
    for i in range(num_img):#for i, img_path in enumerate(image_path_list):
        img = image_cube[0,:,:,i].reshape(h,w,1)#img = plt.imread(img_path)[:,:,:CHANNELS]
        output[(pad+h)*i+pad:(pad+h)*(i+1) , pad:w+pad] =img
 
    return output
    
    
def concat_n_images(image_cube, rows, cols, pad=10):
    [_, h , w , num_img] = image_cube.shape
    if rows*cols != num_img:
        print('This is not supported')
        raise Exception('This is not supported')
        
    total_height = h*rows
    total_width = w*cols

    #output =  np.zeros(shape=(total_height+pad*(rows+1), total_width+pad*(cols+1) ,CHANNELS), dtype=np.float32)
    row_images =  np.zeros(shape=(1, h+2*pad , total_width+pad*(cols+1) ,rows), dtype=np.float32)
    
    for i in range(num_img):
        temp_max= np.amax(image_cube[:,:,:,i])
        temp_min= np.amin(image_cube[:,:,:,i])
        if temp_max != temp_min:
            image_cube[:,:,:,i] = (image_cube[:,:,:,i] -temp_min) / (temp_max - temp_min)
            
    for row in range(rows):
        temp = concat_n_images_row(image_cube[:,:,:, cols*row: cols*(row+1)],pad=pad)
        
        row_images[0,:,:,row] = temp[:,:,0]
        
    output = concat_n_images_col(row_images,pad=pad)   

    
    return output
    




def put_kernels_on_grid_old (kernel, grid_Y, grid_X, pad=1, deconv = False):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad
    if deconv == True:
        kernel = tf.transpose(kernel, (0, 1, 3, 2))
    C = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, C] ))
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, C]))
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    x9 = tf.transpose(x8, (3, 1, 2, 0))
    
    return x9



def put_kernels_on_grid (kernel, grid_Y, grid_X, pad=1, deconv = False):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    
    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,0],[pad,0],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad
    if deconv == True:
        kernel = tf.transpose(kernel, (0, 1, 3, 2))
    C = kernel.get_shape()[2]
    N = kernel.get_shape()[3]
    
    #[H, W, C, N]
    
    x1 = tf.transpose(x, (3,2,0,1))#[N, C, H, W]
    
    x2 = tf.reshape(x1, tf.stack([N*C, Y, X , 1])) # [N*C, H, W, 1]
    
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, 1] ))  # [grid_X, Y * grid_Y, X, 1]
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))   # [grid_X, X, Y * grid_Y,  1] 
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, 1])) # [ 1 , X * grid_X, Y * grid_Y , 1]
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))   # [ Y * grid_Y ,  X * grid_X, 1 , 1]    
    
    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))      # [ 1 ,Y * grid_Y ,  X * grid_X, 1 ] 

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    x9 = tf.transpose(x8, (3, 1, 2, 0))
    
    return x9



    
    
    
    
    
    
            