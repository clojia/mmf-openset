""" Utility classes for writing samples to file. 
"""

import os.path

import numpy as np
import scipy.misc

class SampleWriter(object):
    """ Sample writer base class 
    """
    def write(self, samples, file_name_sufix):
        """ Write samples to file.
        """
        pass

class ImageGridWriter(SampleWriter):
    """ Write images to file in a grid. 
    """
    def __init__(self, sample_directory, grid_size=[6, 6], img_dims=[32, 32]):
        super(ImageGridWriter, self).__init__()
        self.sample_directory = sample_directory
        self.grid_size = grid_size
        self.img_dims = img_dims

    def write(self, images, file_name_sufix):
        images =  np.reshape(images, [images.shape[0], self.img_dims[0], self.img_dims[1]])
        merged_imgs = ImageGridWriter.merge_img(images, grid_size=self.grid_size)

        if not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)
        scipy.misc.imsave(self.sample_directory+'/fig'+file_name_sufix+'.png', merged_imgs)

    @staticmethod
    def merge_img(images, grid_size=[6, 6]):
        """
        """
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * grid_size[0], w * grid_size[1]))

        for idx, image in enumerate(images):
            i = idx % grid_size[1]
            j = idx // grid_size[1]
            img[j*h:j*h+h, i*w:i*w+w] = image

        return img
