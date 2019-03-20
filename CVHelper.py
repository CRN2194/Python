
import matplotlib.cm as cm  #
import matplotlib.pyplot as plt
from skimage import data, io
import numpy as np
# from skimage.color import rgb2grays
import os
import fire


class CVHelper(object):
    """Utilities for image processing and computer vision apps"""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def read(self, folder_path, formate, recursive=False, grayscale=False, folderout=""):

        image_list = []
        image = [".jpg", ".png", ".bmp", ".tiff"]
        nameimage = []
        i = 0
        if (recursive):
            print("Recursive search: ")
        else:
            print("Non-recursive search")
        #loop of repetition which allows to cross the files of form of refusal or not, depending on the case
        for root, dirs, files in os.walk(folder_path):
            if recursive:
                for name in files:
                    (namef, extend) = os.path.splitext(name)
                    if (extend in image):
                        if (extend != ".DS_Store"):
                            image_list.append(os.path.join(root, namef + extend))
                            nameimage.append(name.rstrip(extend))
                            
            else:
                while len(dirs) > 0:
                    dirs.pop()
                    for name in files:
                        (namef, extend) = os.path.splitext(name)
                        if (extend in image):
                            if (extend != ".DS_Store"):
                                image_list.append(root + namef + extend)
                                nameimage.append(name.rstrip(extend))

        if self.verbose:
            print(image_list)
        #function that calculates the average of each pixel
        def prom(pixel):
            return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
        #Conditional that allows you to convert the image to gray scales
        if grayscale:
            print("Change to scale Gray")
            for gray in image_list:

                gray_scale = io.imread(gray)
                grey = np.zeros((gray_scale.shape[0], gray_scale.shape[1]))
                for row in range(len(gray_scale)):
                    for col in range(len(gray_scale[row])):
                        grey[row][col] = prom(gray_scale[row][col])
                #converts the type supported by the image to ubytes
                ctype = grey.astype(np.ubyte)
                out = os.path.join(folderout, nameimage[i])
                io.imsave(out + formate, ctype)
                i += 1

#line of command that allows to use the bookstore fire
if __name__ =='__main__':
    fire.Fire(CVHelper)

#path = "/home/cesar/Desktop/imagenes/zero/test_images/test_images/"
#out = "/home/cesar/Desktop/grayscale"
#cvhelper = CVHelper(True)
#image_list = cvhelper.read(path, ".jpg", False, True, out)

#Test
#python3 CVHelper.py read "/home/cesar/Desktop/imagenes/zero/test_images/test_images/" ".jpg" "False" "True"
# "/home/cesar/Desktop/grayscale" --verbose="True"
