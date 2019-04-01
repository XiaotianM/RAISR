import os, re, math
import numpy as np
from PIL import Image
from scipy import misc
from scipy import interpolate


def loadImageList(path=None, regx="\.(png|bmp)", keep_prefix=False):
    if path is None:
        path = os.getcwd()

    fileList = os.listdir(path)
    resultList = []
    for f in fileList:
        if re.search(regx, f):
            resultList.append(f)
    
    if keep_prefix:
        for i,f in enumerate(resultList):
            resultList[i] = os.path.joint(path,resultList[i])

    return resultList



def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)



def setImageAlign(image, alignment=2):
    sz = im.shape
    h = sz[0] // alignment * alignment
    w = sz[1] // alignment * alignment
    ims = im[0:h, 0:w, ...]
    return ims



def resize(img, scale, method=Image.BICUBIC):
    h, w = img.shape[0], img.shape[1]
    h_, w_ = int(h*scale), int(w*scale)
    image = Image.fromarray(img)
    image = image.resize([w_, h_], resample=method)
    image = np.asarray(image)
    return image


def bilinear(img, scale):
    '''
        if img.shape = [2,2], output.shape = [2*scale-1, 2*scale-1]
        e.g. scale = x2, the kernel for bilinear is 4
        so when h,w = img.shape, the bilinear upsampled image is 2*h-1, 2*w-1
    '''
    img = np.float32(img)
    h, w = img.shape
    hgrid = np.linspace(0, h-1, h)
    wgrid = np.linspace(0, w-1, w)
    bilinearInterp = interpolate.interp2d(w, h, img, kind="linear")
    hgrid = np.linspace(0. h-1, h*scale-1)
    wgrid = np.linspace(0. w-1, w*scale-1)
    upsampled = bilinearInterp(wgrid, hgrid)
    return upsampled




def shave(img, border):
    '''
    like ANR, remove border to calculate MSE
    '''
    assert len(np.array(img).shape) <= 3, 'wrong input images shapes to shave'
    if len(np.array(img).shape) == 3:
        return img[border:-border, border:-border, :]
    return img[border:-border, border:-border]