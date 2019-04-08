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


def gaussian2d(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def shave(img, border):
    '''
    like ANR, remove border to calculate MSE
    '''
    assert len(np.array(img).shape) <= 3, 'wrong input images shapes to shave'
    if len(np.array(img).shape) == 3:
        return img[border:-border, border:-border, :]
    return img[border:-border, border:-border]



def hashkey(block, weight, Qangle, Qangle, Qstrength, Qcoherence):

    gy,gx = np.gradient(block)
    gx = gx.ravel()
    gy = gy.ravel()

    # SVD calc
    G = np.vstack((gx, gy)).T
    GTWG = G.T.dot(weight).dot(G)
    [eigenvalues,eigenvectors] = np.linalg.eig(GTWG)
    
    #For angle
    angle = np.math.atan2(eigenvectors[0,1],eigenvectors[0,0])
    if angle<0:
        angle += np.pi
    
    #For strength
    strength = eigenvalues.max()/(eigenvalues.sum()+0.0001)
    
    #For coherence
    lamda1 = np.math.sqrt(eigenvalues.max())
    lamda2 = np.math.sqrt(eigenvalues.min())
    coherence = np.abs((lamda1-lamda2)/(lamda1+lamda2+0.0001))
    
    #Quantization
    angle = np.floor(angle/(np.pi/Qangle)-1)
    strength = np.floor(strength/(1.0/Qstrenth)-1)
    coherence = np.floor(coherence/(1.0/Qcoherence)-1)
    
    return int(angle),int(strength),int(coherence)

