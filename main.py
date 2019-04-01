import os, argparse
import numpy as np
from PIL import Image
from helper import args, utils


args = Args.getArgs()

upscale = args.rate
patchSize = args.patchSize               # patchSize
gradientNeigbor = args.neigborSize       # control gradient size
margin = floor(max(patchSize, neigborSize) / 2)
patchMargin = floor(patchSize / 2)
gradientMargin = floor(gradientNeigbor / 2)
stride = args.stride
Qangle = args.Qangle
Qstrength = args.Qstrength
Qcoherence = args.Qcoherence
trainPath = args.train_dataset



def train():

    # A belongs to MN x d^2, each upsamped LR patch forming a row in A, 
    # b belongs to MN, each pixle corresponding to the center of LR patch
    # Q is d^2 x d^2 matrix, Q = AT*A
    # V is d^2 x 1 matrix,   V = AT*b
    # h is d^2 x 1 matrix,   min_h{||Qh - V||}
    Q = np.zeros((Qangle, Qstrength, Qcoherence, upscale*upscale, patchSize*patchSize, patchSize*patchSize))
    V = np.zeros((Qangle, Qstrength, Qcoherence, upscale*upscale, patchSize*patchSize, 1))
    h = np.zeros((Qangle, Qstrength, Qcoherence, upscale*upscale, patchSize*patchSize, 1))

    ## calculate Q, V from image dataset
    trainList = utils.loadImageList(trainPath)
    for i, file in enumerate(trainList):

        HR = np.array(Image.open(trainPath + file))
        HR = utils.setImageAlign(HR, alignment=upscale)
        LR = utils.resize(HR, 1/upscale, method=Image.BICUBIC)   # LR is downscaling by Bicubic 
        HR_y = utils.rgb2ycbcr(HR, only_y=True)
        LR_y = utils.rgb2ycbcr(LR, only_y=True)
        Upsampled_y = utils.bilinear(LR_y, upscale)              # cheap upscaling by BILINEAR

        height, width = Upsampled_y.shape
        cnt = 0  
        for row in range(margin, height-margin, stride):
            for col in range(margin, width-margin, stride)
                cnt += 1
                patch = Upsampled_y[row-patchMargin:row+patchMargin+1, col-patchMargin:col+patchMargin+1]
                patch = patch.reshape([1, patchSize*patchSize])  # flatten the vector
                pixelHR = HR_y[row,col]

                gradientblock = Upsampled_y[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
                angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)
                pixeltype = ((row-upscale) % R) * R + ((col-upscale) % R)
                
                ATA = np.dot(patch.T, patch)
                ATb = np.dot(patch.T, pixelHR)
                
                Q[angle,strength,coherence,pixeltype] += ATA
                V[angle,strength,coherence,pixeltype] += ATb

        print(print("process image %d, patches: %d , (%d / %d)..." % (file, cnt, i, len(trainList))))


    ## optimize h for each bucket 
    print('Computing h ...')
    


if __name__ == "__main__":
    train()