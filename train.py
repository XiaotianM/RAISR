import os, argparse, math
import numpy as np
from PIL import Image
from helper import Args, utils, hashkey


args = Args.getArgs()

upscale = args.rate
patchSize = args.patchSize               # patchSize 11         
gradientNeigbor = args.neigborSize       # control gradient size 9
margin = math.floor(max(patchSize, gradientNeigbor) / 2)
patchMargin = math.floor(patchSize / 2)
gradientMargin = math.floor(gradientNeigbor / 2)
stride = args.stride
trainPath = args.train_dataset
Qangle = args.Qangle
Qstrength = args.Qstrength
Qcoherence = args.Qcoherence


def train():

    # A belongs to MN x d^2, each upsamped LR patch forming a row in A, 
    # b belongs to MN, each pixle corresponding to the center of LR patch
    # Q is d^2 x d^2 matrix, Q = AT*A
    # V is d^2 x 1 matrix,   V = AT*b
    # h is d^2 x 1 matrix,   min_h{||Qh - V||}
    Q = np.zeros((Qangle, Qstrength, Qcoherence, upscale*upscale, patchSize*patchSize, patchSize*patchSize))
    V = np.zeros((Qangle, Qstrength, Qcoherence, upscale*upscale, patchSize*patchSize, 1))
    binCount = np.zeros((Qangle, Qstrength, Qcoherence, upscale*upscale))
    h = np.zeros((Qangle, Qstrength, Qcoherence, upscale*upscale, patchSize*patchSize, 1))
    W = np.diag(utils.gaussian2d([gradientNeigbor, gradientNeigbor], 2))

    weighting = utils.gaussian2d([gradientNeigbor, gradientNeigbor], 2)
    weighting = np.diag(weighting.ravel())

    ## calculate Q, V from image dataset
    print("Dataset Q,V collecting...")
    trainList = utils.loadImageList(trainPath)
    for i, file in enumerate(trainList):

        HR = np.array(Image.open(trainPath + file))
        HR = utils.setImageAlign(HR, alignment=upscale)
        LR = utils.resize(HR, 1/upscale, method=Image.BICUBIC)   # LR is downscaling by Bicubic 
        HR = HR / 255.0
        LR = LR / 255.0
        HR_y = utils.rgb2ycbcr(HR, only_y=True)
        LR_y = utils.rgb2ycbcr(LR, only_y=True)
        Upsampled_y = utils.bilinear(LR_y, upscale)              # cheap upscaling by BILINEAR
        Upsampled_y_dh, Upsampled_y_dw = np.gradient(Upsampled_y) 


        height, width = Upsampled_y.shape
        cnt = 0  
        for row in range(margin, height-margin, stride):
            for col in range(margin, width-margin, stride):
                cnt += 1
                patch = Upsampled_y[row-patchMargin:row+patchMargin+1, col-patchMargin:col+patchMargin+1]
                patch = patch.reshape([1, patchSize*patchSize])  # flatten the vector
                pixelHR = HR_y[row,col]
                
                # gradientMargin < blockmargin; can remove border artificats
                # gradientblock = Upsampled_y[row-gradientMargin:row+gradientMargin+1, col-gradientMargin:col+gradientMargin+1]
                dh = Upsampled_y_dh[row-gradientMargin:row+gradientMargin+1, col-gradientMargin:col+gradientMargin+1]
                dw = Upsampled_y_dw[row-gradientMargin:row+gradientMargin+1, col-gradientMargin:col+gradientMargin+1]

                angle, strength, coherence = hashkey.hashkey(dh, dw, weighting, args)
                pixeltype = ((row-margin) % upscale) * upscale + ((col-margin) % upscale)

                ATA = np.dot(patch.T, patch)
                ATb = np.dot(patch.T, pixelHR)
                Q[angle,strength,coherence,pixeltype] += ATA
                V[angle,strength,coherence,pixeltype] += ATb
                binCount[angle,strength,coherence,pixeltype] += 1

        print("process image %s, patches: %d , (%d / %d)..." % (file, cnt, i, len(trainList)))


    ## optimize h for each bucket 
    CNT= 0
    print('Computing h ...')
    flag = np.zeros((Qangle, Qstrength, Qcoherence, upscale*upscale))
    for i in range(0, Qangle):
        for j in range(0, Qstrength):
            for t in range(0, Qcoherence):
                for k in range(0, upscale):
                    if(binCount[i,j,t,k] >= 100):             # if dataset in one bin is small, skip
                        h[i,j,t,k] = np.linalg.inv(Q[i,j,t,k] + 0.001).dot(V[i,j,t,k])
                        flag[i,j,t,k] = 1
                    else:
                        CNT = cnt + 1

    print("skip %d bin since the dataset in this bin is too small..." % CNT)
    np.save("filters.npy", h)
    np.save("flag.npy", flag)

if __name__ == "__main__":
    train()
