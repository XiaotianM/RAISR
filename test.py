import os, argparse, math
import numpy as np
from PIL import Image
from helper import Args, utils, hashkey
import time


args = Args.getArgs()

upscale = args.rate
patchSize = args.patchSize
gradientSize = args.neigborSize
margin = math.floor(max(patchSize, gradientSize) / 2)
patchMargin = math.floor(patchSize / 2)
gradientMargin = math.floor(gradientSize / 2)
testPath = args.test_dataset


if __name__ == "__main__":

    h = np.load('models/filters.npy')
    flag = np.load('models/flag.npy')

    weighting = utils.gaussian2d([gradientSize, gradientSize], 2)
    weighting = np.diag(weighting.ravel())
    testList = utils.loadImageList(testPath)
    for i, file in enumerate(testList):
        print("processing [%d / %d] image..." % (i+1, len(testList)))
        HR = np.array(Image.open(testPath +file))
        HR = utils.setImageAlign(HR, alignment=upscale)
        LR = utils.resize(HR, 1/upscale, method=Image.BICUBIC)
        HR = HR / 255.
        LR = LR / 255.
        HR_y = utils.rgb2ycbcr(HR, only_y=True)
        LR_y = utils.rgb2ycbcr(LR, only_y=True)
        
        start = time.time()
        Upsampled_y = utils.bilinear(LR_y, upscale) 
        sr_y = Upsampled_y.copy()        
        Upsampled_y_dh, Upsampled_y_dw = np.gradient(Upsampled_y)

        height, width = Upsampled_y.shape
        for row in range(margin, height-margin, 1):
            for col in range(margin, width-margin, 1):
                patch = Upsampled_y[row-patchMargin:row+patchMargin+1, col-patchMargin:col+patchMargin+1]
                patch = patch.reshape([1, patchSize*patchSize])

                dh = Upsampled_y_dh[row-gradientMargin:row+gradientMargin+1, col-gradientMargin:col+gradientMargin+1]
                dw = Upsampled_y_dw[row-gradientMargin:row+gradientMargin+1, col-gradientMargin:col+gradientMargin+1]

                angle, strength, coherence = hashkey.hashkey(dh, dw, weighting, args)
                pixeltype = ((row-margin) % upscale) * upscale +((col-margin) % upscale)

                if flag[angle, strength, coherence, pixeltype] == 1:
                    sr_y[row, col] = patch.dot(h[angle, strength, coherence, pixeltype]).ravel()

        sr_y = np.clip(sr_y * 255.0, 0, 255)
        sr_y = np.uint8(sr_y)
        print("Done... time cost %f" % (time.time() - start))

        Upsampled_y = np.clip(Upsampled_y * 255.0, 0, 255)
        Upsampled_y = np.uint8(Upsampled_y)

        Image.fromarray(sr_y).save('./result/' + file[:-4] + "_sr.png")
        Image.fromarray(Upsampled_y).save('./result/' + file[:-4] + "_bilinear.png")



