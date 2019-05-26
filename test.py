import os, argparse, math
import numpy as np
from PIL import Image
from helper import Args, utils, hashkey

args = Args.getArgs()

upscale = args.rate
patchSize = args.patchSize
gradientSize = args.neigborSize
margin = math.floor(max(patchSize, gradientSize) / 2)
patchMargin = math.floor(patchSize / 2)
gradientMargin = math.floor(gradientSize / 2)
testPath = args.test_dataset


if __name__ == "__main__":
    weighting = utils.gaussian2d([gradientSize, gradientSize], 2)
    weighting = np.diag(weighting.ravel())
    testList = utils.loadImageList(testPath)
    for i, file in enumerate(testList):
        HR = np.array(Image.open(testPath +file))
        HR = utils.setImageAlign(HR, alignment=upscale)
        LR = utils.resize(HR, 1/upscale, method=Image.BICUBIC)
        HR = HR / 255.
        LR = LR / 255.
        HR_y = utils.rgb2ycbcr(HR, only_y=True)
        LR_y = utils.rgb2ycbcr(LR, only_y=True)
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

                sr_y[row, col] = (h[angle, strength, coherence, pixeltype].T * patch).ravel()

        sr_y = np.clip(sr_y * 255.0, 0, 255)
        sr_y = np.uint8(sr_y)

        Image.fromarray(sr_y).save('./result/' + file)


