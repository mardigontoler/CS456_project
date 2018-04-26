# -*- coding: utf-8 -*-
"""
Mardigon Toler
Gregory Hughes
"""

import morpho
import matplotlib.pyplot as plt
#import skimage.util as util
import skimage.color as color
import skimage.feature as feature
import skimage.io as io
import numpy as np

"""
Our first script for finding eyes using template matching"
"""

img = io.imread("test4.png")
imgB = color.rgb2grey(img) > 0.333
io.imshow(imgB)
plt.show()
se1 = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
se1 = np.asarray(se1)
imgE = morpho.morphoErode(imgB, se1, 0)
io.imshow(imgE)
plt.show()
imgD = morpho.morphoDilate(imgB, se1, 0)
io.imshow(imgD)
plt.show()

# find hit-or-miss transform
imgH = imgE ^ imgD
io.imshow(imgH)
plt.show()

#closing
imgC = morpho.morphoErode(morpho.morphoDilate(imgH, se1, 0), se1, 0)
io.imshow(imgC)
plt.show()

# region of interest
roi = morpho.findROI(imgC)
imgroi = imgH[roi[2]:len(imgH),roi[0]:roi[1]]
io.imshow(imgroi)
plt.show()
se2 = np.ones(len(imgroi[0])) >=1
unit = round(len(imgroi)/75)
mid = len(se2)//2
se2[mid-(unit*5):mid+(unit*5)] = 0
se2[0:(unit*3)] = 0
se2[len(se2)-(unit*3):len(se2)]
se2 = np.reshape(se2,(1,len(se2)))

# match SE for row with eyes
imgT = feature.match_template(imgroi, se2, True)
io.imshow(imgT)
plt.show()
imgF = feature.match_template(imgroi, se2, False)
io.imshow(np.repeat(imgF,100,axis=1))
plt.show()
