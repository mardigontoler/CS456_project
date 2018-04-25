# -*- coding: utf-8 -*-

"""
Mardigon Toler
Gregory Hughes

Python implementations of morphological functions
"""

import skimage.util as util
import numpy as np

def morphoMatch(I,B,padVal):
    M = len(I)
    N = len(I[0])
    m = len(B)
    n = len(B[0])
    Ip = util.pad(I,[m,n],"constant",constant_values=[padVal])

    S = np.zeros((M,N))
    for x in range(0,M):
        for y in range (0,N):
            partial = False
            perfect = True
            for a in range(0,m):
                for b in range(0,n):
                    v = abs(B[a,b]-Ip[x+a,y+b])
                    if v == 1:
                        perfect = False
                    else:
                        partial = True
            if perfect:
                S[x,y] = 1
            elif partial:
                S[x,y] = 0.5
    return S

def morphoErode(I,B,padVal):
    return morphoMatch(I,B,padVal) > .5

def morphoDilate(I,B,padVal):
    return morphoMatch(I,B,padVal)  >= .5

def findROI(I):
    M = len(I)
    N = len(I[0])
    rMost = 25
    lMost = N-25
    uMost = M-25
    for row in range(25,M//2 - 24):
        for col in range (25,N-25):
            if I[row,col]:
                if col < lMost:
                    lMost = col
                if col > rMost:
                    rMost = col
                if row < uMost:
                    uMost = row
    return [lMost, rMost, uMost]

