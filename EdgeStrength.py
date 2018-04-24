"""
 Edge Strength Transform adn Demonstration

 Mardigon Toler
 Gregory Hughes

 IDEA: Continuously find hit or miss transforms of image
 with a solid stucturing element. At each stage, find the edges of the
 current image.
 Form a new image (floating point) of normalized sum of all of the edge detected images
 Result should be an image of edges where intensity is related to thickness of edges.

 uses edge detection from scikit-image
"""
import numpy as np
import scipy.ndimage.morphology as scimorph
import skimage.io as skio
import skimage.filters as filters
import matplotlib.pyplot as plt
import skimage.util as util
import skimage.data as data
from skimage.feature import match_template

"""
 Similar to morphomatch, but marks the top left corner of the SE's
 location whenever
"""
def corner_match(I,B):
    M = len(I)
    N = len(I[0])
    m = len(B)
    n = len(B[0])

    S = np.zeros((M,N))
    for row in range(0,M-m):
        for col in range(0,N-n):
            if sum(sum(I[row:row+m,col:col+n] == B)) == m*n:
                S[row,col] = 1
    return S


"""
Finds the edge strength transform of image I in n passes.
using SE as a structuring element (or template)
the default "hit or miss" is provided by
scikit-image, can be replaced with custom implementation
edge_thresh is the threshold value between 0 and 1 for binarizing edges.
edge_func is the function you wish to use for edge detection
Assumes I is a binary image (boolean type)
If demo is True, this method will show the progress at each step.
"""
def EdgeStrengths(I, SE, n, edge_thresh=0.5, feature_scanner=corner_match,
                  edge_func=filters.sobel,
                  demo = False):

    img = np.copy(I)
    hitMissed = [feature_scanner(I,SE)]
    hitMissedEdges = [edge_func(hitMissed[0])]
    show(hitMissed[0], hitMissedEdges[0])
    for i in range(1, n):
        # find the hit or miss transform of the previous stage
        hitMissed.append(feature_scanner(hitMissed[i-1],SE))
        # edge detected images are magnitudes from 0 to 1, so threshold
        hitMissedEdges.append(edge_func(hitMissed[i]) > edge_thresh )
        show(hitMissed[i], hitMissedEdges[i])

    # add images together and normalize
    return sum(hitMissedEdges)/n


def show(A,B=None):
    if B is not None:
        plt.subplot(1,2,1)
    skio.imshow(A)
    if B is not None:
        plt.subplot(1,2,2)
        skio.imshow(B)
    plt.show()




if __name__ == '__main__':
    # for demonstration, make a 3x3 solid structuring element
    se = np.asarray([[1,1,1],[1,1,1],[1,1,1]])
    camera = 255 - data.camera()
    camera = camera > 100
    show(camera)
    show(EdgeStrengths(camera,se,5))
