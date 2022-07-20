# CREDIT - https://github.com/bmershon/procrustes


#Purpose: Code that students fill in to implement Procrustes Alignment
#and the Iterative Closest Points Algorithm
import numpy as np

#Purpose: To compute the centroid of a point cloud
#Inputs:
#PC: 3 x N matrix of points in a point cloud
#Returns: A 3 x 1 matrix of the centroid of the point cloud
def getCentroid(PC):
    # mean of column vectors (axis 1) 
    return np.mean(PC, 1)[:, np.newaxis] 

#Purpose: Given an estimate of the aligning matrix Rx that aligns
#X to Y, as well as the centroids of those two point clouds, to
#find the nearest neighbors of X to points in Y
#Inputs:
#X: 3 x M matrix of points in X
#Y: 3 x N matrix of points in Y (the target point cloud)
#Cx: 3 x 1 matrix of the centroid of X
#Cy: 3 x 1 matrix of the centroid of corresponding points in Y
#Rx: Current estimate of rotation matrix for X
#Returns:
#idx: An array of size N which stores the indices 
def getCorrespondences(X, Y, Cx, Cy, Rx):
    X_ = np.dot(Rx, X - Cx);
    Y_ = Y - Cy;
    ab = np.dot(X_.T, Y_) # each cell is X_i dot Y_j
    xx = np.sum(X_*X_, 0)
    yy = np.sum(Y_*Y_, 0)
    D = (xx[:, np.newaxis] + yy[np.newaxis, :]) - 2*ab
    idx = np.argmin(D, 1)
    return idx 

#Purpose: Given correspondences between two point clouds, to center
#them on their centroids and compute the Procrustes alignment to
#align one to the other
#Inputs:
#X: 3 x M matrix of points in X
#Y: 3 x N matrix of points in Y (the target point cloud)
#Returns:
#A Tuple (Cx, Cy, Rx):
#Cx: 3 x 1 matrix of the centroid of X
#Cy: 3 x 1 matrix of the centroid of corresponding points in Y
#Rx: A 3x3 rotation matrix to rotate and align X to Y after
#they have both been centered on their centroids Cx and Cy
def getProcrustesAlignment(X, Y, idx):
    Cx = getCentroid(X)
    Cy = getCentroid(Y[:, idx])
    X_ = X - Cx
    Y_ = Y[:, idx] - Cy
    (U, S, Vt) = np.linalg.svd(np.dot(Y_, X_.T)) 
    R = np.dot(U, Vt)
    return (Cx, Cy, R)    

#Purpose: To implement the loop which ties together correspondence finding
#and procrustes alignment to implement the interative closest points algorithm
#Do until convergence (i.e. as long as the correspondences haven't changed)
#Inputs:
#X: 3 x M matrix of points in X
#Y: 3 x N matrix of points in Y (the target point cloud)
#MaxIters: Maximum number of iterations to perform, regardless of convergence
#Returns: A tuple of (CxList, CyList, RxList):
#CxList: A list of centroids of X estimated in each iteration (these
#should actually be the same each time)
#CyList: A list of the centroids of corresponding points in Y at each 
#iteration (these might be different as correspondences change)
#RxList: A list of rotations matrices Rx that are produced at each iteration
#This is all of the information needed to animate exactly
#what the ICP algorithm did
def doICP(X, Y, MaxIters, doPCA=True, use2d=False):
    if use2d:
        X = X[:2,:]
        Y = Y[:2,:]

    CxList = []
    CyList = []
    RxList = []
    Cx = getCentroid(X)
    Cy = getCentroid(Y)

    Covx = np.cov(X - Cx)
    Covy = np.cov(Y - Cy)

    if doPCA:
        from sklearn.decomposition import PCA
        pca_x = PCA()
        pca_x.fit((X-Cx).T)
        pca_y = PCA()
        pca_y.fit((Y-Cy).T)

        Rpcax = pca_x.components_
        Rpcay = pca_y.components_
        if not use2d:
            Rpcax[2,:] = np.cross(Rpcax[0,:], Rpcax[1,:])
            Rpcay[2,:] = np.cross(Rpcay[0,:], Rpcay[1,:])

        Rx = np.dot(Rpcay.T, Rpcax) # np.eye(3, 3)
    else:
        if use2d:
            Rx = np.eye(2)
        else:
            Rx = np.eye(3)

    CxList.append(Cx)
    CyList.append(Cy)
    RxList.append(Rx)
    last = Cy
    for i in range(MaxIters):
        idx = getCorrespondences(X, Y, Cx, Cy, Rx)
        (Cx, Cy, Rx) = getProcrustesAlignment(X, Y, idx)

        if use2d:
            nCx = np.zeros((3,1))
            nCy = np.zeros((3,1))
            nRx = np.eye(3)

            nCx[:2] = Cx
            nCy[:2] = Cy
            nRx[:2,:2] = Rx

            CxList.append(nCx)
            CyList.append(nCy)
            RxList.append(nRx)
        else:
            CxList.append(Cx)
            CyList.append(Cy)
            RxList.append(Rx)

        d = Cy - last
        if np.sum(d*d) == 0.0:
            break;
        last = Cy

    return (CxList, CyList, RxList)
