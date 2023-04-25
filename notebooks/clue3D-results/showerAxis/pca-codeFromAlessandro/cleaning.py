import numpy as np
from sklearn.decomposition import PCA
from wpca import WPCA, EMPCA
import pandas as pd

def cleaning(_feats, pca_algo):

    X = _feats
    X = np.concatenate((X,X[:,4][:, np.newaxis]), axis=1) #After this step the structure of X is the following [x,y,z,L,energy,energy]
    X = np.concatenate((X,X[:,4][:, np.newaxis]), axis=1) #After this step the structure of X is the following [x,y,z,L,energy,energy,energy]


    ####------------------------------------------------ fill array with max E LC per layer ------------------------------------------------####
    maxvalarr = []
    for i in range(int(np.max(X[:,3]))+1): # X[:, 3] is L thus i in range(maxLayer+1)
        if (len(X[:,3][X[:,3] == i]) ==0): # X[:,3][X[:,3] == i] is all values of L such that L == i ie number of layer clusters in layer i
            continue
        layi = X[:,4][X[:,3] == i].copy() # energies of LC on layer i
        maxidx = np.argmax(layi)
        xmax = X[:,0][X[:,3] == i][maxidx]
        ymax = X[:,1][X[:,3] == i][maxidx]
        zmax = X[:,2][X[:,3] == i][maxidx]
        lmax = X[:,3][X[:,3] == i][maxidx]
        emax = X[:,4][X[:,3] == i][maxidx]
        maxvalarr.append([xmax,ymax,lmax,emax,zmax])

    maxvalarr = np.array(maxvalarr) # List of LC that are the maximum energy LC on their layer, as [x, y, layer, E, z]

    #### get maxE LC index
    #Consider only layers<26 to find the maxE LC layer
    tmp = np.array([i for i in X if i[3]<26]) # all LC with layer < 26
    maxidx = np.argmax(tmp[:,4]) # index of max energy LC overall (layer < 26)
    maxl = tmp[maxidx,3] # layer of max energy LC overall (layer < 26)
    maxe = tmp[maxidx,4]
    maxx = tmp[maxidx,0]
    maxy = tmp[maxidx,1]
    maxz = tmp[maxidx,2]

    ####------------------------------------------------ fill array with max E LC per layer for +-N layers from the maxE LC layer ------------------------------------------------####
    cleanarr = []
    for i in range(0,16,1):
        if len(X[:,4][X[:,3] == maxl + i])==0 :
            continue

        maxeidx = np.argmax(X[:,4][X[:,3] == maxl + i])
        xclean = X[:,0][X[:,3] == maxl + i][maxeidx]
        yclean = X[:,1][X[:,3] == maxl + i][maxeidx]
        zclean = X[:,2][X[:,3] == maxl + i][maxeidx]
        lclean = X[:,3][X[:,3] == maxl + i][maxeidx]
        eclean = X[:,4][X[:,3] == maxl + i][maxeidx]
        cleanarr.append([xclean,yclean,zclean,lclean,eclean,eclean,eclean]) #eclean is added three times to prepare the array for the weighted PCA

    for i in range(-1,-11,-1):
        if len(X[:,4][X[:,3] == maxl + i])==0 :
            continue

        maxeidx = np.argmax(X[:,4][X[:,3] == maxl + i])
        xclean = X[:,0][X[:,3] == maxl + i][maxeidx]
        yclean = X[:,1][X[:,3] == maxl + i][maxeidx]
        zclean = X[:,2][X[:,3] == maxl + i][maxeidx]
        lclean = X[:,3][X[:,3] == maxl + i][maxeidx]
        eclean = X[:,4][X[:,3] == maxl + i][maxeidx]
        cleanarr.append([xclean,yclean,zclean,lclean,eclean,eclean,eclean])

    cleanarr = np.array(cleanarr)

    ####------------------------------------------------ PCA with +-N layers from the maxE LC layer ------------------------------------------------####
    cleanarr_pca = np.array([i for i in cleanarr if i[3]<26]) # To compute the PCA axis consider only LC in the EM compartment


    #Energy-weighted PCA
    if pca_algo == 'std':
        pca = PCA(n_components=3)
        pca.fit(cleanarr_pca[:,:3])
    elif pca_algo == 'stdAllLCs':
        pca = PCA(n_components=3)
        pca.fit(X[:,:3])
    elif pca_algo == 'eWeighted':
        pca = WPCA(n_components=3)
        pca.fit(cleanarr_pca[:,:3], weights = cleanarr_pca[:,4:])
    elif pca_algo == 'eWeightedAllLCs':
        pca = WPCA(n_components=3)
        pca.fit(X[:,:3], weights = X[:,4:])

    mincompidx = np.argmax(pca.explained_variance_)

    pca_axis        = np.array([*pca.components_[0]])
    pca_axis_sub    = np.array([*pca.components_[1]])
    pca_axis_subsub = np.array([*pca.components_[2]])

    origin = [maxx,maxy,maxl,maxz]

    ####------------------------------------------------ fill array with LC with least dist to PCA ------------------------------------------------####
    pcaminarr = []
    distance = []
    index = []
    for i in range(int(np.max(X[:,3]))+1): #Loop on layers
        if (len(X[:,3][X[:,3] == i]) ==0):
            continue
        XL = X[X[:,3] == i].copy()
        dpar = []
        for j in range(XL.shape[0]): #Loop over lcs in layer
            dist = np.linalg.norm(np.cross(pca.components_[mincompidx],XL[j,:3] - np.array([maxx,maxy,maxz])))
            dpar.append(dist)
        dparmin = np.argmin(dpar)
        index.append(i)
        distance.append(dpar[dparmin])

        xpcamin = X[:,0][X[:,3] == i][dparmin]
        ypcamin = X[:,1][X[:,3] == i][dparmin]
        zpcamin = X[:,2][X[:,3] == i][dparmin]
        lpcamin = X[:,3][X[:,3] == i][dparmin]
        epcamin = X[:,4][X[:,3] == i][dparmin]
        pcaminarr.append([xpcamin,ypcamin,lpcamin,epcamin,zpcamin])

    pcaminarr = np.array(pcaminarr)


    ####----------------------------- fill array with LC with least dist to PCA with +-N layers from the maxE LC lay--------------------------------####
    pcaminarr = np.array(pcaminarr)
    pcaminarr = pd.DataFrame(pcaminarr, columns=['x','y','layer','energy','z'])
    cleanpcaarr = np.array(pcaminarr[(pcaminarr['layer']<maxl+15) & (pcaminarr['layer']>maxl-12)])

    return cleanpcaarr, pca_axis, origin, pca_axis_sub, pca_axis_subsub
