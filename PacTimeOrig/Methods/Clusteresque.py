import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance as EMD
import numpy.matlib
import pickle

# approaches:
# .1 affinity prop over windows
# 2.
# .2 clustering with Umap using sub-setted windows. Each window then gets flattened together, creating a variable x time-point side input
# .3 Could also parametric U-map with convolutional network as input.
# CNN + VAE with LSTM maybe?
# https://github.com/tejaslodaya/timeseries-clustering-vae
# Self-supervised learning in general
# ARHMM (higher order is probably best)
# TWARHMM
# Similarity measures on pre-defined windows
# parametric UMAP with CNN encoder on time
# DMD with HMM?




def getp1datallsessions(datall,varname,winlength, n_sessions):
    '''
    This works to concatenate all those single prey session data formatted by the window length
    :param datall:
    :param varname:
    :param winlength:
    :param n_sessions:
    :return: returns a dataframe in the format winlength x windows for a given variable "varname"
    '''
    dattmp = []  # list of dataframes
    for i in range(n_sessions):  # loop over sessions and retrieve data
        session = i + 1
        dat = getvarforclustering(datall[str(session)]['p1trialdat'], varname, winlength)
        dattmp.append([pd.DataFrame(dat)])

    # Now combine all sessions
    datcombo = pd.DataFrame()

    for i in range(n_sessions):
        datcombo = pd.concat((datcombo, dattmp[i][0]), axis=1)

    return datcombo

def loadEMDmatrices(fname):
    file = open(fname, 'rb')
    emdmatrices = pickle.load(file)

    return emdmatrices

def getvarforclustering(trialdat, varname, winlength):
    # Make vectors of each variable and then reshape
    tmp = pd.DataFrame()
    for trial in range(len(trialdat)):
        tmp = pd.concat((tmp, trialdat[trial][varname]), axis=0)

    tmp.reset_index(drop=True)
    tmp = tmp.to_numpy().reshape(int(len(tmp.to_numpy()) / winlength), winlength).T

    return tmp


def recusiveCDF(dat, histrange, varname='speed'):
    recdf=np.zeros((dat.shape[1],len(histrange[varname])-1))
    for i in tqdm(range(dat.shape[1])):
        a = np.histogram(dat[:, i], bins=histrange[varname])[0]
        a = a / np.histogram(dat[:, i], bins=histrange[varname])[0].sum()
        a = np.cumsum(a)  # compute cdf
        recdf[i,:] = a

    recdf = pd.DataFrame(recdf)
    return recdf


def EMDfast(recdf, normalize=True):
    tmp=[]
    for i in tqdm(range(recdf.columns.stop)):
        tmp.append([np.matlib.repmat(recdf[i].to_numpy(), len(recdf[i]), 1) - np.matlib.repmat(recdf[i].to_numpy(), len(recdf[i]),1).T])

    EMDmat = np.zeros(tmp[0][0].shape)
    for i in range(recdf.columns.stop):
        EMDmat += np.array(tmp[i][0])

    EMDmat = np.abs(EMDmat / recdf.columns.stop)

    if normalize:
        EMDmat = ((EMDmat * -1) / np.max(EMDmat))

    return EMDmat


def EMDRecursive(recdf, normalize=True):
    '''
    (1) recursively implement wasserstein distance for 1D variables
    (2) compute normalized distances
    (3) average variables for each bin'''
    EMDmat = np.zeros((len(recdf), len(recdf)))
    for i in tqdm(range(len(recdf))):
        for j in range(len(recdf)):
            EMDmat[i, j] = EMD(recdf.loc[i, :], recdf.loc[j, :])

    if normalize:
        EMDmat = ((EMDmat * -1) / np.max(EMDmat))

    return EMDmat


def normalize_range(vals, min, max):
    # TODO check
    return ((vals - np.min(vals)) / (np.max(vals) - np.min(vals)) * (max - min)) + min
