import numpy as np
from scipy.io import loadmat
from glemr import GLEMR
from utils import report, binarize


# load dataset
dataset_list = ['sj', '3dfe', 'ns', 'mov', 'alpha', 'heat', 'spo', 
                'diau', 'cdc', 'elu', 'gene', 'emo6', 'twit', 'flic']
dataset = '3dfe'
assert dataset in dataset_list
data = loadmat('datasets/%s.mat' % dataset)
X, D = data['features'], data['label_distribution']
L = binarize(D)

# load the optimal hyperparameters
p = np.load('config.npy', allow_pickle=True).item()[dataset]
p['trace_step'] = np.inf
p['verbose'] = 10

# train GLEMR
glemr = GLEMR(**p).fit(X, L)
Drec = glemr.label_distribution_

# report the results
report(Drec, D, ds=dataset)