import os
import numpy as np
from scipy.io import loadmat

def load_data_to_timseries(data_dir, input_dim, output_dim, seq_len, stride):
    # Load data, separate and stack sequences
    X, Y = np.zeros((0,seq_len,input_dim)), np.zeros((0,seq_len,output_dim))
    for matFile in sorted(os.listdir(data_dir)):
        mat = loadmat(os.path.join(data_dir,matFile))
        total_len = len(mat['X_markers'])
        assert total_len >= seq_len, 'Sequence length longer than data length.'
        mat['X_markers'] = ( mat['X_markers'] - mat['X_markers'][:,[0]] )[:,1:] # Minus and remove the base/reference coordinate
        mat['Y_markers'] = ( mat['Y_markers'] - mat['Y_markers'][:,[0]] )[:,1:] # Minus and remove the base/reference coordinate
        feature = np.concatenate((mat['Pressure'], mat['Flex']), axis=1)
        feature = np.stack([feature[i:i + seq_len] for i in range(0, len(feature)-seq_len+1, stride)])
        response = np.concatenate((mat['Force'], mat['ForcePos'], mat['X_markers'], mat['Y_markers']), axis=1)
        response = np.stack([response[i:i + seq_len] for i in range(0, len(response)-seq_len+1, stride)])
        X, Y = np.concatenate((X,feature), axis=0), np.concatenate((Y,response), axis=0)
    return X, Y

def prepare_data(data_dir, input_dim, output_dim, seq_len, stride):
    ## Determine normalizing parameters on "flattened" array first.
    X, Y = load_data_to_timseries(data_dir, input_dim, output_dim, 1, 1)
    keys = ['x_min', 'x_max', 'y_min', 'y_max', 'min_pressure', 'min_flex', 'min_force', 'x', 'y']
    d = {}
    for key in keys: d[key] = []
    x = X.reshape(-1, input_dim)
    y = Y.reshape(-1, output_dim)
    d['x_min'] = np.min(x, axis=0)
    d['x_max'] = np.max(x, axis=0)
    d['y_min'] = np.min(y, axis=0)
    d['y_max'] = np.max(y, axis=0)
    d['min_pressure'] = x[:,0].min(); x[:,0] -= x[:,0].min()
    d['min_flex'] = x[:,1].min(); x[:,1] -= x[:,1].min()
    d['min_force'] = y[:,0].min(); y[:,0] -= y[:,0].min()
    d['x'] = (np.mean(x,axis=0), np.std(x,axis=0))
    d['y'] = (np.mean(y,axis=0), np.std(y,axis=0))

    X, Y = load_data_to_timseries(data_dir, input_dim, output_dim, seq_len, stride)
    return X, Y, d
    
def normalize_data(X, Y, norm_param, method):
    X_norm = X.reshape(-1, X.shape[-1]).copy()
    Y_norm = Y.reshape(-1, Y.shape[-1]).copy()
    if method == 'scale':
        X_norm = (X_norm - norm_param['x_min']) / (norm_param['x_max'] - norm_param['x_min'])
        X_norm = 2 * X_norm - 1
        Y_norm = (Y_norm - norm_param['y_min']) / (norm_param['y_max'] - norm_param['y_min'])
        Y_norm = 2 * Y_norm - 1
    elif method == 'standardize':
        X_norm[:,0] -= norm_param['min_pressure']
        X_norm[:,1] -= norm_param['min_flex']
        X_norm = ( X_norm - norm_param['x'][0] ) / norm_param['x'][1]
        Y_norm[:,0] -= norm_param['min_force']
        Y_norm = ( Y_norm - norm_param['y'][0] ) / norm_param['y'][1]
    else:
        raise TypeError("Method not known. Only scale or standardize.")
    X_norm = X_norm.reshape(*X.shape)
    Y_norm = Y_norm.reshape(*Y.shape)
    return X_norm, Y_norm
    
def denormalize_data(pred, norm_param, method, std=None):
    pred_denorm = pred.reshape(-1, pred.shape[-1]).copy()
    if method == 'standardize':
        pred_denorm = pred_denorm * norm_param['y'][1] + norm_param['y'][0]
        pred_denorm[:,0] += norm_param['min_force']
        if std is not None: std_denorm = std * norm_param['y'][1]
    elif method == 'scale':
        pred_denorm = 0.5 * (pred_denorm + 1)
        pred_denorm = pred_denorm * (norm_param['y_max'] - norm_param['y_min']) + norm_param['y_min']
        if std is not None: std_denorm = std * (norm_param['y_max'] - norm_param['y_min']) / 2 
    else:
        raise TypeError("Method not known. Only scale or standardize.")
    pred_denorm = pred_denorm.reshape(*pred.shape)
    return pred_denorm, std_denorm
