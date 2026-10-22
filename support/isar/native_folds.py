"""Temporal validation folds bound to complete native-symbol groups.

The native rows of a selected information-weighted bin always stay together.
This avoids using the two sides of one averaged observation as independent
training and test samples. Zero-weight bins retain their chronological mean.
"""
import numpy as np


def training_mask(time,first_s):
    return np.floor((np.asarray(time)-first_s)/.25).astype(int)%4!=1


def fitting_mask(time,first_s,selection='all'):
    """Use the complete aperture unless a legacy diagnostic subset is explicit."""
    if selection=='all':return np.ones(np.asarray(time).shape,bool)
    if selection=='diagnostic-subset':return training_mask(time,first_s)
    raise ValueError('Fitting samples must be all or diagnostic-subset')


def fit_sample_selection(requested=None,*,saved=None,reproduce=False):
    """Legacy replays retain their data; new estimation uses every sample."""
    selection=requested or ((saved or {}).get('fit_samples','diagnostic-subset') if reproduce else 'all')
    if selection not in ('all','diagnostic-subset'):raise ValueError('Unknown fitting sample selection')
    if reproduce and selection!=(saved or {}).get('fit_samples','diagnostic-subset'):
        raise ValueError('Changing fitting samples is a new fit, not an identical-likelihood replay')
    return selection


def native_indices(rows,information,first_s,last_s,fold,*,block_rows=16):
    rows,information=np.asarray(rows),np.asarray(information)
    if (fold not in ('all','training','testing') or not first_s<last_s or block_rows<1 or
            rows.ndim!=2 or rows.shape[1]<5 or information.shape!=(len(rows),) or
            np.any(information<0) or np.any(np.diff(rows[:,1])<=0)):
        raise ValueError('Ordered native rows, nonnegative information and a valid temporal fold are required')
    n=len(rows)//block_rows*block_rows
    time=rows[:n,1].reshape(-1,block_rows)
    weight=(information[:n]*rows[:n,4]).reshape(time.shape)
    total=weight.sum(axis=1)
    epoch=np.divide(np.sum(time*weight,axis=1),total,out=time.mean(axis=1),where=total>0)
    selected=(epoch>=first_s)&(epoch<last_s)
    if fold!='all':
        train=training_mask(epoch,first_s)
        selected&=train if fold=='training' else ~train
    return np.flatnonzero(np.repeat(selected,block_rows))
