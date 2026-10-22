"""Native candidate images use the exact original waveform-block fold IDs."""
import json
from pathlib import Path
import numpy as np


def select(first,last,native_time,native_information,training):
    epochs=np.asarray(native_time).reshape(-1,16);information=np.asarray(native_information).reshape(-1,16)
    if len(epochs)!=len(training):raise ValueError('Training folds require original waveform block identities')
    block=np.searchsorted(epochs[:,0],first,side='right')-1;inside=(block>=0)&(block<len(epochs));block=np.clip(block,0,len(epochs)-1)
    inside&=(np.asarray(first)>=epochs[block,0]-2e-8)&(np.asarray(last)<=epochs[block,-1]+2e-8)
    supported=information.sum(axis=1)>0
    return inside&np.asarray(training)[block]&supported[block]


def apply(apertures,directory):
    directory=Path(directory);description=json.loads((directory/'analysis.json').read_text())
    if 'composite_range_reference' in description:return apply(apertures,description['composite_range_reference']['outer'])
    output=[]
    for c,a in enumerate(apertures):
        with np.load(directory/f'channel-{c}'/'epochs.npz') as source:
            mask=select(a['first_time_s'],a['last_time_s'],source['native_time_s'],source['native_information'],source['training'])
        record=dict(a);record['information']=np.asarray(a['information'])*mask
        record['training_epoch_selection']=dict(field=str(directory),selected_native_bins=int(mask.sum()),total_native_bins=len(mask),
            scope='Exact original 16-symbol waveform-block identities; original known-template erasures and recording-wide training folds')
        output.append(record)
    return output
