#!/usr/bin/env python

import h5py as h5
import numpy as np

def save_dict(fid, group, data):
    for key,value in data.iteritems():
        if isinstance(value, dict):
            new_group = fid.create_group(group.name + '/' + key)
            save_dict(fid, new_group, value)
        elif type(value) in (int,float,tuple,str):
            group.attrs.create(key,value)
        else:
            group.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=9)

def save_h5_file(filename, **kwargs):
    with h5.File(filename, 'w') as fid:
        save_dict(fid, fid, kwargs)

def load_h5_file(filename):
    with h5.File(filename, 'r') as fid:
        pass

def main():
    save_h5_file('spam.h5', a=1, b='spam', c=np.random.uniform(size=10000),
                 d={'e': 2, 'f': 'foo', 'g': [4.,5.,6.]}, h=(7,8,9))

if __name__ == '__main__':
    main()
