
import os
import time
import h5py as h5
import numpy as np

__all__ = ['save_h5_file','load_h5_file','make_output_filename']

def save_dict(fid, group, data):
    for key,value in data.iteritems():
        if isinstance(value, dict):
            new_group = fid.create_group(group.name + '/' + key)
            save_dict(fid, new_group, value)
        elif type(value) in (int,float,tuple,str) or np.isscalar(value):
            group.attrs.create(key,value)
        else:
            try:
                group.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=9)
            except:
                new_group = fid.create_group(group.name + '/' + key)
                for i,data in enumerate(value):
                    new_group.create_dataset('%04d'%i, data=data, compression='gzip', compression_opts=9)

def save_h5_file(filename, **kwargs):
    with h5.File(filename, 'w') as fid:
        save_dict(fid, fid, kwargs)

def load_dict(group, data):
    for k,v in group.iteritems():
        k = k.encode('ascii','ignore')
        try:
            data[k] = v[:]
        except:
            data[k] = {}
            load_dict(v, data[k])
    for k,v in group.attrs.iteritems():
        data[k.encode('ascii','ignore')] = v

def load_h5_file(filename):
    data = {}
    with h5.File(filename, 'r') as fid:
        load_dict(fid, data)
    return data

def make_output_filename(prefix='', extension='.out'):
    filename = prefix
    if prefix != '' and prefix[-1] != '_':
        filename = filename + '_'
    now = time.localtime(time.time())
    filename = filename + '%d%02d%02d-%02d%02d%02d' % \
        (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    if extension[0] != '.':
        extension = '.' + extension
    suffix = ''
    k = 0
    while os.path.exists(filename + suffix + extension):
        k = k+1
        suffix = '_%d' % k
    return filename + suffix + extension

def main():
    filename = 'spam.h5'
    save_h5_file(filename, a=1, b='spam', c=np.random.uniform(size=10000),
                 d={'e': 2, 'f': 'foo', 'g': [4.,5.,6.]}, h=(7,8,9), i=[np.arange(10),np.arange(11,16)])
    data = load_h5_file(filename)
    print('Contents of file %s:' % filename)
    print(data)

if __name__ == '__main__':
    main()
