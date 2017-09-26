#!/usr/bin/env python

import os
import sys
import csv
import glob
import efel
import json
import pickle
import numpy as np
import igor.binarywave as ibw
import argparse as arg
import matplotlib.pyplot as plt

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.4g')

progname = os.path.basename(sys.argv[0])

feature_names = ['AP_height','AHP_slow_time','ISI_CV','doublet_ISI',
                 'adaptation_index2','mean_frequency','AHP_depth_abs_slow',
                 'AP_width','time_to_first_spike','AHP_depth_abs']

############################################################
###                        WRITE                         ###
############################################################


def write_features():
    parser = arg.ArgumentParser(description='Write configuration file using features from multiple cells.',\
                                prog=progname+' write')
    parser.add_argument('files', type=str, nargs='+',
                        help='pkl files containing the data relative to each cell')
    parser.add_argument('--features-file', default=None,
                        help='output features file name (deault: features.json)')
    parser.add_argument('--protocols-file', default=None,
                        help='output protocols file name (deault: protocols.json)')
    parser.add_argument('-o', '--suffix', default='',
                        help='suffix for the output file names (default: no suffix)')

    args = parser.parse_args(args=sys.argv[2:])

    for f in args.files:
        if not os.path.isfile(f):
            print('%s: %s: no such file.' % (progname,f))
            sys.exit(1)

    if args.features_file is None:
        features_file = 'features'
    if args.suffix != '':
        features_file += '_' + args.suffix.replace('_','')
    features_file += '.json'

    if args.protocols_file is None:
        protocols_file = 'protocols'
    if args.suffix != '':
        protocols_file += '_' + args.suffix.replace('_','')
    protocols_file += '.json'

    amplitudes = []
    features = []
    for f in args.files:
        data = pickle.load(open(f,'r'))
        features.append(data['features'])
        amplitudes.append(data['current_amplitudes'])

    nsteps = 3
    X = [[] for i in range(nsteps)]
    desired_amps = np.zeros((len(amplitudes),nsteps))
    for i,(amplitude,feature) in enumerate(zip(amplitudes,features)):
        amps = np.unique(amplitude)
        if len(amps) < nsteps:
            print('Not enough distinct amplitudes.')
            sys.exit(0)
        min_amp = amps[0]
        max_amp = amps[-1]
        amp_step = (max_amp-min_amp)/(nsteps-1)
        desired_amps[i,:] = np.arange(min_amp,max_amp+amp_step/2,amp_step)
        for j in range(nsteps):
            if not desired_amps[i,j] in amps:
                desired_amps[i,j] = amps[np.argmin(np.abs(amps - desired_amps[i,j]))]
        for j,amp in enumerate(desired_amps[i,:]):
            idx, = np.where(amplitude == amp)
            X[j].append([[feature[jdx][name] for jdx in idx] for name in feature_names])

    flatten = lambda l: [item for sublist in l for item in sublist]
    
    nfeatures = len(feature_names)
    Xm = np.nan + np.zeros((len(X),nfeatures))
    Xs = np.nan + np.zeros((len(X),nfeatures))
    for i,x in enumerate(X):
        for j in range(nfeatures):
            try:
                y = flatten([flatten(y[j]) for y in x])
                Xm[i,j] = np.mean(y)
                Xs[i,j] = np.std(y)
            except:
                pass

    protocols_dict = {}
    features_dict = {}
    for i in range(nsteps):
        stepnum = 'Step%d'%(i+1)
        protocols_dict[stepnum] = {'stimuli': [{
            'delay': 500, 'amp': np.mean(desired_amps[:,i])*1e-3,
            'duration': 2000, 'totduration': 3000}]}
        features_dict[stepnum] = {'soma': {}}
        for j in range(nfeatures):
            if not np.isnan(Xm[i,j]):
                features_dict[stepnum]['soma'][feature_names[j]] = [Xm[i,j],Xs[i,j]]

    json.dump(protocols_dict,open(protocols_file,'w'),indent=4)
    json.dump(features_dict,open(features_file,'w'),indent=4)


############################################################
###                       EXTRACT                        ###
############################################################


def extract_features_from_file(file_in,stim_dur,stim_start,sampling_rate):
    stim_end = stim_start + stim_dur

    data = ibw.load(file_in)
    voltage = data['wave']['wData']
    if len(voltage.shape) == 1:
        voltage = np.array([voltage])
    elif voltage.shape[0] > voltage.shape[1]:
        voltage = voltage.T
    time = np.arange(voltage.shape[1]) / sampling_rate

    idx, = np.where((time>stim_start) & (time<=stim_end))
    traces = [{'T': time[idx], 'V': sweep[idx], 'stim_start': [stim_start], 'stim_end': [stim_end]} \
              for sweep in voltage]
    plt.plot(time,voltage.T,lw=1)

    return efel.getFeatureValues(traces,feature_names)


def extract_features_from_files(files_in,current_amplitudes,stim_dur,stim_start,sampling_rate=20,files_out=[]):
    if type(files_out) != list:
        files_out = [files_out]
    if len(files_out) == 1:
        features = []
        with_spikes = []
        offset = 0
        for i,f in enumerate(files_in):
            feat = extract_features_from_file(f,stim_dur,stim_start,sampling_rate)
            jdx, = np.where([not all(v is None for v in f.values()) for f in feat])
            if len(jdx) > 0:
                for j in jdx:
                    features.append(feat[j])
                    with_spikes.append(offset + j)
            offset += len(feat)
        amplitudes = [current_amplitudes[i] for i in with_spikes]
        idx = np.argsort(amplitudes)
        amplitudes = [amplitudes[jdx] for jdx in idx]
        features = [features[jdx] for jdx in idx]
        data = {'features': features, 'current_amplitudes': amplitudes, \
                'stim_dur': stim_dur, 'stim_start': stim_start}
        pickle.dump(data,open(files_out[0],'w'))
    else:
        if len(files_out) == 0:
            files_out = map(lambda x: x+'.pkl',files_in)
        elif len(files_out) != len(files_in):
            raise Exception('There must be as many input as output files')
        for f_in,f_out in zip(files_in,files_out):
            feat,_ = extract_features_from_file(f_in)
            pickle.dump(feat,open(f_out,'w'))

    print('-------------------------------------------------------')
    for feat,amp in zip(features,amplitudes):
        print('>>> Amplitude: %g nA' % amp)
        for name,values in feat.items():
            try:
                print('%s has the following values: %s' % \
                      (name, ', '.join([str(x) for x in values])))
            except:
                print('{} has the following values: \033[91m'.format(name) + str(values) + '\033[0m')
        print('-------------------------------------------------------')


def read_tab_delim_file(filename):
    with open(filename,'r') as fid:
        header = fid.readline()
        keys = [k.strip().lower().replace(' ','_') for k in header.split('\t')]
        data = {k: [] for k in keys}
        reader = csv.reader(fid,delimiter='\t')
        for line in reader:
            for k,v in zip(keys,line):
                if v == '':
                    data[k].append(None)
                elif '.' in v:
                    try:
                        data[k].append(float(v))
                    except:
                        data[k].append(v)
                else:
                    try:
                        data[k].append(int(v))
                    except:
                        data[k].append(v)
    return data


def extract_features():
    parser = arg.ArgumentParser(description='Extract ephys features from recordings.',\
                                prog=progname+' extract')
    parser.add_argument('-d', '--folder', default=None,
                        help='the folder where data is stored')
    parser.add_argument('-f', '--file', default=None,
                        help='the file where data is stored')
    parser.add_argument('-F', '--sampling-rate', default=20., type=float,
                        help='the sampling rate at which data was recorded (default 20 kHz)')
    parser.add_argument('--history-file', default='history.txt', type=str,
                        help='history file (default: history.txt)')
    parser.add_argument('--stim-dur', default=500., type=float,
                        help='Stimulus duration (default 500 ms)')
    parser.add_argument('--stim-start', default=125., type=float,
                        help='Beginning of the stimulus (default 125 ms)')
    parser.add_argument('--Imin', default=None, type=float,
                        help='Minimum injected current, in nA')
    parser.add_argument('--Istep', default=0.1, type=float,
                        help='Current step (default 0.1 nA)')

    args = parser.parse_args(args=sys.argv[2:])

    if args.folder is not None and args.file is not None:
        print('Cannot specify both folder and file simultaneously.')
        sys.exit(1)

    if args.folder is not None:
        if not os.path.isdir(args.folder):
            print('%s: %s: no such directory.' % (progname,args.folder))
            sys.exit(1)
        folder = os.path.abspath(args.folder)
        mode = 'CA3'
    else:
        if not os.path.isfile(args.file):
            print('%s: %s: no such file.' % (progname,args.file))
            sys.exit(1)
        filename = os.path.abspath(args.file)
        folder = os.path.dirname(filename)
        mode = 'cortex'

    if args.sampling_rate <= 0:
        print('%s: the sampling rate must be positive.' % progname)
        sys.exit(2)

    if args.stim_dur <= 0:
        print('The duration of the stimulus must be positive.')
        sys.exit(3)

    if mode == 'CA3':
        history_file = folder + '/' + args.history_file
        if not os.path.isfile(history_file):
            print('%s: %s: no such file.' % (progname,args.history_file))
            sys.exit(4)
        info = read_tab_delim_file(history_file)

    if mode == 'CA3':
        files_in = []
        file_out = folder.split('/')[-1] + '.pkl'
        current_amplitudes = []
        n = len(info['sweep_index'])
        for i in range(n):
            if info['builder_name'][i] == 'StepPulse' and info['sweep_index'].count(info['sweep_index'][i]) == 2:
                params = info['builder_parameters'][i].split(';')
                for p in params:
                    if 'duration' in p:
                        dur = float(p.split('=')[1])
                    elif 'amplitude' in p:
                        amp = float(p.split('=')[1])*info['multiplier'][i]*1e-3
                if dur == args.stim_dur and amp>0:
                    print('[%02d] dur=%g ms, amp=%g nA' % (info['sweep_index'][i],dur,amp))
                    files_in.append('ad0_%d.ibw' % info['sweep_index'][i])
                    current_amplitudes.append(amp)
    elif mode == 'cortex':
        files_in = [filename]
        file_out = os.path.basename(filename).split('.')[0] + '.pkl'
        data = ibw.load(filename)
        Istep = args.Istep
        nsteps = len(data['wave']['wData'])
        if args.Imin is None:
            s = np.std(data['wave']['wData'],0)
            Imin = -args.Istep*np.argmin(s)
            print('Guessing the minimum value of injected current: %g nA.' % Imin)
        else:
            Imin = args.Imin
        current_amplitudes = np.arange(nsteps)*Istep + Imin
        current_amplitudes = current_amplitudes.tolist()

    extract_features_from_files(files_in,current_amplitudes,args.stim_dur,args.stim_start,
                                args.sampling_rate,files_out=[folder+'/'+file_out])

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.savefig(folder+'/'+file_out.split('.pkl')[0]+'.pdf',dpi=300)
    plt.show()



############################################################
###                         HELP                         ###
############################################################


def help():
    if len(sys.argv) > 2 and sys.argv[2] in commands:
        cmd = sys.argv[2]
        sys.argv = [sys.argv[0], cmd, '-h']
        commands[cmd]()
    else:
        print('Usage: %s <command> [<args>]' % progname)
        print('')
        print('Available commands are:')
        print('   extract        Extract the features from a given cell.')
        print('   write          Write a configuration file using data from multiple cells.')
        print('')
        print('Type \'%s help <command>\' for help about a specific command.' % progname)


############################################################
###                         MAIN                         ###
############################################################


# all the commands currently implemented
commands = {'help': help, 'extract': extract_features, 'write': write_features}

def main():
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        commands['help']()
        sys.exit(0)
    if not sys.argv[1] in commands:
        print('%s: %s is not a recognized command. See \'%s --help\'.' % (progname,sys.argv[1],progname))
        sys.exit(1)
    commands[sys.argv[1]]()


if __name__ == '__main__':
    main()
