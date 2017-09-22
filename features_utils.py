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
    parser.add_argument('folder', default='.', type=str, action='store', nargs='?',
                        help='folder where data is stored (default: .)')
    parser.add_argument('--features-file', default=None,
                        help='output features file name (deault: features.json)')
    parser.add_argument('--protocols-file', default=None,
                        help='output protocols file name (deault: protocols.json)')
    parser.add_argument('-o', '--suffix', default='',
                        help='suffix for the output file names (default: no suffix)')

    args = parser.parse_args(args=sys.argv[2:])

    if not os.path.isdir(args.folder):
        print('%s: %s: no such directory.' % (progname,args.folder))
        sys.exit(1)
    folder = args.folder
    while folder[-1] == '/':
        folder = folder[:-1]

    if args.features_file is None:
        features_file = 'features_' + args.suffix.replace('_','') + '.json'
    elif suffix != '':
        print('Suffix will be ignored.')

    if args.protocols_file is None:
        protocols_file = 'protocols_' + args.suffix.replace('_','') + '.json'
    elif suffix != '' and args.features_file is None:
        print('Suffix will be ignored.')

    amplitudes = []
    features = []
    for cell in [os.path.join(folder, f) for f in os.listdir(folder) \
                 if os.path.isdir(os.path.join(folder,f))]:
        amplitudes.append(pickle.load(open(cell + '/amplitudes.pkl','r')))
        features.append(pickle.load(open(cell + '/features.pkl','r')))

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
    parser.add_argument('folder', default='.', type=str, action='store', nargs='?',
                        help='folder where data is stored (default: .)')
    parser.add_argument('-F', '--sampling-rate', default=20., type=float,
                        help='the sampling rate at which data was recorded (default 20 kHz)')
    parser.add_argument('--history-file', default='history.txt', type=str,
                        help='history file (default: history.txt)')
    parser.add_argument('--stim-dur', default=500., type=float,
                        help='Stimulus duration (default 500 ms)')
    parser.add_argument('--stim-start', default=125., type=float,
                        help='Stimulus duration (default 125 ms)')
    parser.add_argument('--recording-dur', default=1000., type=float,
                        help='Recording duration (default 1000 ms)')

    args = parser.parse_args(args=sys.argv[2:])

    if not os.path.isdir(args.folder):
        print('%s: %s: no such directory.' % (progname,args.folder))
        sys.exit(1)
    folder = os.path.abspath(args.folder)

    if args.sampling_rate <= 0:
        print('%s: the sampling rate must be positive.' % progname)
        sys.exit(2)
        
    history_file = folder + '/' + args.history_file
    if not os.path.isfile(history_file):
        print('%s: %s: no such file.' % (progname,args.history_file))
        sys.exit(3)

    if args.stim_dur <= 0:
        print('The duration of the stimulus must be positive.')
        sys.exit(4)

    if args.recording_dur <= 0 or args.recording_dur < args.stim_dur:
        print('The duration of the recording must be positive and greater than that of the stimulus.')
        sys.exit(4)

    if args.stim_start <= 0 or args.stim_start+args.stim_dur > args.recording_dur:
        print('The beginning of the stimulus must be within 0 and the end of the recording minus the stimulus duration.')
        sys.exit(4)

    info = read_tab_delim_file(history_file)

    stim_dur = args.stim_dur
    total_dur = args.recording_dur
    stim_start = args.stim_start
    stim_end = stim_start + stim_dur

    nsamples = total_dur * args.sampling_rate + 1
    time = np.arange(nsamples) / args.sampling_rate

    durations = []
    amplitudes = []
    traces = []

    for f in glob.glob(folder + '/ad0_*.ibw'):
        data = ibw.load(f)

        if data['wave']['wave_header']['npnts'] != nsamples:
            continue

        try:
            sweep_index = int(f[:-4].split('_')[-1])
        except:
            continue
        
        idx, = np.where(np.array(info['sweep_index']) == sweep_index)
        if len(idx) > 2:
            continue
        if info['channel_units'][idx[0]] == 'pA':
            jdx = idx[0]
        else:
            jdx = idx[1]

        voltage = data['wave']['wData']
        values = info['builder_parameters'][jdx].split(';')
        dur = float(values[0].split('=')[1])
        amp = float(values[1].split('=')[1]) * info['multiplier'][jdx]
        if np.abs(dur-stim_dur) < 1e-6  and amp > 0 and np.max(voltage) > 0:
            amplitudes.append(amp)
            durations.append(dur)
            trace = {'T': time, 'V': voltage,
                     'stim_start': [stim_start], 'stim_end': [stim_end]}
            #plt.plot(trace['T']*1e-3,trace['V'])
            #plt.show()
            traces.append(trace)

    features = efel.getFeatureValues(traces,feature_names)

    idx = np.argsort(amplitudes)
    amplitudes = [amplitudes[jdx] for jdx in idx]
    durations = [durations[jdx] for jdx in idx]
    features = [features[jdx] for jdx in idx]

    pickle.dump(features,open(folder + '/features.pkl','w'))
    pickle.dump(amplitudes,open(folder + '/amplitudes.pkl','w'))
    pickle.dump(durations,open(folder + '/durations.pkl','w'))

    print('-------------------------------------------------------')
    for feat,amp in zip(features,amplitudes):
        print('>>> Amplitude: %f pA' % amp)
        for name,values in feat.items():
            try:
                print('%s has the following values: %s' % \
                      (name, ', '.join([str(x) for x in values])))
            except:
                print('{} has the following values: \033[91m'.format(name) + str(values) + '\033[0m')
        print('-------------------------------------------------------')


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
