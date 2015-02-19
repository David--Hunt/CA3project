#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse as arg
import itertools as it
import time
import copy
import CA3
from CA3.utils import *
from scipy.interpolate import interp1d,UnivariateSpline
from neuron import h
from emoo import Emoo
from emoo import mpi4py_loaded
if mpi4py_loaded:
    from mpi4py import MPI
    processor_name = MPI.Get_processor_name()

DEBUG = False
ReducedNeuron = CA3.cells.SimplifiedNeuron

# the list of objectives
objectives = ['hyperpolarizing_current_steps','spike_onset','spike_offset','isi']

# the variables and their lower and upper search bounds
if False:
    variables = [
        ['Cm', 0.99, 1.01],                       # [uF/cm2] membrane capacitance (0.6,3)
        ['Rm', 9.99e3, 10.01e3],                    # [Ohm cm2] membrane resistance (10e3,30e3)
        ['El', -71., -69.],                    # [mV] reversal potential of leak conductance (-85,-50)
        ['scaling', 0.49, 0.51],                  # [1] scaling of dendritic capacitance and membrane resistance (0.5,2)
        ['nat_gbar_soma', 499., 501.],           # [pS/um2] (0,500)
        ['nat_gbar_hillock', 4999., 5001.],      # [pS/um2] (0,20000)
        ['nat_gbar_ais', 9999., 10001.],          # [pS/um2] (0,20000)
        ['nat_half_dist', 49., 51.],           # [um] (0,500)
        ['nat_lambda', 9.9, 10.1],              # [um] (1,500)
        ['nat_dend_scaling', 0.09, 0.11],          # [1] (0,2)
        ['kdr_gbar', 49., 151.],                # [pS/um2] (0,100)
        ['kdr_dend_scaling', 0.49, 0.51]]          # [1] (0,2)

if True:
    variables = [
        ['Cm', 0.6, 3.],                       # [uF/cm2] membrane capacitance (0.6,3)
        ['Rm', 5e3, 30e3],                     # [Ohm cm2] membrane resistance (10e3,30e3)
        ['El', -85., -50.],                    # [mV] reversal potential of leak conductance (-85,-50)
        ['scaling', 0.3, 2.],                  # [1] scaling of dendritic capacitance and membrane resistance (0.5,2)
        ['nat_gbar_soma', 0., 500.],           # [pS/um2] (0,500)
        ['nat_gbar_hillock', 0., 20000.],      # [pS/um2] (0,20000)
        ['nat_gbar_ais', 0., 20000.],          # [pS/um2] (0,20000)
        ['nat_gbar_distal', 0., 100.],         # [pS/um2] (0,100)
        ['nat_lambda', 1., 100.],              # [um] (1,500)
        ['kdr_gbar_soma', 0., 100.],           # [pS/um2] (0,100)
        ['kdr_gbar_distal', 0., 10.],          # [pS/um2] (0,10)
        ['kdr_lambda', 1., 100.]]              # [um] (0,100)

if False:
    variables = [
        ['Cm', 0.6, 3.],                       # [uF/cm2] membrane capacitance (0.6,3)
        ['Rm', 10e3, 30e3],                    # [Ohm cm2] membrane resistance (10e3,30e3)
        ['El', -85., -50.],                    # [mV] reversal potential of leak conductance (-85,-50)
        ['scaling', 0.5, 2.],                  # [1] scaling of dendritic capacitance and membrane resistance (0.5,2)
        ['nat_gbar_soma', 0., 500.],           # [pS/um2] (0,500)
        ['nat_gbar_hillock', 0., 20000.],      # [pS/um2] (0,20000)
        ['nat_gbar_ais', 0., 20000.],          # [pS/um2] (0,20000)
        ['nat_half_dist', 0., 500.],           # [um] (0,500)
        ['nat_lambda', 1., 500.],              # [um] (1,500)
        ['nat_dend_scaling', 0., 2.],          # [1] (0,2)
        ['kdr_gbar', 0., 100.],                # [pS/um2] (0,100)
        ['kdr_dend_scaling', 0., 2.],          # [1] (0,2)
        ['nap_gbar', 0., 5.],                  # [pS/um2] in the paper, 0 < gbar < 4.1
        ['km_gbar', 0., 2.],                   # [pS/um2]
        ['kahp_gbar', 0., 500.],               # [pS/um2]
        ['kd_gbar', 0., 0.01],                 # [pS/um2]
        ['kap_gbar', 0., 100.],                # [pS/um2]
        ['ih_gbar_soma', 0., 0.1],             # [pS/um2]
        ['ih_dend_scaling', 0., 10.],          # [1]
        ['ih_half_dist', 0., 500.],            # [um]
        ['ih_lambda', 1., 500.]]               # [um]

# the neuron parameters that have been obtained by the previous
# optimization of the passive properties
neuron_pars = {'soma': {'Ra': None, 'area': None},
               'proximal': {'Ra': None, 'area': None, 'L': None},
               'distal': {'Ra': None, 'area': None, 'L': None},
               'basal': {'Ra': None, 'area': None, 'L': None}}

# the electrophysiological data used in the optimization
ephys_data = None

resampling_frequency = 200 # [kHz]

def make_simplified_neuron(parameters):
    pars = copy.deepcopy(neuron_pars)
    for k,v in parameters.iteritems():
        if k == 'scaling':
            pars[k] = v
        elif k == 'El':
            for lbl in 'soma','proximal','distal','basal':
                pars[lbl][k] = v
        elif k in ('Cm','Rm'):
            pars['soma'][k] = v
        else:
            key = k.split('_')[0]
            value = '_'.join(k.split('_')[1:])
            try:
                pars[key][value] = v
            except:
                pars[key] = {value: v}
    # the passive properties of the axon are the same as the soma
    pars['axon'] = pars['soma'].copy()
    try:
        pars['axon'].pop('L')
    except:
        pass
    try:
        pars['axon'].pop('diam')
    except:
        pass
    return ReducedNeuron(pars, with_axon=True, with_active=True)

def extract_average_trace(t,x,events,window,interp_dt=-1,token=None):
    if not token is None:
        logger('start','extract_average_trace',token)
    if interp_dt > 0:
        offset = 0.2
        window = [float(window[0])-offset,float(window[1])+offset]
    else:
        window = [float(w) for w in window]
    dt = t[1]-t[0]
    dx = (x[:,2:] - x[:,:-2]) / (2*dt)
    before = -np.round(window[0] / dt)
    after = np.round(window[1] / dt)
    T = np.arange(before+after+1)*dt
    X = np.zeros((sum([len(e) for e in events]),before+after+1))
    dX = np.zeros((sum([len(e) for e in events]),before+after+1))
    cnt = 0
    for i in range(x.shape[0]):
        for e in events[i]:
            k = np.round(e/dt)
            try:
                X[cnt,:] = x[i,k-before:k+after+1]
                dX[cnt,:] = dx[i,k-1-before:k+after]
                cnt += 1
            except:
                pass
    X = X[:cnt,:]
    dX = dX[:cnt,:]
    if interp_dt > 0:
        n = X.shape[0]
        Tint = np.arange(0,np.sum(np.abs(window)),interp_dt)
        Xint = np.zeros((n,len(Tint)))
        for i in range(n):
            Xint[i,:] = UnivariateSpline(T, X[i,:], k=3, s=0.5)(Tint)
        return extract_average_trace(Tint,Xint,
                                     [[Tint[i]] for i in np.argmax(Xint,axis=1)],
                                     [window[0]+offset, window[1]-offset],
                                     interp_dt=-1,token=token)
    if not token is None:
        logger('end','extract_average_trace',token)
    return T+window[0],np.mean(X,axis=0),np.mean(dX,axis=0)

def logger(log_type, msg, token):
    if log_type.lower() == 'start':
        message = '%d STARTED %s @ %s.' % (token,msg,timestamp())
    elif log_type.lower() == 'end':
        message = '%d FINISHED %s @ %s.' % (token,msg,timestamp())
    else:
        return
    print(message)
    
def current_steps(neuron, amplitudes, dt=0.05, dur=500, tbefore=100, tafter=100, V0=-70, token=None):
    if not token is None:
        logger('start', 'current_steps', token)
    neuron.save_properties('%09d.h5' % token)
    stim = h.IClamp(neuron.soma[0](0.5))
    stim.dur = dur
    stim.delay = tbefore
    rec = {'t': h.Vector(), 'v': h.Vector()}
    rec['t'].record(h._ref_t)
    rec['v'].record(neuron.soma[0](0.5)._ref_v)
    time = []
    voltage = []
    T = np.arange(0, dur+tbefore+tafter+dt/2, dt)
    V = np.zeros((len(amplitudes),len(T)))
    time = []
    voltage = []
    for i,amp in enumerate(amplitudes):
        stim.amp = amp
        CA3.utils.run(tend=dur+tbefore+tafter, V0=V0, temperature=36)
        f = interp1d(rec['t'],rec['v'])
        V[i,:] = f(T)
        time.append(np.array(rec['t']))
        voltage.append(np.array(rec['v']))
    stim.amp = 0
    del stim
    if not token is None:
        logger('end', 'current_steps', token)
    return T,V

def hyperpolarizing_current_steps_error(t,V,Iinj,Vref):
    #### ===>>> Iinj is an array of values, not a matrix like V and Vref
    idx, = np.where(Iinj <= 0)
    return np.sum((V[idx,:]-Vref[idx,:])**2)

def spike_shape_error(t,V,tp,window,token=None):
    if np.isscalar(window[0]):
        window = [window]
    m = min([w[0] for w in window])
    M = max([w[1] for w in window])
    tavg,Vavg,dVavg = extract_average_trace(t,V,tp,[m,M],interp_dt=1./resampling_frequency,token=token)
    err = []
    for w in window:
        idx, = np.where((ephys_data['tavg']>=w[0]) & (ephys_data['tavg']<=w[1]))
        jdx, = np.where((tavg>=w[0]) & (tavg<=w[1]))
        err.append(np.sum((Vavg[jdx] - ephys_data['Vavg'][idx])**2) + 0.1*np.sum((dVavg[jdx] - ephys_data['dVavg'][idx])**2))
    return err

def isi_error(tp):
    err = None
    for i in range(len(tp)):
        n = min(len(tp[i]),len(ephys_data['tp'][i]))
        if n > 2:
            if err is None:
                err = 0
            err += np.sum((np.diff(tp[i][:n]) - np.diff(ephys_data['tp'][i][:n]))**2)
    if err is None:
        return 1e10
    return err

def check_prerequisites(t,V,ton,toff,tp,Vp,width=None,token=None):
    retval = True
    if not token is None:
        logger('start','check_prerequisites',token)
    n = V.shape[0]
    idx, = np.where((t>toff-200) & (t<toff))
    for i in range(n):
        m = np.mean(V[i,idx])
        s = np.std(V[i,idx])
        # number of spikes in the reference trace (ephys data) and in the simulated one
        ref_nspikes = len(np.where((ephys_data['tp'][i]>ton) & (ephys_data['tp'][i]<=toff))[0])
        nspikes = len(np.where((tp[i]>ton) & (tp[i]<=toff))[0])
        # spikes where there shouldn't be any
        if ref_nspikes == 0 and nspikes != 0:
            print('%d check_prerequistes: spikes where there shouldn\'t be any.' % token)
            retval = False
            break
        # no spikes where there should be some
        elif ref_nspikes > 0 and nspikes == 0:
            print('%d check_prerequistes: no spikes where there should be some.' % token)
            retval = False
            break
        # spike block?
        elif ref_nspikes > 0 and nspikes > 0 and m > -40 and s < 1:
            print('%d check_prerequistes: mean(voltage) > -40 and std(voltage) < 1.' % token)
            retval = False
            break
        if not width is None:
            # no spike width should exceed 3 ms
            if nspikes > 0 and np.max(width[i]) > 3:
                retval = False
                break
        # decrease in spike height shouldn't exceed 20%
        if nspikes > 2:
            for j in range(2,nspikes):
                if Vp[i][j] < 0.8*Vp[i][0]:
                    print('%d check_prerequistes: spike height decreased by more than 20%%.' % token)
                    retval = False
                    break
            if not retval:
                break
    if not token is None:
        logger('end','check_prerequisites',token)
    return retval

def objectives_error(parameters):
    try:
        objectives_error.ncalls += 1
    except:
        objectives_error.__dict__['ncalls'] = 1
    token = int(1e9 * np.random.uniform())
    logger('start', 'objectives_error', token)
    # build the neuron with the current parameters
    neuron = make_simplified_neuron(parameters)
    # simulate the injection of current steps into the neuron
    t,V = current_steps(neuron, ephys_data['I_amplitudes'], ephys_data['dt'], ephys_data['dur'],
                        ephys_data['tbefore'], ephys_data['tafter'], np.mean(ephys_data['V'][:,0]), token)
    # extract significant features from the traces
    logger('start', 'extractAPPeak', token)
    tp,Vp = extractAPPeak(t, V, threshold=0, min_distance=1)
    logger('end', 'extractAPPeak', token)

    measures = {'hyperpolarizing_current_steps': 1e20,
                'spike_onset': 1e20,
                'spike_offset': 1e20,
                'isi': 1e20}

    if check_prerequisites(t,V,ephys_data['tbefore'],ephys_data['tbefore']+ephys_data['dur'],tp,Vp,token=token):
        logger('start', 'extractAPHalfWidth', token)
        Vhalf,width,interval = extractAPHalfWidth(t, V, threshold=0, tpeak=tp, Vpeak=Vp, interp=False)
        logger('end', 'extractAPHalfWidth', token)
        if check_prerequisites(t,V,ephys_data['tbefore'],ephys_data['tbefore']+ephys_data['dur'],tp,Vp,width,token):
            measures['hyperpolarizing_current_steps'] = hyperpolarizing_current_steps_error(t,V,ephys_data['I_amplitudes'],ephys_data['V'])
            measures['isi'] = isi_error(tp)
            err = spike_shape_error(t,V,tp,[[-0.5,-0.1],[0.1,14]],token)
            measures['spike_onset'] = err[0]
            measures['spike_offset'] = err[1]

    #import pylab as p
    #for i,v in enumerate(ephys_data['V']):
    #    p.plot(t,v,'k')
    #    p.plot(ephys_data['tp'][i],ephys_data['Vp'][i],'ko')
    #    p.plot(t,V[i,:],'r')
    #    p.plot(tp[i],Vp[i],'ro')
    #p.xlabel('Time (ms)')
    #p.ylabel('Membrane voltage (mV)')
    #p.show()

    logger('end', 'objectives_error ' + str(measures), token)
    return measures

def check_population(population, columns, gen):
    if emoo.master_mode:
        if gen == 0:
            CA3.utils.h5.save_h5_file(h5_filename, 'w', columns=columns)
        CA3.utils.h5.save_h5_file(h5_filename,'a',generations={('%d'%gen): population})

def optimize():
    # parse the command-line arguments
    parser = arg.ArgumentParser(description='Fit a reduced morphology to a detailed one considering only passive properties')
    parser.add_argument('filename', type=str, action='store', help='Path of the H5 file containing the results of the optimization of passive properties')
    parser.add_argument('-N', '--population-size', default=512, type=int,
                        help='Population size for the genetic algorithm (default: 700)')
    parser.add_argument('-G', '--generation-number', default=200, type=int,
                        help='Number of generations for the genetic algorithm (default: 200)')
    parser.add_argument('--pm', default=0.1, type=float,
                        help='Probability of mutation (default: 0.1)')
    parser.add_argument('--etam-start', default=10., type=float,
                        help='Initial value of the mutation parameter (default: 10)')
    parser.add_argument('--etam-end', default=500., type=float,
                        help='Final value of the mutation parameter (default: 500)')
    parser.add_argument('--etac-start', default=5., type=float,
                        help='Initial value of the crossover parameter (default: 5)')
    parser.add_argument('--etac-end', default=50., type=float,
                        help='Final value of the crossover parameter (default: 50)')
    parser.add_argument('-o','--out-file', type=str, help='Output file name (default: same as morphology file)')
    parser.add_argument('-d','--data-file', type=str, help='Data file for fitting')
    args = parser.parse_args(args=sys.argv[2:])

    if args.filename is None:
        print('You must provide the path of the H5 file with the results of the optimization of passive properties.')
        sys.exit(1)
    
    if not os.path.isfile(args.filename):
        print('%s: no such file.' % args.filename)
        sys.exit(2)

    if args.data_file is None:
        print('You must provide the path of the data file.')
        sys.exit(1)

    if not os.path.isfile(args.data_file):
        print('%s: no such file.' % args.data_file)
        sys.exit(2)

    # load the data relative to the optimization of the passive properties
    data = CA3.utils.h5.load_h5_file(args.filename)

    # output filename
    global h5_filename
    if args.out_file is None:
        h5_filename = CA3.utils.h5.make_output_filename(os.path.basename(data['swc_filename']).rstrip('.swc'), '.h5')
    else:
        h5_filename = args.out_file

    # find the ``best'' individual
    last = str(len(data['generations'])-1)
    err = np.zeros((data['generations'][last].shape[0],len(data['objectives'])))
    norm_err = np.zeros((data['generations'][last].shape[0],len(data['objectives'])))
    for i,obj in enumerate(data['objectives']):
        err[:,i] = data['generations'][last][:,data['columns'][obj]]
        norm_err[:,i] = (err[:,i] - min(err[:,i])) / (max(err[:,i]) - min(err[:,i]))
    best = np.argmin(np.sum(norm_err[:,:2],axis=1))

    # fill in the fixed parameters
    for lbl in 'soma','basal','proximal','distal':
        neuron_pars[lbl]['area'] = data['areas'][lbl]
        neuron_pars[lbl]['Ra'] = data['generations'][last][best,data['columns']['Ra_'+lbl]]
        if lbl != 'soma':
            neuron_pars[lbl]['L'] = data['generations'][last][best,data['columns']['L_'+lbl]]

    # which model to use
    global ReducedNeuron
    if data['model_type'].lower() == 'thorny':
        ReducedNeuron = CA3.cells.ThornyNeuron
    elif data['model_type'].lower() == 'athorny':
        ReducedNeuron = CA3.cells.AThornyNeuron
    elif data['model_type'].lower() != 'simplified':
        print('The model type must be one of "simplified", "thorny" or "athorny".')
        sys.exit(3)

    # load the ephys data to use in the optimization
    global ephys_data
    ephys_data = CA3.utils.h5.load_h5_file(args.data_file)
    # transpose the data so that number of rows = number of trials
    ephys_data['V'] = ephys_data['V'].transpose()
    ephys_data['I'] = ephys_data['I'].transpose()
    ephys_data['t'] = np.arange(ephys_data['V'].shape[1]) * ephys_data['dt']
    # find the duration, beginning and end of the stimulation
    i = 0
    while np.min(ephys_data['I'][i,:] >= 0):
        i += 1
    idx, = np.where(ephys_data['I'][i,:] < 0)
    ephys_data['tbefore'] = ephys_data['t'][idx[0]]
    ephys_data['dur'] = ephys_data['t'][idx[-1]] - ephys_data['tbefore']
    ephys_data['tafter'] = ephys_data['t'][-1] - ephys_data['dur'] - ephys_data['tbefore']
    # find the average spike shape: this makes sense only for regular spiking cells
    tp = [ephys_data['tp'][str(i)] for i in range(ephys_data['V'].shape[0])]
    ephys_data['tavg'],ephys_data['Vavg'],ephys_data['dVavg'] = extract_average_trace(ephys_data['t'],ephys_data['V'],
                                                                                      tp,window=[-1,15],interp_dt=1./resampling_frequency)
    # find the current amplitudes
    j = int((ephys_data['tbefore']+ephys_data['dur'])/2/ephys_data['dt'])
    idx, = np.where(ephys_data['I'][:,j] <= 0)
    # take only the current amplitudes <= 0 and the largest injected current
    ephys_data['I_amplitudes'],idx = np.unique(ephys_data['I'][idx,j] * 1e-3, return_index=True)
    ephys_data['I_amplitudes'] = np.append(ephys_data['I_amplitudes'], ephys_data['I'][-1,j]*1e-3)
    idx = np.append(idx, ephys_data['I'].shape[0]-1)
    ephys_data['V'] = ephys_data['V'][idx,:]
    ephys_data['I'] = ephys_data['I'][idx,:]
    n = len(ephys_data['I_amplitudes'])
    for lbl in 'th','p','end','ahp','adp','half':
        ephys_data['t'+lbl] = [ephys_data['t'+lbl][str(k)] for k in idx]
        ephys_data['V'+lbl] = [ephys_data['V'+lbl][str(k)] for k in idx]

    # initiate the Evolutionary Multiobjective Optimization
    global emoo
    emoo = Emoo(N=args.population_size, C=2*args.population_size, variables=variables, objectives=objectives)

    d_etam = (args.etam_end - args.etam_start) / args.generation_number
    d_etac = (args.etac_end - args.etac_start) / args.generation_number
    emoo.setup(eta_m_0=args.etam_start, eta_c_0=args.etac_start, p_m=args.pm, finishgen=0, d_eta_m=d_etam, d_eta_c=d_etac)
    # Parameters:
    # eta_m_0, eta_c_0: defines the initial strength of the mutation and crossover parameter (large values mean weak effect)
    # p_m: probability of mutation of a parameter (holds for each parameter independently)

    emoo.get_objectives_error = objectives_error
    emoo.checkpopulation = check_population
    
    emoo.evolution(generations=args.generation_number)

    if emoo.master_mode:
        CA3.utils.h5.save_h5_file(h5_filename, 'a', parameters={'etam_start': args.etam_start, 'etam_end': args.etam_end,
                                                                'etac_start': args.etac_start, 'etac_end': args.etac_end,
                                                                'p_m': args.pm},
                                  objectives=objectives, variables=variables, model_type=data['model_type'],
                                  h5_file=args.filename, ephys_file=args.data_file, ephys_data=ephys_data,
                                  neuron_pars=neuron_pars)

def display():
    parser = arg.ArgumentParser(description='Fit the parameters of a reduced morphology to electrophysiological data')
    parser.add_argument('filename', type=str, action='store', help='Path of the file containing the morphology')
    args = parser.parse_args(args=sys.argv[2:])
    if not os.path.isfile(args.filename):
        print('%s: no such file.' % args.filename)
        sys.exit(1)

    # load the data
    data = CA3.utils.h5.load_h5_file(args.filename)
    print(data['neuron_pars']['distal']['L'])

    # get the optimal parameters
    ngen = len(data['generations'])
    last = str(ngen-1)
    err = np.zeros((data['generations'][last].shape[0],len(data['objectives'])))
    norm_err = np.zeros((data['generations'][last].shape[0],len(data['objectives'])))
    for i,obj in enumerate(data['objectives']):
        err[:,i] = data['generations'][last][:,data['columns'][obj]]
        norm_err[:,i] = (err[:,i] - min(err[:,i])) / (max(err[:,i]) - min(err[:,i]))
    #best = np.argmin(np.sum(norm_err**2,axis=1))
    best = np.argmin(err[:,1])
    print('The best individual is #%d.' % best)

    # find the model type
    if data['model_type'].lower() == 'thorny':
        ctor = CA3.cells.ThornyNeuron
    elif data['model_type'].lower() == 'athorny':
        ctor = CA3.cells.AThornyNeuron
    elif data['model_type'].lower() == 'simplified':
        ctor = CA3.cells.SimplifiedNeuron
    else:
        print('Unknown model type [%s].' % data['model_type'])

    # build the parameters dictionary
    pars = copy.deepcopy(data['neuron_pars'])
    parameters = {}
    for v in data['variables']:
        parameters[v[0]] = data['generations'][last][best,data['columns'][v[0]]]
    for k,v in parameters.iteritems():
        if k == 'scaling':
            pars[k] = v
        elif k == 'El':
            for lbl in 'soma','proximal','distal','basal':
                pars[lbl][k] = v
        elif k in ('Cm','Rm'):
            pars['soma'][k] = v
        else:
            key = k.split('_')[0]
            value = '_'.join(k.split('_')[1:])
            try:
                pars[key][value] = v
            except:
                pars[key] = {value: v}
    pars['axon'] = copy.deepcopy(pars['soma'])
    try:
        pars['axon'].pop('L')
    except:
        pass
    try:
        pars['axon'].pop('diam')
    except:
        pass

    # construct the model
    neuron = ctor(pars, with_axon=True, with_active=True)

    # simulate the injection of currents into the model
    t,V = current_steps(neuron, data['ephys_data']['I_amplitudes'], data['ephys_data']['dt'][0],
                        data['ephys_data']['dur'], data['ephys_data']['tbefore'],
                        data['ephys_data']['tafter'], np.mean(data['ephys_data']['V'][:,0]))

    import pylab as p
    for i in range(data['ephys_data']['V'].shape[0]):
        p.plot(data['ephys_data']['t'],data['ephys_data']['V'][i,:],'k')
        p.plot(t,V[i,:],'r')
    p.xlabel('Time (ms)')
    p.ylabel('Membrane voltage (mV)')
    p.ylim([-90,60])
    p.show()

def help():
    print('This script optimizes the active properties of a reduced morphology.')
    print('')
    print('Author: Daniele Linaro - danielelinaro@gmail.com')
    print('  Date: January 2015')

if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','-H','--help'):
        help()
    elif sys.argv[1] == 'optimize':
        optimize()
    elif sys.argv[1] == 'display':
        display()
    else:
        print('Unknown working mode: enter "%s -h" for help.' % os.path.basename(sys.argv[0]))

