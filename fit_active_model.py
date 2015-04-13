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
from CA3.utils.graphics import *
from scipy.interpolate import interp1d,UnivariateSpline
from neuron import h
from emoo import Emoo
from emoo import mpi4py_loaded
if mpi4py_loaded:
    from mpi4py import MPI
    processor_name = MPI.Get_processor_name()
try:
    import matplotlib.pyplot as p
    set_rc_defaults()
except:
    pass

SAVE_DEBUG_INFO = False
model_type = 'simplified'
ReducedNeuron = CA3.cells.SimplifiedNeuron

# the list of objectives
objectives = []

# minimum set of variables to optimize
variables = [
    ['Cm', 0.6, 3.],                       # [uF/cm2] membrane capacitance (0.6,3)
    ['Rm', 5e3, 30e3],                     # [Ohm cm2] membrane resistance (10e3,30e3)
    ['El', -70., -50.],                    # [mV] reversal potential of leak conductance (-85,-50)
    ['nat_gbar_soma', 0., 500.],           # [pS/um2] (0,500)
    ['kdr_gbar_soma', 0., 500.]]           # [pS/um2] (0,100)

# the neuron parameters that have been obtained by the previous
# optimization of the passive properties
neuron_pars = {'soma': {'Ra': None, 'area': None},
               'proximal': {'Ra': None, 'area': None, 'L': None},
               'distal': {'Ra': None, 'area': None, 'L': None},
               'basal': {'Ra': None, 'area': None, 'L': None}}

# the electrophysiological data used in the optimization
ephys_data = None

resampling_frequency = 200. # [kHz]

# the threshold for spike detection
ap_threshold = -20. # [mV]

# default windows for computation of the spike shape error
spike_shape_error_window = [[-2.,0.],[0.,14.]]

# the (integer) power to which each ISI component is raised before being summed
isi_error_power = 2

def make_simplified_neuron(parameters):
    with_axon = False
    with_active = True
    pars = copy.deepcopy(neuron_pars)
    for k,v in parameters.iteritems():
        if k == 'scaling':
            pars[k] = v
        elif k == 'El':
            for lbl in 'soma','proximal','distal','basal':
                pars[lbl][k] = v
        elif k in ('Cm','Rm'):
            pars['soma'][k] = v
        elif k == 'vtraub':
            pars[k] = v
        else:
            key = k.split('_')[0]
            value = '_'.join(k.split('_')[1:])
            try:
                pars[key][value] = v
            except:
                pars[key] = {value: v}
    if 'nat_gbar_ais' in parameters and 'nat_gbar_hillock' in parameters:
        with_axon = True
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
    return ReducedNeuron(pars, with_axon, with_active)

def extract_average_trace(t,x,events,window,interp_dt=-1,token=None):
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
        # the next few lines are just for debugging purposes: return extract_average_trace(...) is sufficient
        Tint,Xint,dXint = extract_average_trace(Tint,Xint,
                                                [[Tint[i]] for i in np.argmax(Xint,axis=1)],
                                                [window[0]+offset, window[1]-offset],
                                                interp_dt=-1,token=token)
        logger('end','extract_average_trace',token)
        return Tint,Xint,dXint
    logger('end','extract_average_trace',token)
    return T+window[0],np.mean(X,axis=0),np.mean(dX,axis=0)

def logger(log_type, msg, token):
    if token is None or not log_type.lower() in ('start','end'):
        return
    if log_type.lower() == 'start':
        message = '%d STARTED %s @ %s.' % (token,msg,timestamp())
    else:
        message = '%d FINISHED %s @ %s.' % (token,msg,timestamp())
    print(message)
    
def current_steps(neuron, amplitudes, dt=0.05, dur=500, tbefore=100, tafter=100, V0=-70, token=None):
    if not token is None:
        logger('start', 'current_steps', token)
        if SAVE_DEBUG_INFO:
            opts = {'%s' % token: {'parameters': neuron.parameters, 'has_active':neuron.has_active,
                                   'has_axon': neuron.has_axon, 'neuron_type':neuron.__class__.__name__}}
            CA3.utils.h5.save_h5_file(h5_filename, 'a', **opts)
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

def spike_shape_error(t,V,tp,Vth,window=spike_shape_error_window,token=None):
    flat_mean = lambda x: np.mean([z for y in x for z in y])
    logger('start','spike_shape_error',token)
    if np.isscalar(window[0]):
        window = [window]
    m = min([w[0] for w in window])
    M = max([w[1] for w in window])
    tavg,Vavg,dVavg = extract_average_trace(t,V,tp,[m,M],interp_dt=1./resampling_frequency,token=token)
    # the voltage threshold for the ephys data
    ref_thresh = flat_mean(ephys_data['Vth'])
    # the voltage threshold for the simulated data
    thresh = flat_mean(Vth)
    err = []
    for w in window:
        idx, = np.where((ephys_data['tavg']>=w[0]) & (ephys_data['tavg']<=w[1]))
        jdx, = np.where((tavg>=w[0]) & (tavg<=w[1]))
        err.append(
            np.sum(
                ( (Vavg[jdx]-thresh) - (ephys_data['Vavg'][idx]-ref_thresh) )**2
                )
            + 0.1*np.sum(
                (dVavg[jdx] - ephys_data['dVavg'][idx])**2
                )
            )
    logger('end','spike_shape_error',token)
    return err

def isi_error(tp):
    err = None
    for i in range(len(tp)):
        if len(tp[i]) == 0 or len(ephys_data['tp'][i]) == 0:
            continue
        for j in range(len(tp[i])):
            if err is None:
                err = 0
            if j < len(ephys_data['tp'][i]):
                err += abs(ephys_data['tp'][i][j] - tp[i][j])**isi_error_power
            else:
                err += abs(ephys_data['tp'][i][-1] - tp[i][j])**isi_error_power
        if j < len(ephys_data['tp'][i]):
            for k in range(j+1,len(ephys_data['tp'][i])):
                err += abs(ephys_data['tp'][i][k] - tp[i][-1])**isi_error_power
    if err is None:
        return 1e10
    return err

def spike_rate_error(tp, dur):
    import pdb
    pdb.set_trace()
    rate_ref = np.array(map(len, ephys_data['tp'])) / (dur*1e-3)
    return 0

def accommodation_index_error(tp):
    return 0

def latency_error(tp, t0):
    return 0

def AP_overshoot_error(Vp):
    return 0

def AHP_depth_error(Vahp):
    return 0

def AP_width_error(width):
    return 0

def check_prerequisites(t,V,ton,toff,tp,Vp,width=None,token=None):
    retval = True
    logger('start','check_prerequisites',token)
    n = V.shape[0]
    idx, = np.where((t>toff-200) & (t<toff))
    for i in range(n):
        m = np.mean(V[i,idx])
        s = np.std(V[i,idx])
        # number of spikes in the reference trace (ephys data)
        ref_nspikes = len(np.where((np.array(ephys_data['tp'][i])>ton) & (np.array(ephys_data['tp'][i])<=toff))[0])
        # number of spikes in the simulated trace, after stimulation onset: I consider also
        # the time after the stimulation offset because there should be no spikes there, i.e.
        # the neuron should go back to equilibrium. This also allows removing those situations
        # in which the neuron is tonically spiking when no current is injected.
        nspikes = len(np.where(np.array(tp[i])>ton)[0])
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
        # too many spikes
        elif nspikes > 3*ref_nspikes:
            print('%d check_prerequistes: too many spikes (%d instead of %d).' % (token,nspikes,ref_nspikes))
            retval = False
            break
        # too few spikes
        #elif nspikes < 0.5*ref_nspikes:
        #    print('%d check_prerequistes: too few spikes (%d instead of %d).' % (token,nspikes,ref_nspikes))
        #    retval = False
        #    break
        # spike block?
        elif ref_nspikes > 0 and nspikes > 0 and m > -40 and s < 3:
            print('%d check_prerequistes: mean(voltage) > -40 and std(voltage) < 3.' % token)
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
    tp,Vp = extractAPPeak(t, V, threshold=ap_threshold, min_distance=1)
    logger('end', 'extractAPPeak', token)

    measures = {}
    for obj in objectives:
        measures[obj] = 1e20

    if check_prerequisites(t,V,ephys_data['tbefore'],ephys_data['tbefore']+ephys_data['dur'],tp,Vp,token=token):
        logger('start', 'extractAPThreshold', token)
        tth,Vth = extractAPThreshold(t, V, threshold=ap_threshold, tpeak=tp)
        logger('end', 'extractAPThreshold', token)
        logger('start', 'extractAPHalfWidth', token)
        Vhalf,width,interval = extractAPHalfWidth(t, V, threshold=ap_threshold, tpeak=tp, Vpeak=Vp, tthresh=tth, Vthresh=Vth, interp=False)
        logger('end', 'extractAPHalfWidth', token)
        if check_prerequisites(t,V,ephys_data['tbefore'],ephys_data['tbefore']+ephys_data['dur'],tp,Vp,width,token):
            if SAVE_DEBUG_INFO:
                opts = {'%s_spikes' % token: {'tp': tp, 'Vp': Vp, 'tth': tth, 'Vth': Vth}}
                CA3.utils.h5.save_h5_file(h5_filename, 'a', **opts)
            if 'hyperpolarizing_current_steps' in objectives:
                measures['hyperpolarizing_current_steps'] = hyperpolarizing_current_steps_error(t,V,ephys_data['I_amplitudes'],ephys_data['V'])
            if 'isi' in objectives:
                measures['isi'] = isi_error(tp)
            if 'spike_onset' in objectives or 'spike_offset' in objectives:
                err = spike_shape_error(t,V,tp,Vth,spike_shape_error_window,token)
                if 'spike_onset' in objectives:
                    measures['spike_onset'] = err[0]
                if 'spike_offset' in objectives:
                    measures['spike_offset'] = err[1]
            if 'spike_rate' in objectives:
                measures['spike_rate'] = spike_rate_error(tp, ephys_data['dur'])
            if 'accommodation_index' in objectives:
                measures['accommodation_index'] = accommodation_index_error(tp)
            if 'latency' in objectives:
                measures['latency'] = latency_error(tp, ephys_data['tbefore'])
            if 'ap_overshoot' in objectives:
                measures['ap_overshoot'] = AP_overshoot_error(Vp)
            if 'ahp_depth' in objectives:
                logger('start', 'extractAPAHP', token)
                tahp,Vahp = extractAPAHP(t, V, max_ahp_dur=10, threshold=ap_threshold, tpeak=tp, tthresh=tth)
                logger('end', 'extractAPAHP', token)
                measures['ahp'] = AHP_depth_error(Vahp)
            if 'ap_width' in objectives:
                measures['ap_width'] = AP_width_error(width)

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
        logger('start','check_population',5061983)
        if gen == 0:
            CA3.utils.h5.save_h5_file(h5_filename, 'w', columns=columns)
        CA3.utils.h5.save_h5_file(h5_filename,'a',generations={('%d'%gen): population})
        logger('stop','check_population',5061983)

def extract_spike_rate(spike_times, stim_dur):
    if np.isscalar(spike_times[0]):
        f = len(spike_times) / (stim_dur*1e-3)
    else:
        f = np.array(map(len, spike_times)) / (stim_dur*1e-3)
    return np.mean(f),np.std(f)

def extract_accommodation_index(spike_times):
    # number of spikes
    n_spikes = np.array(map(lambda x: float(len(x)), spike_times))
    # number of ISIs
    n_isi = n_spikes-1
    # number of ISIs that should be discarded
    k = np.min(np.c_[np.round(n_isi/5), 4+np.zeros(len(n_isi))], axis=1)
    A = []
    for i in np.where((n_isi-k >= 2) & (k>=1))[0]:
        print spike_times[i],k[i]
        isi = np.diff(spike_times[i][k[i]:])
        print isi
        A.append(0)
        for j in range(1,len(isi)):
            A[-1] += (isi[j]-isi[j-1]) / (isi[j]+isi[j-1])
        A[-1] /= (n_spikes[i] - k[i] - 1)
    return np.mean(A),np.std(A)

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
    parser.add_argument('--add', action='append', help='List of ion channels to add to the optimization. ' +
                        'Available options are axon, nat-dend, kdr-dend, nap, km, kahp, kd, kap, ih and ih-dend')
    parser.add_argument('--optimize', action='append', help='List of objectives to be optimized. ' +
                        'Available options are hyperpolarizing_current_steps, spike_onset, spike_offset, isi, ' +
                        'spike_rate, accommodation_index, latency, ap_overshoot, ahp_depth and ap_width')
    parser.add_argument('--single-compartment', action='store_true', help='Use a single-compartment neuron model')
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

    if args.optimize is None:
        print('You must specify at least one function to optimize.')
        sys.exit(3)
    global objectives
    if args.optimize[0] == 'all':
        objectives = ['hyperpolarizing_current_steps','spike_onset','spike_offset','isi', # used in Bahl et al., 2012
                      'spike_rate','accommodation_index', 'latency', 'ap_overshoot', 'ahp_depth', 'ap_width'] # used in Druckmann et al., 2007
    elif args.optimize[0] == 'bahl':
        objectives = ['hyperpolarizing_current_steps','spike_onset','spike_offset','isi']
    elif args.optimize[0] == 'druckmann':
        objectives = ['spike_rate','accommodation_index', 'latency', 'ap_overshoot', 'ahp_depth', 'ap_width']
    else:
        for cost in args.optimize:
            objectives.append(cost)
    if 'spike_rate' in objectives or 'accommodation_index' in objectives or \
            'latency' in objectives or 'ap_overshoot' in objectives or \
            'ahp_depth' in objectives or 'ap_width' in objectives:
        print('Not fully implemented yet.')
        sys.exit(0)
            
    with_axon = False
    if not args.add is None:
        if args.add[0] == 'all':
            args.add = ['nap','km','kahp','kd','kap','ih']
        if not args.single_compartment:
            args.add.append('axon')
            args.add.append('nat-dend')
            args.add.append('kdr-dend')
            args.add.append('ih-dend')
        for opt in args.add:
            if opt == 'axon':
                variables.append(['nat_gbar_hillock', 0., 20000.])      # [pS/um2] (0,20000)
                variables.append(['nat_gbar_ais', 0., 20000.])          # [pS/um2] (0,20000)
                variables.append(['nat_gbar_distal', 0., 100.])         # [pS/um2] (0,100)
            elif opt == 'nat-dend':
                variables.append(['nat_lambda', 1., 100.])              # [um] (1,500)
            elif opt == 'kdr-dend':
                variables.append(['kdr_gbar_distal', 0., 10.])          # [pS/um2] (0,10)
                variables.append(['kdr_lambda', 1., 100.])              # [um] (0,100)
            elif opt == 'nap':
                variables.append(['nap_gbar', 0., 5.])                  # [pS/um2] in the paper, 0 < gbar < 4.1
            elif opt == 'km':
                variables.append(['km_gbar', 0., 2.])                   # [pS/um2]
            elif opt == 'kahp':
                variables.append(['kahp_gbar', 0., 500.])               # [pS/um2]
            elif opt == 'kd':
                variables.append(['kd_gbar', 0., 0.01])                 # [pS/um2]
            elif opt == 'kap':
                variables.append(['kap_gbar', 0., 100.])                # [pS/um2]
            elif opt == 'ih':
                variables.append(['ih_gbar_soma', 0., 0.1])             # [pS/um2]
            elif opt == 'ih-dend':
                variables.append(['ih_dend_scaling', 0., 10.])          # [1]
                variables.append(['ih_half_dist', 0., 500.])            # [um]
                variables.append(['ih_lambda', 1., 500.])               # [um]

    if not args.single_compartment:
        variables.append(['scaling', 0.3, 2.])   # [1] scaling of dendritic capacitance and membrane resistance (0.5,2)

    # load the data relative to the optimization of the passive properties
    data = CA3.utils.h5.load_h5_file(args.filename)

    # output filename
    global h5_filename
    if args.out_file is None:
        h5_filename = CA3.utils.h5.make_output_filename(os.path.basename(data['swc_filename']).rstrip('.swc'), '.h5', with_rand=True)
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
    global model_type
    global ReducedNeuron
    if args.single_compartment:
        model_type = 'single_compartment'
        ReducedNeuron = CA3.cells.SingleCompartmentNeuron
    elif data['model_type'].lower() == 'thorny':
        model_type = 'thorny'
        ReducedNeuron = CA3.cells.ThornyNeuron
    elif data['model_type'].lower() == 'athorny':
        model_type = 'athorny'
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
                                                                                      tp,window=[spike_shape_error_window[0][0],
                                                                                                 spike_shape_error_window[1][1]],
                                                                                      interp_dt=1./resampling_frequency)
    # find the firing rate corresponding to the maximum value of the injected current
    idx, = np.where(np.max(ephys_data['I'],axis=1) == np.max(ephys_data['I'][-1,:]))
    ephys_data['spike_rate'] = {}
    ephys_data['spike_rate']['mean'],ephys_data['spike_rate']['std'] = extract_spike_rate(tp[idx], ephys_data['dur'])
    # find the accommodation index
    ephys_data['accommodation_index'] = {}
    ephys_data['accommodation_index']['mean'],ephys_data['accommodation_index']['std'] = extract_accommodation_index(tp)
    # find the current amplitudes
    j = int((ephys_data['tbefore']+ephys_data['dur'])/2/ephys_data['dt'])
    idx, = np.where(ephys_data['I'][:,j] <= 0)
    # take only the current amplitudes <= 0 and the largest injected current
    ephys_data['I_amplitudes'],idx = np.unique(ephys_data['I'][idx,j] * 1e-3, return_index=True)
    if len(objectives) > 1 or objectives[0] != 'hyperpolarizing_current_steps':
        # ... only if we're optimizing also spiking properties
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
                                  objectives=objectives, variables=variables, model_type=model_type,
                                  h5_file=args.filename, ephys_file=args.data_file, ephys_data=ephys_data,
                                  neuron_pars=neuron_pars, ap_threshold=ap_threshold, spike_shape_error_window=spike_shape_error_window)

def display_hyperpolarizing_current_steps(t, V, ephys_data):
    p.figure(figsize=(5,3))
    p.axes([0.15,0.2,0.75,0.7])
    for i in range(V.shape[0]):
        if np.max(ephys_data['V'][i,:]) < -20:
            p.plot(ephys_data['t'][::10], ephys_data['V'][i,::10],'k')
            p.plot(t[::10], V[i,::10], 'r')
    p.yticks(np.floor(np.arange(p.ylim()[0],p.ylim()[1]+1,10)))
    p.xlabel('Time (ms)')
    p.ylabel('Membrane potential (mV)')
    p.title('Hyperpolarizing current steps')
    remove_border()

def display_spike(t, V, ephys_data, wndw, title):
    tp,Vp = extractAPPeak(t, V, threshold=ap_threshold, min_distance=1)
    tth,Vth = extractAPThreshold(t, V, threshold=ap_threshold, tpeak=tp)
    tavg,Vavg,dVavg = extract_average_trace(t, V, tp, wndw, 1./resampling_frequency)
    p.figure(figsize=(7,3))
    p.axes([0.1,0.2,0.2,0.7])
    p.plot(ephys_data['t'],ephys_data['V'][-1],'k')
    p.plot(t, V[-1,:], 'r')
    p.xlabel('Time (ms)')
    p.ylabel('Vm (mV)')
    p.axis([100,700,-80,60])
    p.xticks([100,300,500,700])
    p.yticks([-80,-40,0,40])
    remove_border()

    p.axes([0.4,0.2,0.225,0.7])
    p.plot(ephys_data['tavg'],ephys_data['Vavg']-np.mean([z for y in ephys_data['Vth'].values() for z in y]),'k')
    p.plot(tavg, Vavg-np.mean([z for y in Vth for z in y]), 'r')
    p.plot(wndw,[0,0],'--',color=[.6,.6,.6])
    p.xlim(wndw)
    p.ylim([-20,100])
    p.xticks(np.linspace(wndw[0],wndw[1],3))
    p.yticks([-20,20,60,100])
    p.xlabel('Time (ms)')
    p.ylabel('Vm - AP threshold (mV)')
    p.title(title)
    remove_border()

    p.axes([0.75,0.2,0.2,0.7])
    p.plot(ephys_data['tavg'],ephys_data['dVavg'],'k')
    p.plot(tavg, dVavg, 'r')
    p.xlim(wndw)
    p.ylim([-200,500])
    p.xticks(np.linspace(wndw[0],wndw[1],3))
    p.yticks([-200,0,200,400])
    p.xlabel('Time (ms)')
    p.ylabel('dVm/dt (mV/ms)')
    remove_border()

def display_spike_onset(t, V, ephys_data):
    display_spike(t, V, ephys_data, spike_shape_error_window[0], 'Spike onset')

def display_spike_offset(t, V, ephys_data):
    display_spike(t, V, ephys_data, spike_shape_error_window[1], 'Spike offset')

def display_isi(t, V, ephys_data):
    tp,Vp = extractAPPeak(t, V, threshold=ap_threshold, min_distance=1)
    p.figure(figsize=(5,3))
    p.axes([0.15,0.2,0.75,0.7])
    p.plot(ephys_data['t'],ephys_data['V'][-1],'k')
    p.plot(t, V[-1,:], 'r')
    p.xlabel('Time (ms)')
    p.ylabel('Membrane potential (mV)')
    p.axis([100,700,-80,60])
    p.xticks([100,300,500,700])
    p.yticks([-80,-40,0,40])
    remove_border()
    
def display():
    parser = arg.ArgumentParser(description='Fit the parameters of a reduced morphology to electrophysiological data')
    parser.add_argument('filename', type=str, action='store', help='Path of the file containing the morphology')
    args = parser.parse_args(args=sys.argv[2:])
    if not os.path.isfile(args.filename):
        print('%s: no such file.' % args.filename)
        sys.exit(1)

    # temporary files will be stored here
    base_dir = '/tmp'

    # load the data
    data = CA3.utils.h5.load_h5_file(args.filename)

    # find the model type
    if data['model_type'].lower() == 'single_compartment' or \
            os.path.basename(args.filename) == 'DH070313-.Edit.scaled_20150321-230759.h5':
        ctor = CA3.cells.SingleCompartmentNeuron
    elif data['model_type'].lower() == 'thorny':
        ctor = CA3.cells.ThornyNeuron
    elif data['model_type'].lower() == 'athorny':
        ctor = CA3.cells.AThornyNeuron
    elif data['model_type'].lower() == 'simplified':
        ctor = CA3.cells.SimplifiedNeuron
    else:
        print('Unknown model type [%s].' % data['model_type'])

    # find out whether the model contained active conductances and/or an axon
    if 'nat_gbar_soma' in data['variables']:
        with_active = True
    else:
        with_active = False
    if 'nat_gbar_ais' in data['variables']:
        with_axon = True
    else:
        with_axon = False

    # number of generations
    ngen = len(data['generations'])
    last = str(ngen-1)
    # number of objectives
    nobj = len(data['objectives'])
    # number of individuals
    ngenes = data['generations'][last].shape[0]
 
    # find the optimal parameters for each objective at the last generation
    best_individuals = {}
    opt_pars = {}
    for obj in data['objectives']:
        best_individuals[obj] = np.argmin(data['generations'][last][:,data['columns'][obj]])
        opt_pars[obj] = {}
        for v in data['variables']:
            opt_pars[obj][v[0]] = data['generations'][last][best_individuals[obj],data['columns'][v[0]]]
        print('The best individual for objective "%s" is #%d with error = %g.' %
              (obj,best_individuals[obj],data['generations'][last][best_individuals[obj],data['columns'][obj]]))

    # plot the evolution of the optimization variables
    nbins = 80
    nvar = len(data['variables'])
    c = 3
    r = np.ceil(float(nvar)/c)
    p.figure(figsize=(c*3,r*3))
    for i,v in enumerate(data['variables']):
        var = np.zeros((nbins,ngen))
        for j in range(ngen):
            var[:,j],edges = np.histogram(data['generations']['%d'%j][:,data['columns'][v[0]]], bins=nbins, range=[float(v[1]),float(v[2])])
        tmp = data['generations']['%d'%(ngen-1)][:,data['columns'][v[0]]]
        ax = make_axes(r,c,i+1)
        opt = {'origin': 'lower', 'cmap': p.get_cmap('Greys'), 'interpolation': 'nearest'}
        if edges[-1] > 1000:
            coeff = 1e-3
        elif edges[-1] < 1:
            coeff = 1e3
        else:
            coeff = 1
        p.imshow(var, extent=[1,ngen,edges[0]*coeff,edges[-1]*coeff], aspect=ngen/(edges[-1]-edges[0])/coeff, **opt)
        p.plot(p.xlim(), [float(v[1])*coeff,float(v[1])*coeff], 'r--')
        p.plot(p.xlim(), [float(v[2])*coeff,float(v[2])*coeff], 'r--')
        p.ylim([(float(v[1])-0.05*(float(v[2])-float(v[1])))*coeff,(float(v[2])+0.05*(float(v[2])-float(v[1])))*coeff])
        p.xlabel('Generation #')
        if coeff != 1:
            p.ylabel(v[0] + (' (%.0e)' % coeff))
        else:
            p.ylabel(v[0])
        p.xticks(np.round(np.linspace(0,ngen,6)))
        p.yticks(np.round(np.linspace(edges[0],edges[-1],5)*coeff))
        remove_border()
    p.savefig(base_dir + '/variables.pdf')

    # plot the trade-offs between all possible error pairs at the last generation
    colors = [[1,0.5,0],[0.5,0,1],[0,1,0.5],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1]]
    err = np.zeros((nobj,ngen,ngenes))
    for i,obj in enumerate(data['objectives']):
        for j in range(ngen):
            err[i,j,:] = data['generations']['%d'%j][:,data['columns'][obj]]
            idx, = np.where(err[i,j,:] < 1e10)
            err[i,j,idx] /= np.std(err[i,j,idx])
    tot = nobj*(nobj-1)/2
    if tot > 0:
        c = 3
        r = np.ceil(tot/c)
        subp = 1
        p.figure(figsize=(c*3,r*3))
        for i in range(nobj):
            for j in range(i+1,nobj):
                #idx, = np.where((err[i,-1,:] < 1e10) & (err[j,-1,:] < 1e10))
                idx, = np.where((err[i,-1,:] <= 3) & (err[j,-1,:] <= 3))
                ax = make_axes(r,c,subp)
                p.plot(err[i,-1,idx],err[j,-1,idx],'k.',markersize=2)
                for obj,col in zip(data['objectives'],colors[:len(data['objectives'])]):
                    x = err[i,-1,best_individuals[obj]]
                    y = err[j,-1,best_individuals[obj]]
                    if x <= 3 and y <= 3:
                        p.plot(x,y,'o',color=col,markersize=6)
                p.xlabel(data['objectives'][i].replace('_',' '))
                p.ylabel(data['objectives'][j].replace('_',' '))
                #p.xticks([0,p.xlim()[1]])
                #p.yticks([0,p.ylim()[1]])
                p.axis([0,3,0,3])
                p.xticks([0,1,2,3])
                p.yticks([0,1,2,3])
                remove_border()
                subp += 1
        p.savefig(base_dir + '/errors.pdf')
        got_errors = True
    else:
        got_errors = False

    # plot the performance of the best individual for each objective
    for obj in data['objectives']:
        # build the parameters dictionary
        pars = copy.deepcopy(data['neuron_pars'])
        parameters = {}
        for v in data['variables']:
            parameters[v[0]] = data['generations'][last][best_individuals[obj],data['columns'][v[0]]]
        for k,v in parameters.iteritems():
            if k == 'scaling':
                pars[k] = v
            elif k == 'El':
                for lbl in 'soma','proximal','distal','basal':
                    pars[lbl][k] = v
            elif k in ('Cm','Rm'):
                pars['soma'][k] = v
            elif k == 'vtraub':
                pars[k] = v
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
        neuron = ctor(pars, with_axon, with_active)

        # simulate the injection of currents into the model
        t,V = current_steps(neuron, data['ephys_data']['I_amplitudes'], data['ephys_data']['dt'][0],
                            data['ephys_data']['dur'], data['ephys_data']['tbefore'],
                            data['ephys_data']['tafter'], np.mean(data['ephys_data']['V'][:,0]))
        tp,Vp = CA3.utils.extractAPPeak(t,V,threshold=ap_threshold,min_distance=1)
        tth,Vth = CA3.utils.extractAPThreshold(t,V,threshold=ap_threshold,tpeak=tp)
        global ephys_data
        if obj == 'hyperpolarizing_current_steps':
            err = hyperpolarizing_current_steps_error(t,V,data['ephys_data']['I_amplitudes'],data['ephys_data']['V'])
        elif obj == 'spike_onset':
            ephys_data = {'Vth': [data['ephys_data']['Vth']['%04d'%i] for i in range(len(data['ephys_data']['Vth']))]}
            for k in 'tavg','Vavg','dVavg':
                ephys_data[k] = data['ephys_data'][k]
            err = spike_shape_error(t,V,tp,Vth,window=spike_shape_error_window)
            err = err[0]
        elif obj == 'spike_offset':
            ephys_data = {'Vth': [data['ephys_data']['Vth']['%04d'%i] for i in range(len(data['ephys_data']['Vth']))]}
            for k in 'tavg','Vavg','dVavg':
                ephys_data[k] = data['ephys_data'][k]
            err = spike_shape_error(t,V,tp,Vth,window=spike_shape_error_window)
            err = err[1]
        elif obj == 'isi':
            tp,Vp = extractAPPeak(t, V, threshold=ap_threshold, min_distance=1)
            ephys_data = {'tp': [data['ephys_data']['tp']['%04d'%i] for i in range(len(data['ephys_data']['tp']))]}
            err = isi_error(tp)
        print('%s%s error: %g.' % (obj[0].upper(),obj[1:].replace('_',' '),err))

        globals()['display_' + obj](t,V,data['ephys_data'])
        p.savefig(base_dir + '/' + obj + '.pdf')

    # LaTeX generation
    tex_file = os.path.basename(args.filename)[:-3] + '.tex'
    with open(base_dir + '/' + tex_file, 'w') as fid:
        fid.write('\\documentclass[11pt]{scrartcl}\n')
        fid.write('\\usepackage{graphicx}\n')
        fid.write('\\usepackage[squaren]{SIunits}\n')
        fid.write('\\usepackage[a4paper,margin=0.5in]{geometry}')
        fid.write('\\begin{document}\n')
        fid.write('\\section*{Active model fit summary}\n')
        fid.write('\n\\noindent ')
        fid.write('Filename: %s.\n' % os.path.basename(args.filename).replace('_','\_'))
        fid.write('\n\\noindent ')
        fid.write('Passive properties file: %s.\n' % os.path.basename(data['h5_file']).replace('_','\_'))
        fid.write('\n\\noindent ')
        if args.filename != 'DH070313-.Edit.scaled_20150321-230759.h5':
            fid.write('Model type: %s.\n' % data['model_type'].replace('_',' '))
        else:
            fid.write('Model type: single compartment.\n')
        fid.write('\n\\noindent ')
        fid.write('Objectives:')
        for obj in data['objectives']:
            fid.write(' ' + obj.replace('_',' '))
            if obj != data['objectives'][-1]:
                fid.write(',')
        fid.write('.\n')
        fid.write('\n\\noindent ')
        fid.write('Optimization variables:')
        for var in data['variables']:
            fid.write(' ' + var[0].replace('_','\_'))
            if var[0] != data['variables'][-1][0]:
                fid.write(',')
        fid.write('.\n')
        fid.write('\n\\noindent ')
        fid.write('Number of generations: %s.\n' % ngen)
        fid.write('\n\\noindent ')
        fid.write('Number of organisms: %s.\n' % ngenes)
        fid.write('\n\\noindent ')
        fid.write('Optimization parameters: $\\eta_c^0=%g$, $\\eta_c^{\mathrm{end}}=%g$, $\\eta_m^0=%g$, $\\eta_m^{\mathrm{end}}=%g$, $p_m=%g$.\n' % 
                  (data['parameters']['etac_start'],data['parameters']['etac_end'],
                   data['parameters']['etam_start'],data['parameters']['etam_end'],data['parameters']['p_m']))
        #fid.write('\\subsection*{Parameters of one good solution}')
        #fid.write('\\begin{table}[h!!]\n')
        #fid.write('\\centering\n')
        #fid.write('\\begin{tabular}{|l|ccc|}\n')
        #fid.write('\\hline\n')
        #fid.write('\\textbf{Functional section} & \\textbf{Length} & \\textbf{Diameter} & \\textbf{Axial resistance} \\\\\n')
        #fid.write('\\hline\n')
        #fid.write('Soma & $%g\\,\\micro\\meter$ & $%g\\,\\micro\\meter$ & $%g\\,\\ohm\\cdot\\centi\\meter$ \\\\\n' %
        #          (L_soma,L_soma,Ra_soma))
        #n = ReducedNeuron.n_basal_sections()
        #fid.write('Basal dendrites (%d) & $%g\\,\\micro\\meter$ & $%g\\,\\micro\\meter$ & $%g\\,\\ohm\\cdot\\centi\\meter$ \\\\\n' %
        #          (n,round(L_basal),round(diam_basal/n),round(Ra_basal)))
        #n = ReducedNeuron.n_proximal_sections()
        #fid.write('Proximal apical dendrites (%d) & $%g\\,\\micro\\meter$ & $%g\\,\\micro\\meter$ & $%g\\,\\ohm\\cdot\\centi\\meter$ \\\\\n' %
        #          (n,round(L_proximal),round(diam_proximal/n),round(Ra_proximal)))
        #n = ReducedNeuron.n_distal_sections()
        #fid.write('Distal apical dendrites (%d) & $%g\\,\\micro\\meter$ & $%g\\,\\micro\\meter$ & $%g\\,\\ohm\\cdot\\centi\\meter$ \\\\\n' %
        #          (n,round(L_distal),round(diam_distal/n),round(Ra_distal)))
        #fid.write('\\hline\n')
        #fid.write('\\end{tabular}\n')
        #fid.write('\\caption{Reduced model parameters.}\n')
        #fid.write('\\end{table}\n')
        #fid.write('\n')
        fid.write('\\subsection*{Variables}')
        fid.write('\\begin{figure}[htb]\n')
        fid.write('\\centering\n')
        fid.write('\\includegraphics[width=\\textwidth]{%s/variables.pdf}\n' % base_dir)
        fid.write('\\caption{Evolution of optimization variables with generation. Dark areas indicate clustering of individuals. ')
        fid.write('The red dashed lines indicate the optimization bounds.}\n')
        fid.write('\\end{figure}\n')
        fid.write('\n')
        if got_errors:
            fid.write('\\subsection*{Errors}')
            fid.write('\\begin{figure}[htb]\n')
            fid.write('\\centering\n')
            fid.write('\\includegraphics[width=\\textwidth]{%s/errors.pdf}\n' % base_dir)
            fid.write('\\caption{Trade-offs between objectives at the last generation in units of standard deviation: ')
            fid.write('each black dot indicates a solution. Each colored circle indicates the best ')
            fid.write('solution for a particular objective.}\n')
            fid.write('\\end{figure}\n')
            fid.write('\n')
        fid.write('\\subsection*{Cost functions}')
        for obj in data['objectives']:
            fid.write('\\begin{figure}[htb]\n')
            fid.write('\\centering\n')
            fid.write('\\includegraphics{%s/%s.pdf}\n' % (base_dir,obj))
            fid.write('\\caption{%s%s error. Optimal parameters: %s.}\n' % (obj[0].upper(), obj[1:].replace('_',' '),
                                 ', '.join(['='.join([k.replace('_','\_'),'%.2f'%v]) for k,v in opt_pars[obj].iteritems()])))
            fid.write('\\end{figure}\n')
            fid.write('\n')
        fid.write('\\end{document}\n')
    os.system('pdflatex ' + base_dir + '/' + tex_file)
    os.remove(base_dir + '/' + tex_file)
    os.remove(tex_file[:-4] + '.aux')
    os.remove(tex_file[:-4] + '.log')

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

