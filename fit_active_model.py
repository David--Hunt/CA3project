#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse as arg
import itertools as it
import time
import copy
import ConfigParser
import CA3
from CA3.utils import *
from CA3.utils.graphics import *
from scipy.interpolate import interp1d,UnivariateSpline
from scipy.optimize import curve_fit
from neuron import h
h.celsius = 36

SAVE_DEBUG_INFO = False

model_type = 'simplified'
ReducedNeuron = CA3.cells.SimplifiedNeuron

# default vtraub shift
vtraub_offset = 10.

# the optimization mode
optimization_mode = ''

# the list of objectives
objectives = []

# the list of features
features = {}

# the list of variables to optimize
variables = []

# this dictionary will contain the strings that describe the types of dendritic decays
# used for some active currents (transient sodium, rectifier potassium and Ih)
dendritic_modes = {}

# the neuron parameters that have been obtained by the previous
# optimization of the passive properties
neuron_pars = {'soma': {'Ra': None, 'area': None},
               'proximal': {'Ra': None, 'area': None, 'L': None},
               'distal': {'Ra': None, 'area': None, 'L': None},
               'basal': {'Ra': None, 'area': None, 'L': None}}

# the electrophysiological data used in the optimization
ephys_data = None

# the frequency at which data is resampled for computing the Vm errors
resampling_frequency = 200. # [kHz]

# the threshold for spike detection
ap_threshold = 0. # [mV]

# default windows for computation of the spike shape error
spike_shape_error_window = [[-2.,0.],[0.,14.]]

# the (integer) power to which each ISI component is raised before being summed
isi_error_power = 2

def make_simplified_neuron(parameters):
    with_axon = False
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
            if key in dendritic_modes:
                pars[key]['dend_mode'] = dendritic_modes[key]
    if 'nat' in pars and not 'vtraub_offset_soma' in pars['nat']:
        pars['nat']['vtraub_offset_soma'] = vtraub_offset
    if any(['ais' in k for k in parameters.keys()]) and any(['hillock' in k for k in parameters.keys()]):
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
        try:
            pars['axon'].pop('area')
        except:
            pass
        for key in 'scaling_hillock','scaling_ais':
            if key in pars['nat']:
                factor = pars['nat'].pop(key)
                pars['nat']['gbar_'+key[8:]] = factor * parameters['nat_gbar_soma']
        if not 'vtraub_offset_ais' in pars['nat']:
            pars['nat']['vtraub_offset_ais'] = pars['nat']['vtraub_offset_soma']
        if not 'vtraub_offset_hillock' in pars['nat']:
            pars['nat']['vtraub_offset_hillock'] = pars['nat']['vtraub_offset_soma']
    return ReducedNeuron(pars, with_axon)

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
    # minimal required spike distance
    min_spike_distance = 15. # [ms]
    for i in range(x.shape[0]):
        for j,e in enumerate(events[i]):
            if (j > 0 and e-events[i][j-1] < min_spike_distance) or \
                    (j < len(events[i])-1 and events[i][j+1]-e < min_spike_distance):
                # ignore spikes in a burst for the computation of the average spike shape
                continue
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
    
def current_ramp(neuron, Istop, Istart=0, dt=0.05, dur=500, tbefore=100, tafter=100, V0=-70, token=None):
    if not token is None:
        logger('start', 'current_ramp', token)
    T = np.arange(0, dur+tbefore+tafter+dt/2, dt)
    V = np.zeros((1,len(T)))
    I = np.zeros((1,len(T)))
    idx, = np.where((T>tbefore) & (T<tbefore+dur))
    I[0,idx] = Istart + (Istop-Istart) * (T[idx] - tbefore) / dur
    vec = h.Vector(I[0,:])
    stim = h.IClamp(neuron.soma[0](0.5))
    stim.dur = 1e9
    vec.play(stim._ref_amp,dt)
    rec = {'t': h.Vector(), 'v': h.Vector()}
    rec['t'].record(h._ref_t)
    rec['v'].record(neuron.soma[0](0.5)._ref_v)
    CA3.utils.run(tend=dur+tbefore+tafter, V0=V0, temperature=36)
    f = interp1d(rec['t'],rec['v'])
    V[0,:] = f(T)
    if not token is None:
        logger('stop', 'current_ramp', token)
    return T,V,I

def current_steps(neuron, amplitudes, dt=0.05, dur=500, tbefore=100, tafter=100, V0=-70, token=None):
    if not token is None:
        logger('start', 'current_steps', token)
        if SAVE_DEBUG_INFO:
            opts = {'%s' % token: {'parameters': neuron.parameters, 'has_axon': neuron.has_axon,
                                   'neuron_type':neuron.__class__.__name__}}
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

# The following error measures are the ones used in
#
# Bahl, A., Stemmler, M. B., Herz, A. V. M., & Roth, A. (2012).
# Automated optimization of a reduced layer 5 pyramidal cell model based on experimental data.
# Journal of Neuroscience Methods, 210(1), 22-34.
#
def hyperpolarizing_current_steps_error(t,V,Iinj,Vref):
    #### ===>>> Iinj is an array of values, not a matrix like V and Vref
    idx, = np.where(Iinj <= 0)
    return np.sum((V[idx,:]-Vref[idx,:])**2)

def spike_shape_error(t,V,tp,window=spike_shape_error_window,token=None):
    logger('start','spike_shape_error',token)
    if np.isscalar(window[0]):
        window = [window]
    m = min([w[0] for w in window])
    M = max([w[1] for w in window])
    tavg,Vavg,dVavg = extract_average_trace(t,V,tp,[m,M],interp_dt=1./resampling_frequency,token=token)
    err = []
    for w in window:
        idx, = np.where((ephys_data['tavg']>=w[0]) & (ephys_data['tavg']<=w[1]))
        jdx, = np.where((tavg>=w[0]) & (tavg<=w[1]))
        err.append(np.sum((Vavg[jdx] - ephys_data['Vavg'][idx])**2) + \
                       0.1*np.sum((dVavg[jdx] - ephys_data['dVavg'][idx])**2))
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

# The following error measures are the ones used in
#
# Druckmann, S., Banitt, Y., Gidon, A., Schuermann, F., Markram, H., & Segev, I. (2007).
# A novel multiple objective optimization framework for constraining conductance-based neuron models by experimental data.
# Frontiers in Neuroscience, 1(1), 7-18.
#
def spike_rate_error(tp, dur):
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

# The following error measures employ features that have been extracted from
# CA3 experimental data
def input_resistance_error(value):
    return np.abs(features['input_resistance']['mean'] - value) / features['input_resistance']['std']

def time_constant_error(value):
    return np.abs(features['time_constant']['mean'] - value) / features['time_constant']['std']

def Vm_rest_error(value):
    return np.abs(features['Vm_rest']['mean'] - value) / features['Vm_rest']['std']

def rheobase_error(value):
    return np.abs(features['rheobase']['mean'] - value) / features['rheobase']['std']

def initial_firing_rate_error(value):
    return np.abs(features['initial_firing_rate']['mean'] - value) / features['initial_firing_rate']['std']

def steady_state_firing_rate_error(value):
    return np.abs(features['steady_state_firing_rate']['mean'] - value) / features['steady_state_firing_rate']['std']

def check_prerequisites(t,V,ton,toff,tp,Vp,target_nspikes=None,width=None,check_spike_height_decrease=True,token=None):
    logger('start','check_prerequisites',token)
    n = V.shape[0]
    idx, = np.where((t>toff-200) & (t<toff))
    for i in range(n):
        # something weird in the membrane potential
        if np.mean(V[i,idx]) > -40 and np.std(V[i,idx]) < 3:
            print('%d check_prerequistes: mean(voltage) > -40 and std(voltage) < 3.' % token)
            logger('end','check_prerequisites',token)
            return False
        # first of all, check that if spikes are present, they are only between ton and toff
        if len(np.where(tp[i] <= ton)[0]) > 1:
            print('%d check_prerequisites: more than one spike before stimulus onset.' % token)
            logger('end','check_prerequisites',token)
            return False
        if len(np.where(tp[i] > toff)[0]) > 0:
            print('%d check_prerequisites: spikes after stimulus offset.' % token)
            logger('end', 'check_prerequisites', token)
            return False
        # number of spikes in the simulated trace, between ton and toff
        nspikes = len(np.where((tp[i]>ton) & (tp[i]<toff+10))[0])
        try:
            # this part will raise an exception if the user didn't pass the target number of spikes
            # spikes where there shouldn't be any
            if target_nspikes[i] == 0 and nspikes != 0:
                print('%d check_prerequistes: spikes where there shouldn\'t be any.' % token)
                logger('end','check_prerequisites',token)
                return False
            # no spikes where there should be some
            if target_nspikes[i] > 0 and nspikes == 0:
                print('%d check_prerequistes: no spikes where there should be some.' % token)
                logger('end','check_prerequisites',token)
                return False
            # too many spikes
            if nspikes > target_nspikes[i]*2:
                print('%d check_prerequistes: too many spikes (%d instead of %d).' % (token,nspikes,target_nspikes[i]))
                logger('end','check_prerequisites',token)
                return False
            # too few spikes
            if nspikes < target_nspikes[i]/2:
                print('%d check_prerequistes: too few spikes (%d instead of %d).' % (token,nspikes,target_nspikes[i]))
                logger('end','check_prerequisites',token)
                return False
        except:
            pass
        ### some additional checks
        # no spike width should exceed 3 ms
        if not width is None and nspikes > 0 and np.max(width[i]) > 3:
            logger('end','check_prerequisites',token)
            return False
        # decrease in spike height from the third to the penultimate spike shouldn't exceed 20%
        if check_spike_height_decrease and nspikes > 2:
            for j in range(2,nspikes-1):
                if Vp[i][j] < 0.8*Vp[i][0]:
                    print('%d check_prerequistes: spike height decreased by more than 20%%.' % token)
                    logger('end','check_prerequisites',token)
                    return False
    logger('end','check_prerequisites',token)
    return True

def features_error(parameters):
    try:
        features_error.ncalls += 1
    except:
        features_error.__dict__['ncalls'] = 1

    token = int(1e9 * np.random.uniform())
    logger('start', 'features_error', token)

    # default values for the error measures
    default_measures = {}
    for obj in objectives:
        default_measures[obj] = 100.
    measures = default_measures.copy()

    # build the neuron with the current parameters
    neuron = make_simplified_neuron(parameters)

    # start with passive properties
    if 'input_resistance' in features or 'time_constant' in features or 'Vm_rest' in features:
        I = []               # [nA]
        dt = 0.05            # [ms]
        dur = 500.           # [ms]
        tbefore = 500.       # [ms]
        tafter = 100.        # [ms]
        V0 = -65.            # [mV]

        if 'input_resistance' in features or 'time_constant' in features:
            I.append(-0.2)
        if 'Vm_rest' in features:
            I.append(0)

        # run the simulation
        t,V = current_steps(neuron, I, dt, dur, tbefore, tafter, V0, token)
        # check that the neuron didn't spike
        logger('start', 'extractAPPeak', token)
        tp,Vp = extractAPPeak(t, V, threshold=ap_threshold, min_distance=1)
        logger('end', 'extractAPPeak', token)
        if not check_prerequisites(t,V,tbefore,tbefore+dur,tp,Vp,np.zeros(len(I)),token=token):
            logger('end', 'features_error (1) ' + str(default_measures), token)
            return default_measures

        if 'input_resistance' in features:
            # extract the input resistance
            i, = np.where(np.array(I) < 0)
            idx, = np.where((t>tbefore-200) & (t<tbefore))
            jdx, = np.where((t>tbefore+dur-200) & (t<tbefore+dur))
            if np.std(V[i,idx]) < 0.1 and np.std(V[i,jdx]) < 0.1:
                Vrest = np.mean(V[i,idx])
                Vstep = np.mean(V[i,jdx])
                Rm = (Vrest-Vstep) / (-I[i])
                measures['input_resistance'] = input_resistance_error(Rm)
            else:
                print('%d features_error: subthreshold oscillations.' % token)
                logger('end', 'features_error (2) ' + str(default_measures), token)
                return default_measures
        if 'time_constant' in features:
            # extract the time constant
            i, = np.where(np.array(I) < 0)
            idx, = np.where((t>tbefore) & (t<tbefore+300))
            if np.argmax(V[i,idx]) == 0:
                x = t[idx] - t[idx[0]]
                y = V[i,idx] - np.min(V[i,idx])
                popt,pcov = curve_fit(lambda x,a,tau: a*np.exp(-x/tau), x, y, p0=(y[0],20))
                measures['time_constant'] = time_constant_error(popt[1])
            else:
                print('%d features_error: Vm not decreasing during pulse.' % token)
                logger('end', 'features_error (3) ' + str(default_measures), token)
                return default_measures
        if 'Vm_rest' in features:
            # extract the resting Vm
            i, = np.where(np.array(I) == 0)
            idx, = np.where((t>tbefore-200) & (t<tbefore))
            if np.std(V[i,idx]) < 0.1:
                Vrest = np.mean(V[i,idx])
                measures['Vm_rest'] = Vm_rest_error(Vrest)
            else:
                print('%d features_error: subthreshold oscillations.' % token)
                logger('end', 'features_error (4) ' + str(default_measures), token)
                return default_measures

        #import pylab as p
        #for v in V:
        #    p.plot(t,v,'k')
        #p.xlabel('Time (ms)')
        #p.ylabel('Membrane voltage (mV)')
        #p.show()

    # then apply a ramp of current
    if 'rheobase' in features:
        I0 = (features['rheobase']['mean'] - 5*features['rheobase']['std']) * 1e-3
        I1 = (features['rheobase']['mean'] + 5*features['rheobase']['std']) * 1e-3
        dt = 0.05            # [ms]
        dur = 1000.          # [ms]
        tbefore = 100.       # [ms]
        tafter = 100.        # [ms]
        try:
            V0 = features['Vm_rest']['mean']
        except:
            V0 = -65.            # [mV]

        # run the simulation
        t,V,I = current_ramp(neuron, I1, I0, dt, dur, tbefore, tafter, V0, token)
        logger('start', 'extractAPPeak', token)
        tp,Vp = extractAPPeak(t, V, threshold=ap_threshold, min_distance=1)
        logger('end', 'extractAPPeak', token)
        # check for the presence of at least 3 spikes during the injection of the ramp
        if len(np.where((tp[0] > tbefore) & (tp[0] < tbefore+dur))[0]) < 3:
            print('%d features_error: fewer than 3 spikes during the injection of a ramp of current.' % token)
            logger('end', 'features_error (5) ' + str(default_measures), token)
            return default_measures
        if not check_prerequisites(t,V,tbefore,tbefore+dur,tp,Vp,check_spike_height_decrease=False,token=token):
            logger('end', 'features_error (6) ' + str(default_measures), token)
            return default_measures
        rheobase = I[0,t==tp[0][tp[0]>tbefore][0]][0] * 1e3
        measures['rheobase'] = rheobase_error(rheobase)
        #import pylab as p
        #p.plot(t,V[0,:],'k')
        #p.plot(t,I[0,:]*1e2-70,'r')
        #p.plot([tp[0][0],tp[0][0]],[-100,50],'k--')
        #p.plot([t[0],t[-1]],[rheobase*0.1-70,rheobase*0.1-70],'r--')
        #p.show()

    if 'initial_firing_rate' in features or 'steady_state_firing_rate' in features:
        if 'rheobase' in features:
            I = [rheobase*1.25*1e-3]               # [nA]
        else:
            I = [0.2]
        dt = 0.05            # [ms]
        dur = 1000.          # [ms]
        tbefore = 200.       # [ms]
        tafter = 100.        # [ms]
        try:
            V0 = features['Vm_rest']['mean']
        except:
            V0 = -65.            # [mV]

        # run the simulation
        t,V = current_steps(neuron, I, dt, dur, tbefore, tafter, V0, token)
        logger('start', 'extractAPPeak', token)
        tp,Vp = extractAPPeak(t, V, threshold=ap_threshold, min_distance=1)
        logger('end', 'extractAPPeak', token)
        # check the prerequisites once to discard situations in which Vm is weird (which will mess up the extraction of the AP half-width
        if not check_prerequisites(t,V,tbefore,tbefore+dur,tp,Vp,check_spike_height_decrease=True,token=token):
            logger('end', 'features_error (7) ' + str(default_measures), token)
            return default_measures
        # extract the threshold values that will be used to detect the half-widths
        logger('start', 'extractAPThreshold', token)
        tth,Vth = extractAPThreshold(t, V, threshold=ap_threshold, tpeak=tp, model=True)
        logger('end', 'extractAPThreshold', token)
        try:
            # try to extract the half-widths
            logger('start', 'extractAPHalfWidth', token)
            Vhalf,width,interval = extractAPHalfWidth(t, V, threshold=ap_threshold, tpeak=tp, Vpeak=Vp, tthresh=tth, Vthresh=Vth, interp=False)
            logger('end', 'extractAPHalfWidth', token)
            ok = True
        except:
            logger('end', 'extractAPHalfWidth+++', token)
            ok = False
        if not ok or not check_prerequisites(t,V,tbefore,tbefore+dur,tp,Vp,width=width,check_spike_height_decrease=True,token=token):
            logger('end', 'features_error (8) ' + str(default_measures), token)
            return default_measures
        if 'initial_firing_rate' in features:
            try:
                rate = 1e3 / np.diff(tp[0][:2])[0]
            except:
                # only one spike
                rate = 1e3 / dur
            measures['initial_firing_rate'] = initial_firing_rate_error(rate)
        if 'steady_state_firing_rate' in features:
            ttran = 100
            idx, = np.where((tp[0]>tbefore+ttran) & (tp[0]<tbefore+dur))
            rate = len(idx) / ((dur-ttran)*1e-3)
            measures['steady_state_firing_rate'] = steady_state_firing_rate_error(rate)
        #import pylab as p
        #p.ion()
        #p.plot(t,V[0,:],'k')
        #p.plot(tp[0],Vp[0],'ro')
        #p.show()
        #import pdb
        #pdb.set_trace()

    logger('end', 'features_error ' + str(measures), token)
    return measures

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

    # number of spikes in the reference trace (ephys data)
    target_nspikes = [sum((tp>ephys_data['tbefore']) & (tp<ephys_data['tbefore']+ephys_data['dur'])) \
                          for tp in ephys_data['tp']]
    if check_prerequisites(t,V,ephys_data['tbefore'],ephys_data['tbefore']+ephys_data['dur'],tp,Vp,target_nspikes,token=token):
        logger('start', 'extractAPThreshold', token)
        tth,Vth = extractAPThreshold(t, V, threshold=ap_threshold, tpeak=tp, model=True)
        logger('end', 'extractAPThreshold', token)
        logger('start', 'extractAPHalfWidth', token)
        try:
            Vhalf,width,interval = extractAPHalfWidth(t, V, threshold=ap_threshold, tpeak=tp, Vpeak=Vp, tthresh=tth, Vthresh=Vth, interp=False)
            ok = True
            logger('end', 'extractAPHalfWidth', token)
        except:
            ok = False
            logger('end', 'extractAPHalfWidth+++', token)
        if ok and check_prerequisites(t,V,ephys_data['tbefore'],ephys_data['tbefore']+ephys_data['dur'],tp,Vp,target_nspikes,width,token):
            if SAVE_DEBUG_INFO:
                opts = {'%s_spikes' % token: {'tp': tp, 'Vp': Vp, 'tth': tth, 'Vth': Vth}}
                CA3.utils.h5.save_h5_file(h5_filename, 'a', **opts)
            if 'hyperpolarizing_current_steps' in objectives:
                measures['hyperpolarizing_current_steps'] = hyperpolarizing_current_steps_error(t,V,ephys_data['I_amplitudes'],ephys_data['V'])
            if 'isi' in objectives:
                measures['isi'] = isi_error(tp)
            if 'spike_onset' in objectives or 'spike_offset' in objectives:
                err = spike_shape_error(t,V,tp,spike_shape_error_window,token)
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
        isi = np.diff(spike_times[i][k[i]:])
        A.append(0)
        for j in range(1,len(isi)):
            A[-1] += (isi[j]-isi[j-1]) / (isi[j]+isi[j-1])
        A[-1] /= (n_spikes[i] - k[i] - 1)
    return np.mean(A),np.std(A)

def optimize():
    from emoo import Emoo
    from emoo import mpi4py_loaded
    # parse the command-line arguments
    parser = arg.ArgumentParser(description='Fit a multicompartmental neuron model to data.')
    parser.add_argument('config_file', type=str, action='store', help='Configuration file')
    parser.add_argument('--single-compartment', action='store_true', help='Use a single-compartment neuron model')
    args = parser.parse_args(args=sys.argv[2:])

    if not os.path.isfile(args.config_file):
        print('%s: no such file.' % args.config_file)
        sys.exit(1)

    cp = ConfigParser.ConfigParser()
    cp.optionxform = str
    cp.read(args.config_file)

    try:
        passive_opt_file = cp.get('Optimization','passive_results')
    except:
        print('Option [Optimization/passive_results] missing. It should contain the path of the H5 file with the results of the optimization of passive properties.')
        sys.exit(1)
    if not os.path.isfile(passive_opt_file):
        print('%s: no such file.' % passive_opt_file)
        sys.exit(1)

    global optimization_mode
    if 'objectives' in [_[0] for _ in cp.items('Optimization')]:
        for obj in cp.get('Optimization','objectives').split(','):
            objectives.append(obj)
            if obj in ('spike_onset','spike_offset'):
                idx = 1
                if obj == 'spike_onset':
                    idx = 0
                try:
                    spike_shape_error_window[idx] = map(float, cp.get(obj,'window').split(','))
                except:
                    pass
        try:
            data_file = cp.get('Optimization','data_file')
        except:
            print('Option [Optimization/data_file] missing. It should contain the path of the file containing the electrophysiological data.')
            sys.exit(1)
        if not os.path.isfile(data_file):
            print('%s: no such file.' % data_file)
            sys.exit(1)
        optimization_mode = 'objectives'
    elif 'features' in [_[0] for _ in cp.items('Optimization')]:
        for feat in cp.get('Optimization','features').split(','):
            features[feat] = {'mean': cp.getfloat(feat,'mean'), 'std': cp.getfloat(feat,'std')}
            objectives.append(feat)
        optimization_mode = 'features'
    else:
        print('Either [Optimization/objectives] or [Optimization/features] must be present in the configuration file.')
        sys.exit(1)

    for var in 'Cm','Rm','El':
        try:
            a = map(float, cp.get('Variables',var).split(','))
            variables.append([var, a[0], a[1]])
        except:
            print('Option [Variables/%s] missing.' % var)
            sys.exit(1)
    if not args.single_compartment:
        try:
            a = map(float, cp.get('Variables','scaling').split(','))
            variables.append(['scaling', a[0], a[1]])
        except:
            print('Option [Variables/scaling] missing and command line switch --single-compartment not specified.')
            sys.exit(1)

    try:
        for cond in cp.get('Variables','conductances').split(','):
            for entry in cp.items(cond):
                try:
                    a = map(float, entry[1].split(','))
                except:
                    if entry[0] == 'dend_mode':
                        dendritic_modes[cond] = entry[1]
                    else:
                        print('Unknown key,value pair in section [%s]: %s,%s.' % (cond,entry[0],entry[1]))
                else:
                    variables.append([cond + '_' + entry[0], a[0], a[1]])
    except ConfigParser.NoSectionError:
        print('Unknown conductance [%s]: fix your configuration file.' % cond)
        sys.exit(0)
    except ConfigParser.NoOptionError:
        print('No active conductances in the model.')
    else:
        if len(dendritic_modes) == 0 and not args.single_compartment:
            print('No dendritic mode specified in a multi-compartment mode.')
            sys.exit(1)

    # load the data relative to the optimization of the passive properties
    data = CA3.utils.h5.load_h5_file(passive_opt_file)

    # output filename
    global h5_filename
    try:
        h5_filename = cp.get('Optimization','out_file')
    except:
        h5_filename = CA3.utils.h5.make_output_filename(os.path.basename(data['swc_filename']).rstrip('.swc'), '.h5', with_rand=True)

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
    elif os.path.basename(passive_opt_file) == 'DH070613-1-.Edit.scaled_20141223-194818.h5':
        ReducedNeuron = CA3.cells.SimplifiedNeuron
    elif data['model_type'].lower() == 'thorny':
        model_type = 'thorny'
        ReducedNeuron = CA3.cells.ThornyNeuron
    elif data['model_type'].lower() == 'athorny':
        model_type = 'athorny'
        ReducedNeuron = CA3.cells.AThornyNeuron
    elif data['model_type'].lower() != 'simplified':
        print('The model type must be one of "simplified", "thorny" or "athorny".')
        sys.exit(1)

    if model_type == 'single_compartment':
        if 'scaling' in [var[0] for var in variables]:
            print('Command line switch --single-compartment conflicts with option [Variables/scaling].')
            sys.exit(1)

    if optimization_mode == 'objectives':
        # load the ephys data to use in the optimization
        global ephys_data
        ephys_data = CA3.utils.h5.load_h5_file(data_file)
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
        # find the average spike shape
        tp = [ephys_data['tp'][str(i)] for i in range(ephys_data['V'].shape[0])]
        ephys_data['tavg'],ephys_data['Vavg'],ephys_data['dVavg'] = extract_average_trace(ephys_data['t'],ephys_data['V'],
                                                                                          tp,window=[spike_shape_error_window[0][0],
                                                                                                     spike_shape_error_window[1][1]],
                                                                                          interp_dt=1./resampling_frequency)
        # find the firing rate corresponding to the maximum value of the injected current
        idx = np.where(np.max(ephys_data['I'],axis=1) == np.max(ephys_data['I'][-1,:]))[0][0]
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
    n_individuals = cp.getint('Algorithm','n_individuals')
    n_generations = cp.getint('Algorithm','n_generations')
    p_m = cp.getfloat('Algorithm', 'mutation_prob')
    etam_start = cp.getfloat('Algorithm', 'etam_start')
    etam_end = cp.getfloat('Algorithm', 'etam_end')
    etac_start = cp.getfloat('Algorithm', 'etac_start')
    etac_end = cp.getfloat('Algorithm', 'etac_end')

    global emoo
    emoo = Emoo(N=n_individuals, C=2*n_individuals, variables=variables, objectives=objectives)

    d_etam = (etam_end - etam_start) / n_generations
    d_etac = (etac_end - etac_start) / n_generations
    emoo.setup(eta_m_0=etam_start, eta_c_0=etac_start, p_m=p_m, finishgen=0, d_eta_m=d_etam, d_eta_c=d_etac)

    if optimization_mode == 'objectives':
        emoo.get_objectives_error = objectives_error
    elif optimization_mode == 'features':
        emoo.get_objectives_error = features_error
    else:
        print('Unknown optimization mode "%s".' % optimization_mode)
        sys.exit(1)
    emoo.checkpopulation = check_population
    emoo.evolution(generations=n_generations)

    if emoo.master_mode:
        CA3.utils.h5.save_h5_file(h5_filename, 'a', parameters={'etam_start': etam_start, 'etam_end': etam_end,
                                                                'etac_start': etac_start, 'etac_end': etac_end,
                                                                'p_m': p_m},
                                  variables=variables, model_type=model_type, objectives=objectives,
                                  h5_file=passive_opt_file, optimization_mode=optimization_mode,
                                  neuron_pars=neuron_pars, ap_threshold=ap_threshold,
                                  dendritic_modes=dendritic_modes)
        if optimization_mode == 'objectives':
            CA3.utils.h5.save_h5_file(h5_filename, 'a', ephys_file=data_file, ephys_data=ephys_data,
                                      spike_shape_error_window=spike_shape_error_window)
        elif optimization_mode == 'features':
            CA3.utils.h5.save_h5_file(h5_filename, 'a', features=features)
        CA3.utils.h5.save_text_file_to_h5_file(h5_filename, args.config_file, 'a', 'config_file')

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
    tth,Vth = extractAPThreshold(t, V, threshold=ap_threshold, tpeak=tp, model=True)
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
    p.plot(ephys_data['tavg'],ephys_data['Vavg'],'k')
    p.plot(tavg,Vavg,'r')
    p.xlim(wndw)
    p.ylim([-70,50])
    p.xticks(np.linspace(wndw[0],wndw[1],3))
    p.yticks([-70,-30,10,50])
    p.xlabel('Time (ms)')
    p.ylabel('Vm (mV)')
    p.title(title)
    remove_border()

    p.axes([0.75,0.2,0.2,0.7])
    p.plot(ephys_data['tavg'],ephys_data['dVavg'],'k')
    p.plot(tavg,dVavg,'r')
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

def display_rheobase(neuron, data):
    try:
        V0 = data['features']['Vm_rest']['mean']
    except:
        V0 = -65.
    Istart = 1e-3*(data['features']['rheobase']['mean']-5*data['features']['rheobase']['std'])
    Istop = 1e-3*(data['features']['rheobase']['mean']+5*data['features']['rheobase']['std'])
    dt = 0.05
    dur = 1000
    tbefore = 100
    tafter = 200
    t,V,I = current_ramp(neuron, Istop, Istart, dt, dur, tbefore, tafter, V0)
    tp,Vp = extractAPPeak(t,V,threshold=ap_threshold,min_distance=1)
    rheobase = I[0,t==tp[0][0]][0]
    p.figure(figsize=(5,3))
    p.axes([0.05,0.05,0.9,0.9])
    p.plot([tp[0][0],tp[0][0]],[-82,np.max(V)+2],'k--')
    p.plot(t,V[0,:],'k')
    p.plot(t,-80 + I[0,:]*20,'r')
    p.plot(tp[0][0],-80+rheobase*20,'ro')
    p.axis([0,tbefore+dur+tafter,-82,np.max(V)+2])
    p.plot([tbefore-20,tbefore-20],[-20,20],'k')
    p.plot([tbefore-20,tbefore-20],[-77,-72],'r') # 250 pA
    p.plot([tbefore-20,tbefore+180],[-20,-20],'k')
    p.text(tbefore+80,-23,'200 ms',horizontalalignment='center',verticalalignment='top')
    p.text(tbefore-30,0,'40 mV',horizontalalignment='right',verticalalignment='center')
    p.text(tbefore-30,-74.5,'250 pA',horizontalalignment='right',verticalalignment='center',color='r')
    p.text(tp[0][0]-20,-80+rheobase*20+3,'%.0f pA' % (rheobase*1e3),horizontalalignment='right')
    p.axis('off')

def display():
    global p
    import matplotlib.pyplot as p
    set_rc_defaults()
    parser = arg.ArgumentParser(description='Fit the parameters of a reduced morphology to electrophysiological data')
    parser.add_argument('filename', type=str, action='store', help='Path of the configuration file to use.')
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

    # find out whether the model contained an axon
    if 'nat_gbar_ais' in data['variables'] or 'nat_scaling_ais' in data['variables']:
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

    if data['optimization_mode'] == 'objectives':
        try:
            global spike_shape_error_window
            spike_shape_error_window = data['spike_shape_error_window']
        except:
            pass
        for i,pos in enumerate(['onset','offset']):
            print('Spike %s window: [%.1f,%.1f] ms.' % (pos,spike_shape_error_window[i][0],spike_shape_error_window[i][1]))

    # plot the evolution of the optimization variables
    nbins = 80
    nvar = len(data['variables'])
    max_num_panels = 12 # maximum number of panels per figure
    for nfig in range(int(np.ceil(float(nvar)/max_num_panels))):
        c = 3
        r = min(max_num_panels/c,np.ceil(float(nvar-nfig*max_num_panels)/c))
        p.figure(figsize=(c*3,r*3))
        i = 1
        for v in data['variables'][nfig*max_num_panels:]:
            var = np.zeros((nbins,ngen))
            for j in range(ngen):
                var[:,j],edges = np.histogram(data['generations']['%d'%j][:,data['columns'][v[0]]], bins=nbins, range=[float(v[1]),float(v[2])])
            tmp = data['generations']['%d'%(ngen-1)][:,data['columns'][v[0]]]
            ax = make_axes(r,c,i,spacing=[0.2/c,0.2/r])
            opt = {'origin': 'lower', 'cmap': p.get_cmap('Greys'), 'interpolation': 'nearest'}
            if edges[-1] > 1000:
                coeff = 1e-3
            elif np.max(np.abs(edges)) < 0.001:
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
            p.yticks(np.linspace(edges[0],edges[-1],3)*coeff)
            remove_border()
            i += 1
            if i > r*c:
                break
        p.savefig(base_dir + '/variables_%d.pdf' % nfig)

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
                ax = make_axes(r,c,subp,offset=[0.1,0.15])
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
                if key in data['dendritic_modes']:
                    pars[key]['dend_mode'] = data['dendritic_modes'][key]
        if 'nat' in pars and not 'vtraub_offset_soma' in pars['nat']:
            pars['nat']['vtraub_offset_soma'] = vtraub_offset
        if with_axon:
            pars['axon'] = copy.deepcopy(pars['soma'])
            try:
                pars['axon'].pop('L')
            except:
                pass
            try:
                pars['axon'].pop('diam')
            except:
                pass
            try:
                pars['axon'].pop('area')
            except:
                pass
            for key in 'nat_scaling_hillock','nat_scaling_ais':
                if key in parameters:
                    factor = pars['nat'].pop(key[4:])
                    pars['nat']['gbar_'+key[12:]] = factor * parameters['nat_gbar_soma']
            if not 'vtraub_offset_ais' in pars['nat']:
                pars['nat']['vtraub_offset_ais'] = pars['nat']['vtraub_offset_soma']
            if not 'vtraub_offset_hillock' in pars['nat']:
                pars['nat']['vtraub_offset_hillock'] = pars['nat']['vtraub_offset_soma']

        # construct the model
        neuron = ctor(pars, with_axon)

        if data['optimization_mode'] == 'objectives':
            # simulate the injection of currents into the model
            t,V = current_steps(neuron, data['ephys_data']['I_amplitudes'], data['ephys_data']['dt'][0],
                                data['ephys_data']['dur'], data['ephys_data']['tbefore'],
                                data['ephys_data']['tafter'], np.mean(data['ephys_data']['V'][:,0]))
            tp,Vp = CA3.utils.extractAPPeak(t,V,threshold=ap_threshold,min_distance=1)
            global ephys_data
            if obj == 'hyperpolarizing_current_steps':
                err = hyperpolarizing_current_steps_error(t,V,data['ephys_data']['I_amplitudes'],data['ephys_data']['V'])
            elif obj == 'spike_onset':
                ephys_data = {'Vth': [data['ephys_data']['Vth']['%04d'%i] for i in range(len(data['ephys_data']['Vth']))]}
                for k in 'tavg','Vavg','dVavg':
                    ephys_data[k] = data['ephys_data'][k]
                err = spike_shape_error(t,V,tp,window=spike_shape_error_window)
                err = err[0]
            elif obj == 'spike_offset':
                ephys_data = {'Vth': [data['ephys_data']['Vth']['%04d'%i] for i in range(len(data['ephys_data']['Vth']))]}
                for k in 'tavg','Vavg','dVavg':
                    ephys_data[k] = data['ephys_data'][k]
                err = spike_shape_error(t,V,tp,window=spike_shape_error_window)
                err = err[1]
            elif obj == 'isi':
                tp,Vp = extractAPPeak(t, V, threshold=ap_threshold, min_distance=1)
                ephys_data = {'tp': [data['ephys_data']['tp']['%04d'%i] for i in range(len(data['ephys_data']['tp']))]}
                err = isi_error(tp)
            print('%s%s error: %g.' % (obj[0].upper(),obj[1:].replace('_',' '),err))

            globals()['display_' + obj](t,V,data['ephys_data'])
            p.savefig(base_dir + '/' + obj + '.pdf')
        else:
            if obj == 'rheobase':
                display_rheobase(neuron,data)
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
        fid.write('\\subsection*{Variables}')
        for i in range(nfig+1):
            fid.write('\\begin{figure}[htb]\n')
            fid.write('\\centering\n')
            fid.write('\\includegraphics[width=\\textwidth]{%s/variables_%d.pdf}\n' % (base_dir,i))
            if i == 0:
                fid.write('\\caption{Evolution of optimization variables with generation. Dark areas indicate clustering of individuals. ')
                fid.write('The red dashed lines indicate the optimization bounds.}\n')
            else:
                fid.write('\\caption{Evolution of optimization variables with generation (part %d).}\n' % (i+1))
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
            filename = '%s/%s.pdf' % (base_dir,obj)
            if os.path.exists(filename):
                fid.write('\\begin{figure}[htb]\n')
                fid.write('\\centering\n')
                fid.write('\\includegraphics{%s}\n' % filename)
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

