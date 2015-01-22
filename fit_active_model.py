#!/usr/bin/env python

import os
import sys
import CA3
from CA3.utils import timestamp
import numpy as np
from scipy.interpolate import interp1d
import argparse as arg
import itertools as it
import time
from neuron import h
from emoo import Emoo
from emoo import mpi4py_loaded
if mpi4py_loaded:
    from mpi4py import MPI
    processor_name = MPI.Get_processor_name()

DEBUG = False
ReducedNeuron = CA3.cells.SimplifiedNeuron

# the list of objectives
objectives = ['current_steps']

# the variables and their lower and upper search bounds
variables = [
    ['Cm', 0.6, 3.],                       # [uF/cm2] membrane capacitance
    ['Rm', 10e3, 30e3],                    # [Ohm cm2] membrane resistance
    ['El', -85., -50.],                    # [mV] reversal potential of leak conductance
    ['scaling', 0.5, 2.],                  # [1] scaling of dendritic capacitance and membrane resistance
    ['nat_gbar_soma', 0., 500.],           # [pS/um2]
    ['nat_gbar_hillock', 0., 20000.],      # [pS/um2]
    ['nat_gbar_ais', 0., 20000.],          # [pS/um2]
    ['nat_half_dist', 0., 500.],           # [um]
    ['nat_lambda', 1., 500.],              # [um]
    ['nat_dend_scaling', 0., 2.],          # [1]
    ['kdr_gbar', 0., 100.],                # [pS/um2]
    ['kdr_dend_scaling', 0., 2.],          # [1]
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

def make_simplified_neuron(parameters):
    pars = neuron_pars.copy()
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
    return ReducedNeuron(pars, with_axon=True, with_active=True)

def current_steps(neuron, amplitudes, dt=0.05, dur=500, tbefore=100, tafter=100, V0=-70):
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
    return T,V

def current_steps_error(parameters):
    neuron = make_simplified_neuron(parameters)
    t,V = current_steps(neuron, ephys_data['hyperpol_I'], ephys_data['dt'], ephys_data['dur'], \
                            ephys_data['tbefore'], ephys_data['tafter'], parameters['El'])
    #import pylab as p
    #for i in range(V.shape[0]):
    #    p.plot(t,ephys_data['hyperpol_V'][i,:],'k')
    #    p.plot(t,V[i,:],'r')
    #p.xlabel('Time (ms)')
    #p.ylabel('Membrane voltage (mV)')
    #p.show()
    return np.sum((V-ephys_data['hyperpol_V'])**2)

def objectives_error(parameters):
    if mpi4py_loaded:
        print('%s >>  STARTED objectives_error @ %s' % (processor_name,timestamp()))
    measures = {}
    measures['current_steps'] = current_steps_error(parameters)
    if mpi4py_loaded:
        print('%s << FINISHED objectives_error @ %s' % (processor_name,timestamp()))
    return measures

def check_population(population, columns, gen):
    if mpi4py_loaded:
        print('Processor name: %s' % processor_name)
    if emoo.master_mode:
        print('Generation %03d. best = %g, mean = %g, std = %g' % (gen+1,population[0,columns['current_steps']],\
                                                                       np.mean(population[:,columns['current_steps']]),\
                                                                       np.std(population[:,columns['current_steps']])))
        sys.stdout.flush()
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
    ephys_data['t'] = np.arange(ephys_data['V'].shape[0]) * ephys_data['dt']
    i = 0
    while np.min(ephys_data['I'][:,i] >= 0):
        i += 1
    idx, = np.where(ephys_data['I'][:,i] < 0)
    ephys_data['tbefore'] = ephys_data['t'][idx[0]]
    ephys_data['dur'] = ephys_data['t'][idx[-1]] - ephys_data['tbefore']
    ephys_data['tafter'] = ephys_data['t'][-1] - ephys_data['dur'] - ephys_data['tbefore']
    j = int((ephys_data['tbefore']+ephys_data['dur'])/2/ephys_data['dt'])
    ephys_data['hyperpol_I'] = np.sort(np.unique(ephys_data['I'][j,:]))
    ephys_data['hyperpol_I'] = ephys_data['hyperpol_I'][ephys_data['hyperpol_I']<0]
    ephys_data['hyperpol_V'] = np.zeros((len(ephys_data['hyperpol_I']),ephys_data['V'].shape[0]))
    for i,amp in enumerate(ephys_data['hyperpol_I']):
        idx, = np.where(ephys_data['I'][j,:] == amp)
        ephys_data['hyperpol_V'][i,:] = np.mean(ephys_data['V'][:,idx],axis=1)
    ephys_data['hyperpol_I'] *= 1e-3

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
    parser = arg.ArgumentParser(description='Fit a reduced morphology to a detailed one considering only passive properties')
    parser.add_argument('filename', type=str, action='store', help='Path of the file containing the morphology')
    args = parser.parse_args(args=sys.argv[2:])
    if not os.path.isfile(args.filename):
        print('%s: no such file.' % args.filename)
        sys.exit(1)

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

