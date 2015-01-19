#!/usr/bin/env python

import os
import sys
import CA3
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

timestamp = lambda : time.strftime('%b %d, %H:%M:%S', time.localtime(time.time()))

DEBUG = False
ReducedNeuron = CA3.cells.SimplifiedNeuron

# the list of objectives
objectives = ['current_steps']

# the variables and their lower and upper search bounds
variables = [
    ['Cm', 0.5, 2.],
    ['Rm', 1e3, 50e3],
    ['El', -80., -50.],
    ['scaling', 0.2, 3.]]

# the neuron parameters that have been obtained by the previous
# optimization of the passive properties
neuron_pars = {'area': {'soma': None, 'proximal': None, 'distal': None, 'basal': None},
               'L': {'proximal': None, 'distal': None, 'basal': None},
               'Ra': {'soma': None, 'proximal': None, 'distal': None, 'basal': None}}

# the electrophysiological data used in the optimization
ephys_data = None

def make_simplified_neuron(pars):
    parameters = {'scaling': pars['scaling'],
                  'soma': {'area': pars['area']['soma'], 'Cm': pars['Cm'],
                           'Ra': pars['Ra']['soma'], 'El': pars['El'], 'Rm': pars['Rm']},
                  'proximal': {'area': pars['area']['proximal'], 'L': pars['L']['proximal'], 'Ra': pars['Ra']['proximal'],
                               'El': pars['El'], 'area': pars['area']['proximal']},
                  'distal': {'area': pars['area']['distal'], 'L': pars['L']['distal'], 'Ra': pars['Ra']['distal'],
                             'El': pars['El'], 'area': pars['area']['distal']},
                  'basal': {'area': pars['area']['basal'], 'L': pars['L']['basal'], 'Ra': pars['Ra']['basal'],
                            'El': pars['El'], 'area': pars['area']['basal']}}
    return ReducedNeuron(parameters, with_axon=False, with_active=False)

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
    pars = neuron_pars.copy()
    for k,v in parameters.iteritems():
        pars[k] = v
    neuron = make_simplified_neuron(pars)
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
    #parser.add_argument('--model-type', default='simplified', type=str,
    #                    help='Specify model type (default is "simplified", other options are "athorny" or "thorny")')
    #parser.add_argument('--proximal-limit', type=float,
    #                    help='Limit of the proximal dendrite, in micrometers')
    parser.add_argument('-o','--out-file', type=str, help='Output file name (default: same as morphology file)')
    parser.add_argument('--optimize-length', action='store_true', help='Optimize also the lengths of the functional compartments')
    #parser.add_argument('--optimize-impedance', action='store_true', help='Optimize the somatic impedance of the cell')
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
        neuron_pars['area'][lbl] = data['areas'][lbl]
        neuron_pars['Ra'][lbl] = data['generations'][last][best,data['columns']['Ra_'+lbl]]
        if lbl != 'soma':
            neuron_pars['L'][lbl] = data['generations'][last][best,data['columns']['L_'+lbl]]

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
                                  ephys_file=args.data_file, ephys_data=ephys_data, neuron_pars=neuron_pars)

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

