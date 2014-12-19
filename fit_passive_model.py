#!/usr/bin/env python

import os
import sys
import CA3
import numpy as np
import argparse as arg
import itertools as it
from neuron import h
from emoo import Emoo

DEBUG = False

# the list of objectives
objectives = ['voltage_deflection']

# the variables and their lower and upper search bounds
variables = [
    ['Ra_soma', 80., 200.],
    ['Ra_basal', 700., 2000.],
    ['Ra_proximal', 150., 300.],
    ['Ra_distal', 500., 1200.]]

def voltage_deflection(neuron, amp=-0.5, dur=500, delay=100):
    stim = h.IClamp(neuron.soma[0](0.5))
    stim.amp = amp
    stim.dur = dur
    stim.delay = delay

    rec_t = h.Vector()
    rec_t.record(h._ref_t)
    recorders = []
    basal_idx = []
    proximal_idx = []
    distal_idx = []
    cnt = 0
    distances = []
    h.distance(sec=neuron.soma[0])
    for sec in it.chain(neuron.soma,neuron.basal,neuron.proximal,neuron.distal):
        for seg in sec:
            rec = h.Vector()
            rec.record(seg._ref_v)
            recorders.append(rec)
            dst = h.distance(seg.x, sec=sec)
            if sec in neuron.basal:
                basal_idx.append(cnt)
                dst *= -1
            elif sec in neuron.proximal:
                proximal_idx.append(cnt)
            elif sec in neuron.distal:
                distal_idx.append(cnt)
            distances.append(dst)
            cnt += 1

    basal_idx = np.array(basal_idx)
    proximal_idx = np.array(proximal_idx)
    distal_idx = np.array(distal_idx)

    CA3.utils.run(tend=stim.dur+stim.delay+100, V0=-70, temperature=36)

    idx = np.where(np.array(rec_t) < stim.dur + stim.delay)[0][-1]
    voltages = []
    for rec in recorders:
        voltages.append(rec[idx])

    distances = np.array(distances)
    voltages = np.array(voltages)
    
    m = np.min(distances[basal_idx])
    M = np.max(distances[basal_idx])
    basal = {'distances': (distances[basal_idx]-m)/(M-m), 'voltages': voltages[basal_idx]}

    m = np.min(distances[proximal_idx])
    M = np.max(distances[proximal_idx])
    proximal = {'distances': (distances[proximal_idx]-m)/(M-m), 'voltages': voltages[proximal_idx]}

    m = np.min(distances[distal_idx])
    M = np.max(distances[distal_idx])
    distal = {'distances': (distances[distal_idx]-m)/(M-m), 'voltages': voltages[distal_idx]}

    return distances,voltages,basal,proximal,distal

def compute_detailed_neuron_voltage_deflection(filename, proximal_limit):
    # fixed parameters for the detailed neuron
    parameters = {'scaling': 1,
                  'soma': {'Cm': 1., 'Ra': 100., 'El': -70., 'Rm': 15e3},
                  'proximal': {'Ra': 100., 'El': -70.},
                  'distal': {'Ra': 100., 'El': -70.},
                  'basal': {'Ra': 100., 'El': -70.},
                  'proximal_limit': proximal_limit,
                  'swc_filename': filename}
    # create the neuron
    neuron = CA3.cells.SWCNeuron(parameters, with_axon=False, with_active=False, convert_to_3pt_soma=False)
    distances,voltages,basal,proximal,distal = voltage_deflection(neuron)
    return neuron,distances,voltages,basal,proximal,distal

def voltage_deflection_error(pars):
    if DEBUG:
        import pylab as p
    d = np.round(np.sqrt(np.sum(detailed_neuron['cell'].soma_areas)/np.pi))
    parameters = {'scaling': 1,
                  'soma': {'Cm': 1., 'Ra': pars['Ra_soma'], 'El': -70., 'Rm': 15e3, 'L': d, 'diam': d},
                  'proximal': {'Ra': pars['Ra_proximal'], 'El': -70.},
                  'distal': {'Ra': pars['Ra_distal'], 'El': -70.},
                  'basal': {'Ra': pars['Ra_basal'], 'El': -70.}}
    if 'L_basal' in pars:
        parameters['basal']['L'] = pars['L_basal']
    else:
        parameters['basal']['L'] = np.max(detailed_neuron['cell'].basal_distances)
    if 'L_proximal' in pars:
        parameters['proximal']['L'] = pars['L_proximal']
    else:
        parameters['proximal']['L'] = np.max(detailed_neuron['cell'].proximal_distances)
    if 'L_distal' in pars:
        parameters['distal']['L'] = pars['L_distal']
    else:
        parameters['distal']['L'] = np.max(detailed_neuron['cell'].distal_distances)
    parameters['basal']['diam'] = np.sum(detailed_neuron['cell'].basal_areas) / (np.pi*parameters['basal']['L'])
    parameters['proximal']['diam'] = np.sum(detailed_neuron['cell'].proximal_areas) / (np.pi*parameters['proximal']['L'])
    parameters['distal']['diam'] = np.sum(detailed_neuron['cell'].distal_areas) / (np.pi*parameters['distal']['L'])

    neuron = CA3.cells.SimplifiedNeuron(parameters, with_axon=False, with_active=False)
    distances,voltages,basal,proximal,distal = voltage_deflection(neuron)

    err = 0
    for region,region_name in zip((basal,proximal,distal),('basal','proximal','distal')):
        for d,v in zip(region['distances'],region['voltages']):
            idx, = np.where((detailed_neuron[region_name]['distances']>d-0.05) & (detailed_neuron[region_name]['distances']<d+0.05))
            err += (v - np.mean(detailed_neuron[region_name]['voltages'][idx]))**2
        if DEBUG:
            p.plot(detailed_neuron[region_name]['distances'],detailed_neuron[region_name]['voltages'],'k.')
            p.plot(region['distances'],region['voltages'],'ro')

    if DEBUG:
        p.figure()
        p.plot(detailed_neuron['distances'],detailed_neuron['voltages'],'k.')
        p.plot(distances,voltages,'ro')
        p.show()

    return err

def func_to_optimize(parameters):
    measures = {}
    for obj in objectives:
        measures[obj] = getattr(sys.modules[__name__],obj + '_error')(parameters)
    return measures

def check_population(population, columns, gen):
    print('Generation %03d.' % (gen+1))
    if emoo.master_mode:
        if gen == 0:
            CA3.utils.h5.save_h5_file(h5_filename, 'w', columns=columns)
        CA3.utils.h5.save_h5_file(h5_filename,'a',generations={('%d'%gen): population})

def optimize():
    # parse the command-line arguments
    parser = arg.ArgumentParser(description='Fit a reduced morphology to a detailed one considering only passive properties')
    parser.add_argument('filename', type=str, action='store', help='Path of the file containing the morphology')
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
    parser.add_argument('--proximal-limit', type=float,
                        help='Limit of the proximal dendrite, in micrometers')
    parser.add_argument('-o','--out-file', type=str, help='Output file name (default: same as morphology file)')
    parser.add_argument('--optimize-length', action='store_true', help='Optimize also the lengths of the functional compartments')

    args = parser.parse_args(args=sys.argv[2:])

    if args.filename is None:
        print('You must provide the path of the morphology file.')
        sys.exit(1)
    
    if not os.path.isfile(args.filename):
        print('%s: no such file.' % args.filename)
        sys.exit(2)

    swc_filename = os.path.abspath(args.filename)
    global h5_filename
    if args.out_file is None:
        h5_filename = CA3.utils.h5.make_output_filename(os.path.basename(swc_filename).rstrip('.swc'), '.h5')
    else:
        h5_filename = args.out_file

    if args.proximal_limit is None:
        print('You must provide the maximal length of the proximal apical region in the detailed morphology.')
        sys.exit(3)

    if args.proximal_limit < 0:
        print('The maximal length of the proximal apical region must be non-negative.')
        sys.exit(4)
        
    n,d,v,bas,prox,dist = compute_detailed_neuron_voltage_deflection(swc_filename, args.proximal_limit)
    global detailed_neuron
    detailed_neuron = {'cell': n, 'distances': d, 'voltages': v, 'basal': bas, 'proximal': prox, 'distal': dist}
    
    if args.optimize_length:
        # set the upper and lower bounds of the lengths of functional compartments
        # using as a reference the corresponding lengths in the detailed model
        global variables
        d = np.round(np.max(neuron.basal_distances))
        variables.append(['L_basal', d-50, d+50])
        d = np.round(np.max(neuron.proximal_distances))
        variables.append(['L_proximal', d-50, d+50])
        d = np.round(np.max(neuron.distal_distances) - np.max(neuron.proximal_distances))
        variables.append(['L_distal', d-50, d+50])

    # initiate the Evolutionary Multiobjective Optimization
    global emoo
    emoo = Emoo(N=args.population_size, C=2*args.population_size, variables=variables, objectives=objectives)

    d_etam = (args.etam_end - args.etam_start) / args.generation_number
    d_etac = (args.etac_end - args.etac_start) / args.generation_number
    emoo.setup(eta_m_0=args.etam_start, eta_c_0=args.etac_start, p_m=args.pm, finishgen=0, d_eta_m=d_etam, d_eta_c=d_etac)
    # Parameters:
    # eta_m_0, eta_c_0: defines the initial strength of the mutation and crossover parameter (large values mean weak effect)
    # p_m: probabily of mutation of a parameter (holds for each parameter independently)

    emoo.get_objectives_error = func_to_optimize
    emoo.checkpopulation = check_population
    
    emoo.evolution(generations=args.generation_number)

    if emoo.master_mode:
        CA3.utils.h5.save_h5_file(h5_filename, 'a', parameters={'etam_start': args.etam_start, 'etam_end': args.etam_end,
                                                                'etac_start': args.etac_start, 'etac_end': args.etac_end,
                                                                'p_m': args.pm}, proximal_limit=args.proximal_limit,
                                  objectives=objectives, variables=variables, swc_filename=swc_filename)

def validate():
    n,d,v,bas,prox,dist = compute_detailed_neuron_voltage_deflection('../morphologies/DH070613-1-.Edit.scaled.swc',200)
    global detailed_neuron
    detailed_neuron = {'cell': n, 'distances': d, 'voltages': v, 'basal': bas, 'proximal': prox, 'distal': dist}
    global DEBUG
    DEBUG = True
    err = voltage_deflection_error({'Ra_soma': 112, 'Ra_basal': 1082, 'Ra_proximal': 274, 'Ra_distal': 581,
                                    'L_basal': 230, 'L_proximal': 151, 'L_distal': 395})
    print err

def help():
    print('This script optimizes a reduced morphology to match a full one.')
    print('')
    print('Author: Daniele Linaro - danielelinaro@gmail.com')
    print('  Date: December 2014')

if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','-H','--help'):
        help()
    elif sys.argv[1] == 'optimize':
        optimize()
    elif sys.argv[1] == 'test':
        validate()
    else:
        print('Unknown working mode: enter "%s -h" for help.' % os.path.basename(sys.argv[0]))

