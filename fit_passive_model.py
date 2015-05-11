#!/usr/bin/env python

import os
import sys
import CA3
from CA3.utils import timestamp
from CA3.utils.graphics import *
import numpy as np
import argparse as arg
import itertools as it
import time
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

DEBUG = False
ReducedNeuron = CA3.cells.SimplifiedNeuron

# the list of objectives
objectives = ['voltage_deflection']

# the variables and their lower and upper search bounds
variables = [
    ['Ra_soma', 80., 400.],       # 80, 200
    ['Ra_basal', 700., 2000.],    # 700, 2000
    ['Ra_proximal', 100., 300.],  # 150, 300
    ['Ra_distal', 100., 1200.]]   # 500, 1200

# the frequencies used in the computation of the transfer function
tf_frequencies = np.array([0.5,1,2,5,10,20,50,100,200,500,1000])

def make_detailed_neuron(filename, proximal_limit):
    # fixed parameters for the detailed neuron
    parameters = {'scaling': 1,
                  'soma': {'Cm': 1., 'Ra': 100., 'El': -70., 'Rm': 15e3},
                  'proximal': {'Ra': 100., 'El': -70.},
                  'distal': {'Ra': 100., 'El': -70.},
                  'basal': {'Ra': 100., 'El': -70.},
                  'proximal_limit': proximal_limit,
                  'swc_filename': filename}
    return CA3.cells.SWCNeuron(parameters, with_axon=False, convert_to_3pt_soma=False)
    
def make_simplified_neuron(pars, detailed_neuron):
    parameters = {'scaling': 1,
                  'soma': {'Cm': 1., 'Ra': pars['Ra_soma'], 'El': -70., 'Rm': 15e3, 'area': np.sum(detailed_neuron.soma_areas)},
                  'proximal': {'Ra': pars['Ra_proximal'], 'El': -70., 'area': np.sum(detailed_neuron.proximal_areas)},
                  'distal': {'Ra': pars['Ra_distal'], 'El': -70., 'area': np.sum(detailed_neuron.distal_areas)},
                  'basal': {'Ra': pars['Ra_basal'], 'El': -70., 'area': np.sum(detailed_neuron.basal_areas)}}
    if 'L_basal' in pars:
        parameters['basal']['L'] = pars['L_basal']
    else:
        parameters['basal']['L'] = np.max(detailed_neuron.basal_distances)
    if 'L_proximal' in pars:
        parameters['proximal']['L'] = pars['L_proximal']
    else:
        parameters['proximal']['L'] = np.max(detailed_neuron.proximal_distances)
    if 'L_distal' in pars:
        parameters['distal']['L'] = pars['L_distal']
    else:
        parameters['distal']['L'] = np.max(detailed_neuron.distal_distances) - np.max(detailed_neuron.proximal_distances)
    return ReducedNeuron(parameters, with_axon=False)

def voltage_deflection(neuron, amp=-0.5, dur=500, delay=100):
    stim = h.IClamp(neuron.soma[0](0.5))
    stim.amp = amp
    stim.dur = dur
    stim.delay = delay

    rec_t = h.Vector()
    rec_t.record(h._ref_t)
    recorders = []
    soma_idx = []
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
            if sec in neuron.soma:
                soma_idx.append(cnt)
            elif sec in neuron.basal:
                basal_idx.append(cnt)
                dst *= -1
            elif sec in neuron.proximal:
                proximal_idx.append(cnt)
            elif sec in neuron.distal:
                distal_idx.append(cnt)
            distances.append(dst)
            cnt += 1

    soma_idx = np.array(soma_idx)
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
    
    soma = {'distances': distances[soma_idx], 'voltages': voltages[soma_idx]}

    if len(basal_idx) > 1:
        m = np.min(distances[basal_idx])
        M = np.max(distances[basal_idx])
        basal = {'distances': (distances[basal_idx]-m)/(M-m), 'voltages': voltages[basal_idx]}
    else:
        basal = {'distances': [0.5], 'voltages': voltages[basal_idx]}

    if len(proximal_idx) > 1:
        m = np.min(distances[proximal_idx])
        M = np.max(distances[proximal_idx])
        proximal = {'distances': (distances[proximal_idx]-m)/(M-m), 'voltages': voltages[proximal_idx]}
    else:
        proximal = {'distances': [0.5], 'voltages': voltages[proximal_idx]}

    if len(distal_idx) > 1:
        m = np.min(distances[distal_idx])
        M = np.max(distances[distal_idx])
        distal = {'distances': (distances[distal_idx]-m)/(M-m), 'voltages': voltages[distal_idx]}
    else:
        distal = {'distances': [0.5], 'voltages': voltages[distal_idx]}

    stim.amp = 0
    del stim
    return distances,voltages,soma,basal,proximal,distal

def voltage_deflection_error(pars):
    neuron = make_simplified_neuron(pars, detailed_neuron['cell'])
    distances,voltages,soma,basal,proximal,distal = voltage_deflection(neuron)
    err = (soma['voltages'][0]-np.mean(detailed_neuron['soma']['voltages']))**2
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

def impedance(neuron, amp=0.1, n_cycles=3, frequencies=np.array([0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000])):
    frequencies = np.array([float(f) for f in frequencies])
    stim = h.SineClamp(neuron.soma[0](0.5))
    stim.amp = amp
    stim.delay = 0
    rec = {'t': h.Vector(), 'v': h.Vector(), 'i': h.Vector()}
    rec['t'].record(h._ref_t)
    rec['v'].record(neuron.soma[0](0.5)._ref_v)
    rec['i'].record(stim._ref_i)
    nfreq = len(frequencies)
    R = np.zeros(nfreq)
    phi = np.zeros(nfreq)
    for i,f in enumerate(frequencies):
        stim.f = f
        stim.dur = n_cycles/f*1e3
        for r in rec.values():
            r.resize(0)
        CA3.utils.run(tend=stim.dur+stim.delay, V0=-70, temperature=36)
        t = np.array(rec['t'])
        v = np.array(rec['v'])
        idx, = np.where(t > (n_cycles-1)/f*1e3)
        dv = np.max(v[idx]) - np.min(v[idx]) # [mV]
        di = 2*amp # [nA]
        R[i] = dv/di
        tref = (n_cycles-0.75)/f*1e3
        tpeak = t[idx[np.argmax(v[idx])]]
        phi[i] = (tref - tpeak)*1e-3*2*f*np.pi
    stim.amp = 0
    del stim
    return R,phi

def impedance_error(pars):
    neuron = make_simplified_neuron(pars, detailed_neuron['cell'])
    R,phi = impedance(neuron,frequencies=tf_frequencies)
    R_err = np.sum((R-detailed_neuron['impedance'])**2)
    phi_err = np.sum((phi-detailed_neuron['phase'])**2)
    return R_err,phi_err

def length_error(pars):
    # This is ``almost'' correct, in the sense that distances in SWCNeuron are computed from the center of the sections
    # and thus do not reflect the exact length of each section. However, since sections in SWCNeuron are almost always
    # made up of one segment, this is a very good approximation of the true length of the morphology
    ref = np.array([np.max(detailed_neuron['cell'].basal_distances),np.max(detailed_neuron['cell'].proximal_distances), \
                        np.max(detailed_neuron['cell'].distal_distances)-np.max(detailed_neuron['cell'].proximal_distances)])
    return np.sum((np.array([pars['L_basal'],pars['L_proximal'],pars['L_distal']]) - ref)**2)

def objectives_error(parameters):
    try:
        objectives_error.ncalls += 1
    except:
        objectives_error.__dict__['ncalls'] = 1
    if mpi4py_loaded:
        print('%s >>  STARTED objectives_error %d @ %s' % (processor_name,objectives_error.ncalls,timestamp()))
    measures = {}
    measures['voltage_deflection'] = voltage_deflection_error(parameters)
    if 'impedance' in objectives:
        R_err,phi_err = impedance_error(parameters)
        measures['impedance'] = R_err
        measures['phase'] = phi_err
    if 'length' in objectives:
        measures['length'] = length_error(parameters)
    if mpi4py_loaded:
        print('%s << FINISHED objectives_error %d @ %s' % (processor_name,objectives_error.ncalls,timestamp()))
    return measures

def quick_test(filename, proximal_limit):
    n = make_detailed_neuron(filename, proximal_limit)
    d,v,som,bas,prox,dist = voltage_deflection(n)
    global detailed_neuron
    detailed_neuron = {'cell': n, 'distances': d, 'voltages': v, 'soma': som,
                       'basal': bas, 'proximal': prox, 'distal': dist}
    #R,phi = impedance(n)
    #detailed_neuron['impedance'] = R
    #detailed_neuron['phase'] = phi
    parameters = {'Ra_soma': 293.925, 'Ra_basal': 1505, 'Ra_proximal': 293, 'Ra_distal': 367,
                  'L_basal': 155, 'L_proximal': 136, 'L_distal': 319}
    global ReducedNeuron
    ReducedNeuron = CA3.cells.AThornyNeuron
    err = voltage_deflection_error(parameters)
    print err

def check_population(population, columns, gen):
    if mpi4py_loaded:
        print('Processor name: %s' % processor_name)
    if emoo.master_mode:
        print('Generation %03d.' % (gen+1))
        sys.stdout.flush()
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
    parser.add_argument('--model-type', default='simplified', type=str,
                        help='Specify model type (default is "simplified", other options are "athorny" or "thorny")')
    parser.add_argument('--proximal-limit', type=float,
                        help='Limit of the proximal dendrite, in micrometers')
    parser.add_argument('-o','--out-file', type=str, help='Output file name (default: same as morphology file)')
    parser.add_argument('--optimize-length', action='store_true', help='Optimize also the lengths of the functional compartments')
    parser.add_argument('--optimize-impedance', action='store_true', help='Optimize the somatic impedance of the cell')
    parser.add_argument('--with-length-error', action='store_true', help='Minimize the discrepancy between the length of ' + \
                        'the original morphology and of the reduced one.')
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
        
    global ReducedNeuron
    if args.model_type.lower() == 'thorny':
        ReducedNeuron = CA3.cells.ThornyNeuron
    elif args.model_type.lower() == 'athorny':
        ReducedNeuron = CA3.cells.AThornyNeuron
    elif args.model_type.lower() != 'simplified':
        print('The model type must be one of "simplified", "thorny" or "athorny".')
        sys.exit(5)

    n = make_detailed_neuron(swc_filename, args.proximal_limit)
    d,v,som,bas,prox,dist = voltage_deflection(n)

    global detailed_neuron
    detailed_neuron = {'cell': n, 'distances': d, 'voltages': v, 'soma': som, 'basal': bas,
                       'proximal': prox, 'distal': dist}

    if args.optimize_impedance:
        objectives.append('impedance')
        objectives.append('phase')
        R,phi = impedance(n,frequencies=tf_frequencies)
        detailed_neuron['impedance'] = R
        detailed_neuron['phase'] = phi

    if args.optimize_length:
        # set the upper and lower bounds of the lengths of functional compartments
        # using as a reference the corresponding lengths in the detailed model
        global variables
        fraction = 0.5
        d = np.round(np.max(n.basal_distances))
        variables.append(['L_basal', (1-fraction)*d, (1+fraction)*d])
        d = np.round(np.max(n.proximal_distances))
        variables.append(['L_proximal', (1-fraction)*d, (1+fraction)*d])
        d = np.round(np.max(n.distal_distances) - np.max(n.proximal_distances))
        variables.append(['L_distal', (1-fraction)*d, (1+fraction)*d])

    if args.with_length_error:
        objectives.append('length')

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
                                                                'p_m': args.pm}, proximal_limit=args.proximal_limit,
                                  objectives=objectives, variables=variables, swc_filename=swc_filename,
                                  model_type=args.model_type.lower(),
                                  areas={'soma': np.sum(detailed_neuron['cell'].soma_areas), \
                                         'basal': np.sum(detailed_neuron['cell'].basal_areas), \
                                         'proximal': np.sum(detailed_neuron['cell'].proximal_areas), \
                                         'distal': np.sum(detailed_neuron['cell'].distal_areas)})

def display():
    parser = arg.ArgumentParser(description='Fit a reduced morphology to a detailed one considering only passive properties')
    parser.add_argument('filename', type=str, action='store', help='Path of the file containing the morphology')
    args = parser.parse_args(args=sys.argv[2:])
    if not os.path.isfile(args.filename):
        print('%s: no such file.' % args.filename)
        sys.exit(1)
    # read the data
    data = CA3.utils.h5.load_h5_file(args.filename)
    # figure out which reduced neuron model was used
    try:
        global ReducedNeuron
        if data['model_type'] == 'thorny':
            model_type = 'Thorny'
            ReducedNeuron = CA3.cells.ThornyNeuron
        elif data['model_type'] == 'athorny':
            ReducedNeuron = CA3.cells.AThornyNeuron
            model_type = 'A-thorny'
        elif data['model_type'] == 'simplified':
            ReducedNeuron = CA3.cells.SimplifiedNeuron
            model_type = 'Simplified'
    except:
        # different neuron models were added only after a while, and
        # therefore some H5 files lack this information
        print('No "model_type" in %s: using SimplifiedNeuron.' % args.filename)
        model_type = 'Simplified'
    # simulate the detailed model
    swc_filename = '../morphologies/' + os.path.basename(data['swc_filename'])
    detailed = {}
    detailed['neuron'] = make_detailed_neuron(swc_filename,data['proximal_limit'])
    detailed['distances'],detailed['voltages'],detailed['soma'],detailed['basal'], \
        detailed['proximal'],detailed['distal'] = voltage_deflection(detailed['neuron'])
    frequencies = np.array([0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000])
    detailed['impedance'],detailed['phase'] = impedance(detailed['neuron'],frequencies=frequencies)
    # get the optimal parameters
    ngen = len(data['generations'])
    last = str(ngen-1)
    err = np.zeros((data['generations'][last].shape[0],len(data['objectives'])))
    norm_err = np.zeros((data['generations'][last].shape[0],len(data['objectives'])))
    for i,obj in enumerate(data['objectives']):
        err[:,i] = data['generations'][last][:,data['columns'][obj]]
        norm_err[:,i] = (err[:,i] - min(err[:,i])) / (max(err[:,i]) - min(err[:,i]))
        err[:,i] /= np.std(err[:,i])
    best = np.argmin(np.sum(norm_err[:,:2]**2,axis=1))
    Ra_soma = data['generations'][last][best,data['columns']['Ra_soma']]
    Ra_basal = data['generations'][last][best,data['columns']['Ra_basal']]
    Ra_proximal = data['generations'][last][best,data['columns']['Ra_proximal']]
    Ra_distal = data['generations'][last][best,data['columns']['Ra_distal']]
    L_soma = np.round(np.sqrt(np.sum(detailed['neuron'].soma_areas)/np.pi))
    if 'L_basal' in data['columns']:
        L_basal = data['generations'][last][best,data['columns']['L_basal']]
    else:
        L_basal = np.max(detailed['neuron'].basal_distances)
    if 'L_proximal' in data['columns']:
        L_proximal = data['generations'][last][best,data['columns']['L_proximal']]
    else:
        L_proximal = np.max(detailed['neuron'].proximal_distances)
    if 'L_distal' in data['columns']:
        L_distal = data['generations'][last][best,data['columns']['L_distal']]
    else:
        L_distal = np.max(detailed['neuron'].distal_distances) - np.max(detailed['neuron'].proximal_distances)
    diam_basal = np.sum(detailed['neuron'].basal_areas) / (np.pi*L_basal)
    diam_proximal = np.sum(detailed['neuron'].proximal_areas) / (np.pi*L_proximal)
    diam_distal = np.sum(detailed['neuron'].distal_areas) / (np.pi*L_distal)
    simplified = {}
    simplified['neuron'] = make_simplified_neuron({'Ra_soma': Ra_soma, 'Ra_basal': Ra_basal,
                                                   'Ra_proximal': Ra_proximal, 'Ra_distal': Ra_distal,
                                                   'L_basal': L_basal, 'L_proximal': L_proximal,
                                                   'L_distal': L_distal}, detailed['neuron'])
    simplified['distances'],simplified['voltages'],simplified['soma'],simplified['basal'],\
        simplified['proximal'],simplified['distal'] = voltage_deflection(simplified['neuron'])
    simplified['impedance'],simplified['phase'] = impedance(simplified['neuron'],frequencies=frequencies)
    base_dir = '/tmp'
    tex_file = os.path.basename(args.filename)[:-3] + '.tex'

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
        p.imshow(var/np.max(var), extent=[1,ngen,edges[0],edges[-1]], aspect=ngen/(edges[-1]-edges[0]), **opt)
        p.plot(p.xlim(), [float(v[1]),float(v[1])], 'r--')
        p.plot(p.xlim(), [float(v[2]),float(v[2])], 'r--')
        p.ylim([float(v[1])-0.05*(float(v[2])-float(v[1])),float(v[2])+0.05*(float(v[2])-float(v[1]))])
        p.xlabel('Generation #')
        p.ylabel(v[0])
        p.xticks(np.round(np.linspace(0,ngen,6)))
        p.yticks(np.round(np.linspace(edges[0],edges[-1],5)))
        remove_border()
    p.savefig(base_dir + '/variables.pdf')

    # plot the trade-offs between all possible error pairs at the last generation
    n = len(data['objectives'])
    tot = n*(n-1)/2
    c = 3
    r = int(np.ceil(float(tot)/c))
    p.figure(figsize=(c*3,r*3))
    subp = 1
    for i in range(n):
        for j in range(i+1,n):
            ax = make_axes(r,c,subp,offset=[0.1,0.2],spacing=0.1)
            p.plot(err[:,i],err[:,j],'k.',markersize=2)
            p.plot(err[best,i],err[best,j],'ro',markersize=6)
            p.axis([0,3,0,3])
            p.xlabel('%s error' % data['objectives'][i].replace('_',' '))
            p.ylabel('%s error' % data['objectives'][j].replace('_',' '))
            p.xticks([0,1,2,3])
            p.yticks([0,1,2,3])
            remove_border()
            subp += 1
    p.savefig(base_dir + '/errors.pdf')

    p.figure(figsize=(6,4))
    p.axes([0.15,0.2,0.75,0.7])
    p.plot(detailed['distances'][:10:-1],detailed['voltages'][:10:-1],'k.',markersize=2,label='Detailed model')
    p.plot(simplified['distances'],simplified['voltages'],'ro',markersize=5,label='Reduced model')
    p.xlabel('Distance to soma (um)')
    p.ylabel('Voltage (mV)')
    vmin = np.floor(np.min([np.min(simplified['voltages']),np.min(detailed['voltages'])]))
    vmax = np.ceil(np.max([np.max(simplified['voltages']),np.max(detailed['voltages'])]))
    dv = np.round((vmax - vmin)/4)
    p.yticks(np.arange(vmin,vmax+dv/2,dv))
    remove_border()
    p.savefig(base_dir + '/voltage_deflection.pdf')

    p.figure(figsize=(6,5))
    p.axes([0.15,0.6,0.75,0.35])
    p.semilogx(frequencies,detailed['impedance'],'k')
    p.semilogx(frequencies,simplified['impedance'],'r')
    remove_border()
    p.yticks([0,20,40,60,80])
    p.ylabel('Impedance (MOhm)')
    p.axes([0.15,0.125,0.75,0.35])
    p.semilogx(frequencies,detailed['phase'],'k',label='Detailed model')
    p.semilogx(frequencies,simplified['phase'],'r',label='Reduced model')
    p.legend(loc='best')
    p.yticks([-1.6,-1.2,-0.8,-0.4,0])
    p.xlabel('Frequency (Hz)')
    p.ylabel('Phase (rad)')
    remove_border()
    p.savefig(base_dir + '/impedance.pdf')

    # LaTeX generation
    with open(base_dir + '/' + tex_file, 'w') as fid:
        fid.write('\\documentclass[11pt]{scrartcl}\n')
        fid.write('\\usepackage{graphicx}\n')
        fid.write('\\usepackage[squaren]{SIunits}\n')
        fid.write('\\usepackage[a4paper,margin=0.5in]{geometry}')
        fid.write('\\begin{document}\n')
        fid.write('\\section*{Morphology reduction summary}\n')
        fid.write('\n\\noindent ')
        fid.write('Filename: %s.\n' % os.path.abspath(args.filename).replace('_','\_'))
        fid.write('\n\\noindent ')
        fid.write('Morphology file: %s.\n' % os.path.basename(data['swc_filename']))
        fid.write('\n\\noindent ')
        fid.write('Proximal dendrites limit: %g\\,\\micro\\meter.\n' % data['proximal_limit'])
        fid.write('\n\\noindent ')
        fid.write('Model type: %s.\n' % model_type)
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
            fid.write(' ' + var[0].replace('_',' '))
            if var[0] != data['variables'][-1][0]:
                fid.write(',')
        fid.write('.\n')
        fid.write('\n\\noindent ')
        fid.write('Number of generations: %s.\n' % len(data['generations']))
        fid.write('\n\\noindent ')
        fid.write('Number of organisms: %s.\n' % len(data['generations']['0']))
        fid.write('\n\\noindent ')
        fid.write('Optimization parameters: $\\eta_c^0=%g$, $\\eta_c^{\mathrm{end}}=%g$, $\\eta_m^0=%g$, $\\eta_m^{\mathrm{end}}=%g$, $p_m=%g$.\n' % 
                  (data['parameters']['etac_start'],data['parameters']['etac_end'],
                   data['parameters']['etam_start'],data['parameters']['etam_end'],data['parameters']['p_m']))
        fid.write('\\subsection*{Parameters of one good solution}')
        fid.write('\\begin{table}[h!!]\n')
        fid.write('\\centering\n')
        fid.write('\\begin{tabular}{|l|ccc|}\n')
        fid.write('\\hline\n')
        fid.write('\\textbf{Functional section} & \\textbf{Length} & \\textbf{Diameter} & \\textbf{Axial resistance} \\\\\n')
        fid.write('\\hline\n')
        fid.write('Soma & $%g\\,\\micro\\meter$ & $%g\\,\\micro\\meter$ & $%g\\,\\ohm\\cdot\\centi\\meter$ \\\\\n' %
                  (L_soma,L_soma,Ra_soma))
        n = ReducedNeuron.n_basal_sections()
        fid.write('Basal dendrites (%d) & $%g\\,\\micro\\meter$ & $%g\\,\\micro\\meter$ & $%g\\,\\ohm\\cdot\\centi\\meter$ \\\\\n' %
                  (n,round(L_basal),round(diam_basal/n),round(Ra_basal)))
        n = ReducedNeuron.n_proximal_sections()
        fid.write('Proximal apical dendrites (%d) & $%g\\,\\micro\\meter$ & $%g\\,\\micro\\meter$ & $%g\\,\\ohm\\cdot\\centi\\meter$ \\\\\n' %
                  (n,round(L_proximal),round(diam_proximal/n),round(Ra_proximal)))
        n = ReducedNeuron.n_distal_sections()
        fid.write('Distal apical dendrites (%d) & $%g\\,\\micro\\meter$ & $%g\\,\\micro\\meter$ & $%g\\,\\ohm\\cdot\\centi\\meter$ \\\\\n' %
                  (n,round(L_distal),round(diam_distal/n),round(Ra_distal)))
        fid.write('\\hline\n')
        fid.write('\\end{tabular}\n')
        #fid.write('\\caption{Reduced model parameters.}\n')
        fid.write('\\end{table}\n')
        fid.write('\n')
        fid.write('\\subsection*{Variables}')
        fid.write('\\begin{figure}[htb]\n')
        fid.write('\\centering\n')
        fid.write('\\includegraphics[width=\\textwidth]{%s/variables.pdf}\n' % base_dir)
        fid.write('\\caption{Evolution of optimization variables with generation. Dark areas indicate clustering of individuals. ')
        fid.write('The red dashed lines indicate the optimization bounds.}\n')
        fid.write('\\end{figure}\n')
        fid.write('\n')
        fid.write('\\subsection*{Errors}')
        fid.write('\\begin{figure}[htb]\n')
        fid.write('\\centering\n')
        fid.write('\\includegraphics[width=\\textwidth]{%s/errors.pdf}\n' % base_dir)
        fid.write('\\caption{Trade-offs between objectives at the last generation in units of standard deviations: ')
        fid.write('each black dot indicates a solution, while the red dot is the chosen solution ')
        fid.write('(i.e., the one that minimizes the sum of the square normalized voltage deflection ')
        fid.write('and impedance error, \\textbf{without} considering the phase error).}\n')
        fid.write('\\end{figure}\n')
        fid.write('\n')
        fid.write('\\begin{figure}[htb]\n')
        fid.write('\\centering\n')
        fid.write('\\includegraphics{%s/voltage_deflection.pdf}\n' % base_dir)
        fid.write('\\caption{Voltage deflection error}\n')
        fid.write('\\end{figure}\n')
        fid.write('\n')
        fid.write('\\begin{figure}[htb]\n')
        fid.write('\\centering\n')
        fid.write('\\includegraphics{%s/impedance.pdf}\n' % base_dir)
        fid.write('\\caption{Impedance and phase error}\n')
        fid.write('\\end{figure}\n')
        fid.write('\\end{document}\n')
    os.system('pdflatex ' + base_dir + '/' + tex_file)
    os.remove(base_dir + '/' + tex_file)
    os.remove(tex_file[:-4] + '.aux')
    os.remove(tex_file[:-4] + '.log')

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
    elif sys.argv[1] == 'display':
        display()
    else:
        print('Unknown working mode: enter "%s -h" for help.' % os.path.basename(sys.argv[0]))

