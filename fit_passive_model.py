
import sys
import os
import itertools as it
import numpy as np
from neuron import h
import CA3
from emoo import Emoo
import pylab as p

DEBUG = False

# Define the variables and their lower and upper search bounds
variables = [
    ['Ra_soma', 80., 200.],
    ['Ra_basal', 700., 2000.],
    ['Ra_proximal', 150., 300.],
    ['Ra_distal', 500., 1200.],
    ['L_basal', -50, 100],
    ['L_proximal', -50, 100],
    ['L_distal', -50, 100]]

# Define the list of objectives (in this case, it is only one)
objectives = ['voltage_deflection']

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

def compute_detailed_neuron_voltage_deflection(filename='../morphologies/DH070613-1-.Edit.scaled.swc'):
    # fixed parameters for the detailed neuron
    parameters = {'scaling': 1,
                  'soma': {'Cm': 1., 'Ra': 100., 'El': -70., 'Rm': 15e3},
                  'proximal': {'Ra': 100., 'El': -70.},
                  'distal': {'Ra': 100., 'El': -70.},
                  'basal': {'Ra': 100., 'El': -70.},
                  'proximal_limit': 400.,
                  'swc_filename': filename}
    # create the neuron
    neuron = CA3.cells.SWCNeuron(parameters, with_axon=False, with_active=False)
    # set the upper and lower bounds of the lengths of functional compartments
    # using as a reference the corresponding lengths in the detailed model
    for var in variables:
        if var[0] == 'L_basal':
            var[1] = np.round(var[1] + np.max(neuron.basal_distances))
            var[2] += var[1]
        elif var[0] == 'L_proximal':
            var[1] = np.round(var[1] + np.max(neuron.proximal_distances))
            var[2] += var[1]
        elif var[0] == 'L_distal':
            var[1] = np.round(var[1] + np.max(neuron.distal_distances) - np.max(neuron.proximal_distances))
            var[2] += var[1]
    distances,voltages,basal,proximal,distal = voltage_deflection(neuron)
    return neuron,distances,voltages,basal,proximal,distal

def voltage_deflection_error(pars):
    d = np.round(np.sqrt(np.sum(detailed_neuron['cell'].soma_areas)/np.pi))
    parameters = {'scaling': 1,
                  'soma': {'Cm': 1., 'Ra': pars['Ra_soma'], 'El': -70., 'Rm': 15e3, 'L': d, 'diam': d},
                  'proximal': {'Ra': pars['Ra_proximal'], 'El': -70., 'L': pars['L_proximal']},
                  'distal': {'Ra': pars['Ra_distal'], 'El': -70., 'L': pars['L_distal']},
                  'basal': {'Ra': pars['Ra_basal'], 'El': -70., 'L': pars['L_basal']}}
    parameters['basal']['diam'] = np.sum(detailed_neuron['cell'].basal_areas) / (np.pi*parameters['basal']['L'])
    parameters['proximal']['diam'] = np.sum(detailed_neuron['cell'].proximal_areas) / (np.pi*parameters['proximal']['L'])
    parameters['distal']['diam'] = np.sum(detailed_neuron['cell'].distal_areas) / (np.pi*parameters['distal']['L'])

    neuron = CA3.cells.SimplifiedNeuron(parameters, with_axon=False, with_active=False)
    distances,voltages,basal,proximal,distal = voltage_deflection(neuron)

    err = 0
    #for d,v in zip(distances,voltages):
    #    idx, = np.where((detailed_neuron['distances']>d-5) & (detailed_neuron['distances']<d+5))
    #    err += (v - np.mean(detailed_neuron['voltages'][idx]))**2

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
    measures = {
        'voltage_deflection': voltage_deflection_error(parameters)
        }
    return measures

def check_population(population, columns, gen):
    print('Generation %03d.' % (gen+1))
    if gen == 0:
        CA3.utils.h5.save_h5_file(h5_filename, 'w', columns=columns)
    CA3.utils.h5.save_h5_file(h5_filename,'a',generations={('%d'%(gen+1)): population})

def main():
    n,d,v,bas,prox,dist = compute_detailed_neuron_voltage_deflection()
    global detailed_neuron
    detailed_neuron = {'cell': n, 'distances': d, 'voltages': v, 'basal': bas, 'proximal': prox, 'distal': dist}
    
    #err = voltage_deflection_error({'Ra_soma': 123, 'Ra_basal': 1513, 'Ra_proximal': 224, 'Ra_distal': 508,
                                    'L_basal': 223, 'L_proximal': 437, 'L_distal': 253})
    #sys.exit(0)

    # Initiate the Evolutionary Multiobjective Optimization
    N = 10
    emoo = Emoo(N=N, C=2*N, variables=variables, objectives=objectives)
    # Parameters:
    # N: size of population
    # C: size of capacity 

    n_gen = 5
    pm = 0.1
    eta_m_0 = 10.
    eta_m_end = 500.
    eta_c_0 = 5.
    eta_c_end = 50.
    finish_gen = 0
    d_eta_m = (eta_m_end - eta_m_0) / (n_gen - finish_gen)
    d_eta_c = (eta_c_end - eta_c_0) / (n_gen - finish_gen)
    emoo.setup(eta_m_0, eta_c_0, pm, finish_gen, d_eta_m, d_eta_c)
    # Parameters:
    # eta_m_0, eta_c_0: defines the initial strength of the mutation and crossover parameter (large values mean weak effect)
    # p_m: probabily of mutation of a parameter (holds for each parameter independently)

    emoo.get_objectives_error = func_to_optimize
    emoo.checkpopulation = check_population
    
    global h5_filename
    h5_filename = 'evolution.h5'
    emoo.evolution(generations=n_gen)

    if emoo.master_mode:
        population = emoo.getpopulation_unnormed() # get the unnormed population

if __name__ == '__main__':
    main()
