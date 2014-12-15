
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
variables = [['Ra_soma', 80, 200], ['Ra_basal', 700, 2000], ['Ra_proximal', 150, 300], ['Ra_distal', 500, 1200]]

# Define the list of objectives (in this case, it is only one)
objectives = ['voltage_deflection']

def voltage_deflection(neuron, amp=-0.5, dur=500, delay=100):
    stim = h.IClamp(neuron.soma[0](0.5))
    stim.amp = amp
    stim.dur = dur
    stim.delay = delay

    rec = {'t': h.Vector()}
    rec['t'].record(h._ref_t)
    cnt = 0
    h.distance(sec=neuron.soma[0])
    for sec in it.chain(neuron.soma,neuron.basal,neuron.proximal,neuron.distal):
        for seg in sec:
            id = 'v-%d' % cnt
            dst = h.distance(seg.x, sec=sec)
            if sec in neuron.basal:
                dst *= -1
            rec[id] = {'dst': dst, 'rec': h.Vector()}
            rec[id]['rec'].record(seg._ref_v)
            cnt += 1

    CA3.utils.run(tend=stim.dur+stim.delay+100, V0=-70, temperature=36)

    idx = np.where(np.array(rec['t']) < stim.dur + stim.delay)[0][-1]
    distances = []
    voltages = []
    for k,v in rec.iteritems():
        if k != 't':
            distances.append(v['dst'])
            voltages.append(v['rec'][idx])

    distances = np.array(distances)
    voltages = np.array(voltages)
    idx = np.argsort(distances)

    return distances[idx],voltages[idx]

def compute_detailed_neuron_voltage_deflection(filename='../morphologies/DH070613-1-.Edit.scaled.swc'):
    parameters = {'scaling': 1,
                  'soma': {'Cm': 1., 'Ra': 100., 'El': -70., 'Rm': 15e3},
                  'proximal': {'Ra': 100., 'El': -70.},
                  'distal': {'Ra': 100., 'El': -70.},
                  'basal': {'Ra': 100., 'El': -70.},
                  'proximal_limit': 400.,
                  'swc_filename': filename}
    neuron = CA3.cells.SWCNeuron(parameters, with_axon=False, with_active=False)
    distances,voltages = voltage_deflection(neuron)
    return neuron,distances,voltages

def voltage_deflection_error(parameters):
    d = np.round(np.sqrt(np.sum(detailed_neuron['cell'].soma_areas)/np.pi))
    parameters = {'scaling': 1,
                  'soma': {'Cm': 1., 'Ra': parameters['Ra_soma'], 'El': -70., 'Rm': 15e3, 'L': d, 'diam': d},
                  'proximal': {'Ra': parameters['Ra_proximal'], 'El': -70., 'L': np.max(detailed_neuron['cell'].proximal_distances)},
                  'distal': {'Ra': parameters['Ra_distal'], 'El': -70., 
                             'L': np.max(detailed_neuron['cell'].distal_distances)-np.max(detailed_neuron['cell'].proximal_distances)},
                  'basal': {'Ra': parameters['Ra_basal'], 'El': -70., 'L': np.max(detailed_neuron['cell'].basal_distances)}}
    parameters['basal']['diam'] = np.sum(detailed_neuron['cell'].basal_areas) / (np.pi*parameters['basal']['L'])
    parameters['proximal']['diam'] = np.sum(detailed_neuron['cell'].proximal_areas) / (np.pi*parameters['proximal']['L'])
    parameters['distal']['diam'] = np.sum(detailed_neuron['cell'].distal_areas) / (np.pi*parameters['distal']['L'])

    neuron = CA3.cells.SimplifiedNeuron(parameters, with_axon=False, with_active=False)
    distances,voltages = voltage_deflection(neuron)

    err = 0
    for d,v in zip(distances,voltages):
        idx, = np.where((detailed_neuron['distances']>d-5) & (detailed_neuron['distances']<d+5))
        err += (v - np.mean(detailed_neuron['voltages'][idx]))**2

    if DEBUG:
        import pylab as p
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
    n,d,v = compute_detailed_neuron_voltage_deflection()
    global detailed_neuron
    detailed_neuron = {'cell': n, 'distances': d, 'voltages': v}
    
    #err = voltage_deflection_error({'Ra_soma': 135, 'Ra_basal': 704, 'Ra_proximal': 262, 'Ra_distal': 997})
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
