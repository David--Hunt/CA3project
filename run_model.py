#!/usr/bin/env python

import sys
import itertools as it
import numpy as np
import pylab as p
from neuron import h
h.load_file('stdlib.hoc')
import synapses as syn

def run_synaptic_activation():
    n = CellA(El=-70., Rm=10e3, dend_scaling=0.5, with_synapses=True)
    rate = 2
    tend = 1000
    c = 0.5
    spike_times_c = syn.generate_poisson_spike_times(rate*c, tend*1e-3)[0]
    presynaptic_spike_times = []
    for synapse in n.synapses:
        if np.random.uniform() < 0.: # correlated
            spike_times_i = syn.generate_poisson_spike_times(rate*(1-c), tend*1e-3)[0]
            spike_times = np.sort(np.append(spike_times_i,spike_times_c))
        else: # uncorrelated
            spike_times = syn.generate_poisson_spike_times(rate, tend*1e-3)[0]
        synapse.set_presynaptic_spike_times(spike_times*1e3)
        presynaptic_spike_times.append(spike_times)
    h.topology()
    rec = make_recorders(n)
    for lbl in 'gampa','iampa':
        rec[lbl] = h.Vector()
    rec['gampa'].record(n.synapses[0].syn._ref_g)
    #rec['gnmda'].record(n.synapses[0].syn._ref_gnmda)
    rec['iampa'].record(n.synapses[0].syn._ref_i)
    #rec['inmda'].record(n.synapses[0].syn._ref_inmda)
    nu,edges,count = psth(presynaptic_spike_times,0.05,[0,tend*1e-3])

    run_model(tend)

    #ax = p.subplot(3,1,1)
    hndl = p.figure()
    p.plot(rec['t'],rec['vsoma'],'k',label='Soma')
    p.plot(1e3*(edges[:-1]+edges[1:])/2,nu*10,'m',label='Rate*10')
    #p.plot([0,tend],[rate,rate],'m')
    #p.plot([0,tend],[np.mean(nu)*10,np.mean(nu)*10],'b')
    p.plot(rec['t'],rec['vapical'],'r',label='Apical')
    p.plot(rec['t'],rec['vbasal'],'g',label='Basal')
    p.ylabel('Membrane voltage (mV)')
    p.legend(loc='best')
    #p.subplot(3,1,2,sharex=ax)
    #p.plot(rec['t'],np.array(rec['gampa'])*1e3,'k',label='AMPA')
    #p.plot(rec['t'],np.array(rec['gnmda'])*1e3,'r',label='NMDA')
    #p.legend(loc='best')
    #p.ylabel('Conductance (nS)')
    #p.subplot(3,1,3,sharex=ax)
    #p.plot(rec['t'],np.array(rec['iampa']),'k',label='AMPA')
    #p.plot(rec['t'],np.array(rec['inmda']),'r',label='NMDA')
    #p.legend(loc='best')
    p.xlabel('Time (ms)')
    #p.ylabel('Current (nA)')
    p.savefig('uncorrelated.pdf')
    p.show()

def run_simple_synaptic_activation():
    n = CellA(El=-70., Rm=10e3, dend_scaling=0.5, with_synapses=True)
    for i,synapse in enumerate(n.synapses):
        synapse.set_presynaptic_spike_times([500+i*50])
    rec = make_recorders(n)
    for lbl in 'gampa','iampa':
        rec[lbl] = h.Vector()
    rec['gampa'].record(n.synapses[0].syn._ref_g)
    #rec['gnmda'].record(n.synapses[0].syn._ref_gnmda)
    rec['iampa'].record(n.synapses[0].syn._ref_i)
    #rec['inmda'].record(n.synapses[0].syn._ref_inmda)
    tend = i*50+1000
    run_model(tend)
    #ax = p.subplot(3,1,1)
    p.plot(rec['t'],rec['vsoma'],'k',label='Soma')
    p.plot(rec['t'],rec['vapical'],'r',label='Apical')
    p.plot(rec['t'],rec['vbasal'],'g',label='Basal')
    p.ylabel('Voltage (mV)')
    p.legend(loc='best')
    #p.subplot(3,1,2,sharex=ax)
    #p.plot(rec['t'],np.array(rec['gampa'])*1e3,'k',label='AMPA')
    #p.plot(rec['t'],np.array(rec['gnmda'])*1e3,'r',label='NMDA')
    #p.legend(loc='best')
    #p.ylabel('Conductance (nS)')
    #p.subplot(3,1,3,sharex=ax)
    #p.plot(rec['t'],np.array(rec['iampa']),'k',label='AMPA')
    #p.plot(rec['t'],np.array(rec['inmda']),'r',label='NMDA')
    #p.legend(loc='best')
    p.xlabel('Time (ms)')
    #p.ylabel('Current (nA)')
    p.show()

def simple():
    soma = h.Section()
    soma.insert('pas')
    soma.e_pas = -65
    #soma.g_pas = 1./200e3
    synapse = syn.AMPANMDASynapse(soma, 0.5, 0, 10000)
    synapse.set_presynaptic_spike_times([100])
    h.nmdafactor_AmpaNmda = 0
    rec = {}
    for lbl in 't','vsoma','vapical','vbasal','gampa','gnmda','iampa','inmda':
        rec[lbl] = h.Vector()
    rec['t'].record(h._ref_t)
    rec['vsoma'].record(soma(0.5)._ref_v)
    rec['gampa'].record(synapse.syn._ref_gampa)
    rec['gnmda'].record(synapse.syn._ref_gnmda)
    rec['iampa'].record(synapse.syn._ref_iampa)
    rec['inmda'].record(synapse.syn._ref_inmda)        
    h.load_file('stdrun.hoc')
    h.celsius = 35
    h.cvode_active(1)
    h.cvode.maxstep(10)
    h.tstop = 500
    h.finitialize(soma.e_pas)
    h.run()
    p.subplot(3,1,1)
    p.plot(rec['t'],rec['vsoma'],'k',label='Soma')
    p.subplot(3,1,2)
    p.plot(rec['t'],np.array(rec['gampa'])*1e3,'k',label='AMPA')
    p.plot(rec['t'],np.array(rec['gnmda'])*1e3,'r',label='NMDA')
    p.legend(loc='best')
    p.ylabel('Conductance (nS)')
    p.subplot(3,1,3)
    p.plot(rec['t'],np.array(rec['iampa']),'k',label='AMPA')
    p.plot(rec['t'],np.array(rec['inmda']),'r',label='NMDA')
    p.legend(loc='best')
    p.xlabel('Time (ms)')
    p.ylabel('Current (nA)')
    p.show()

if __name__ == '__main__':
    #simple()
    run_step()
    #run_simple_synaptic_activation()
    #run_synaptic_activation()
