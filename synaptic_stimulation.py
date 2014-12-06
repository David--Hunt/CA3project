#!/usr/bin/env python

import sys
import itertools as it
import numpy as np
import pylab as p
from neuron import h
h.load_file('stdlib.hoc')
import synapses as syn

def psth(spks, binwidth, interval=None):
    if interval is None:
        interval = [min([t[0] for t in spks]), max([t[-1] for t in spks])]
    edges = np.arange(interval[0],interval[1]+binwidth/2,binwidth)
    ntrials = len(spks)
    count = np.zeros((ntrials,len(edges)-1))
    for i,t in enumerate(spks):
        count[i,:] = np.histogram(t,edges)[0]
    if ntrials > 1:
        nu = np.sum(count,0) / (ntrials*binwidth)
    else:
        nu = count / (ntrials*binwidth)
    return nu,edges,count

class CellA:
    def __init__(self, El, Rm):
        self.make_sections()
        self.setup_topology()
        self.compute_nseg()
        self.connect_sections()
        self.compute_total_area()
        self.insert_passive_mech(El, Rm)
        self.insert_active_mech()
        self.add_synapses()

    def make_sections(self):
        self.soma = [h.Section(name='soma')]
        self.basal = [h.Section(name='basal-%d' % i) for i in range(2)]
        self.apical = [h.Section(name='apical-%d' % i) for i in range(4)]
        self.axon = [h.Section(name='axon-%d' % i) for i in range(5)]
        for sec in h.allsec():
            sec.cm = 1
            sec.Ra = 200

    def setup_topology(self):
        self.soma[0].L = 20
        self.soma[0].diam = 20
        self.apical[0].L = 500
        self.apical[0].diam = 10
        for i in range(1,4):
            self.apical[i].L = 200
            self.apical[i].diam = 5
        for d in self.basal:
            d.L = 300
            d.diam = 10
        for i in range(len(self.axon)):
            self.axon[i].L = 20
            self.axon[i].diam = 1

    def compute_nseg(self):
        for sec in h.allsec():
            sec.nseg = int((sec.L/(0.1*h.lambda_f(100))+0.9)/2)*2 + 1
            print('%s has %d segments.' % (h.secname(sec=sec),sec.nseg))

    def connect_sections(self):
        self.apical[0].connect(self.soma[0], 1, 0)
        for i in range(1,4):
            self.apical[i].connect(self.apical[0], 1, 0)
        for d in self.basal:
            d.connect(self.soma[0], 0, 0)
        self.axon[0].connect(self.soma[0], 0, 0)
        for i in range(1,len(self.axon)):
            self.axon[i].connect(self.axon[i-1], 1, 0)
        
    def compute_total_area(self):
        self.total_area = 0
        for sec in h.allsec():
            for seg in sec:
                self.total_area += h.area(seg.x, sec)
        print('Total area: %.0f um^2.' % self.total_area)

    def insert_passive_mech(self, El, Rm):
        for sec in h.allsec():
            sec.insert('pas')
            sec.e_pas = El
            sec.g_pas = 1./Rm

    def insert_active_mech(self):
        for sec in h.allsec():
            sec.insert('hh2')
            if sec in self.soma or sec in self.axon:
                sec.gnabar_hh2 = 0.05
                sec.gkbar_hh2 = 0.005
                if sec is self.axon[0]:
                    sec.gnabar_hh2 = 0.25

    def add_synapses(self):
        #wgt = 0.0035
        wgt = 10000
        n = 10
        self.synapses = []
        self.somatic_synapses = []
        self.basal_synapses = []
        self.apical_synapses = []
        for sec in self.soma:
            for seg in sec:
                for i in range(n):
                    self.somatic_synapses.append(syn.AMPANMDASynapse(sec, seg.x, 0, wgt))
                    self.synapses.append(self.somatic_synapses[-1])
        for sec in self.basal:
            for seg in sec:
                for i in range(n):
                    self.basal_synapses.append(syn.AMPANMDASynapse(sec, seg.x, 0, wgt))
                    self.synapses.append(self.basal_synapses[-1])
        for sec in self.apical:
            for seg in sec:
                for i in range(n):
                    self.apical_synapses.append(syn.AMPANMDASynapse(sec, seg.x, 0, wgt))
                    self.synapses.append(self.apical_synapses[-1])
        print('Added %d synapses.' % len(self.synapses))

class CellB (CellA):
    def __init__(self, El, Rm):
        CellA.__init__(self, El, Rm)

    def setup_topology(self):
        h.pt3dadd(0, 0, 0, 20, sec=self.soma[0])
        h.pt3dadd(0, 0, 10, 20, sec=self.soma[0])

        h.pt3dadd(0, 0, 10, 10, sec=self.apical[0])
        h.pt3dadd(0, 0, 510, 7, sec=self.apical[0])

        h.pt3dadd(0, 0, 510, 7, sec=self.apical[1])
        h.pt3dadd(0, -140, 650, 5, sec=self.apical[1])

        h.pt3dadd(0, 0, 510, 7, sec=self.apical[2])
        h.pt3dadd(0, 0, 710, 5, sec=self.apical[2])

        h.pt3dadd(0, 0, 510, 7, sec=self.apical[3])
        h.pt3dadd(0, 140, 650, 5, sec=self.apical[3])

        h.pt3dadd(0, 0, 0, 10, sec=self.basal[0])
        h.pt3dadd(0, -212, -212, 7, sec=self.basal[0])

        h.pt3dadd(0, 0, 0, 10, sec=self.basal[1])
        h.pt3dadd(0, 212, -212, 7, sec=self.basal[1])

        h.pt3dadd(0, 0, 0, 5, sec=self.axon[0])
        h.pt3dadd(0, 0, -10, 3, sec=self.axon[0])
        h.pt3dadd(0, 0, -10, 5, sec=self.axon[1])
        h.pt3dadd(0, 0, -20, 1, sec=self.axon[1])
        
        for i in range(2,len(self.axon)):
            h.pt3dadd(0, 0, -20-(i-2)*20, 1, sec=self.axon[i])
            h.pt3dadd(0, 0, -20-(i-1)*20, 1, sec=self.axon[i])


def main():
    n = CellA(El=-65., Rm=100e3)

    rate = 1
    tend = 5000
    c = 0.05
    spike_times_c = syn.generate_poisson_spike_times(rate*c, tend*1e-3)[0]

    presynaptic_spike_times = []
    for synapse in n.synapses:
        if np.random.uniform() < 0.5: # correlated
            spike_times_i = syn.generate_poisson_spike_times(rate*(1-c), tend*1e-3)[0]
            jitter = 0.01*np.random.normal(size=len(spike_times_c))
            spike_times = np.sort(np.append(spike_times_i,spike_times_c+jitter))
            spike_times = spike_times[spike_times>0]
        else: # uncorrelated
            spike_times,isi = syn.generate_poisson_spike_times(rate, tend*1e-3)
        synapse.set_presynaptic_spike_times(spike_times*1e3)
        presynaptic_spike_times.append(spike_times)

    h.topology()

    stim = h.IClamp(n.soma[0](0.5))
    stim.delay = 500
    stim.amp = 0.0
    stim.dur = 1000

    rec = {}
    for lbl in 't','vsoma','vapical','vbasal','gampa','gnmda','iampa','inmda':
        rec[lbl] = h.Vector()
        rec['t'].record(h._ref_t)
        rec['vsoma'].record(n.soma[0](0.5)._ref_v)
        rec['vapical'].record(n.apical[2](0.5)._ref_v)
        rec['vbasal'].record(n.basal[0](0.5)._ref_v)
        rec['gampa'].record(n.synapses[0].syn._ref_gampa)
        rec['gnmda'].record(n.synapses[0].syn._ref_gnmda)
        rec['iampa'].record(n.synapses[0].syn._ref_iampa)
        rec['inmda'].record(n.synapses[0].syn._ref_inmda)
        
    h.load_file('stdrun.hoc')
    h.celsius = 35
    h.cvode_active(1)
    h.cvode.maxstep(10)
    h.tstop = tend
    h.run()

    nu,edges,count = psth(presynaptic_spike_times,0.05,[0,tend*1e-3])

    p.subplot(3,1,1)
    p.plot(rec['t'],rec['vsoma'],'k',label='Soma')
#p.plot(1e3*(edges[:-1]+edges[1:])/2,nu,'r')
#p.plot([0,tend],[rate,rate],'m')
#p.plot([0,tend],[np.mean(nu),np.mean(nu)],'b')
#p.plot(rec['t'],rec['vapical'],'r',label='Apical')
#p.plot(rec['t'],rec['vbasal'],'g',label='Basal')
#p.legend(loc='best')
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
    simple()
