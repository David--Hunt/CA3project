#!/usr/bin/env python

from neuron import h
import SWC_neuron as swc
import numpy as np

h.load_file('stdrun.hoc')

filename = '../morphologies/DH070313-.Edit.scaled.swc'
filename = '../morphologies/DH070613-1-.Edit.scaled.swc'
Rm = {'axon': 100, 'soma': 150, 'dend': 75}
Ra = {'axon': 100, 'soma': 75, 'dend': 75}
Cm = {'axon': 1, 'soma': 1, 'dend': 1}
cell = swc.RSCell(filename,Rm,Ra,Cm,5.)

stim = h.IClamp(cell.soma[0](0.5))
stim.dur = 2000.
stim.delay = 500.

rec = {}
for lbl in 't','vsoma','spikes':
    rec[lbl] = h.Vector()
rec['t'].record(h._ref_t)
rec['vsoma'].record(cell.soma[0](0.5)._ref_v)
apc = h.APCount(cell.soma[0](0.5))
apc.record(rec['spikes'])

amplitudes = np.append(np.arange(-0.3,0.31,0.05),np.arange(0.4,1.51,0.1))

h.celsius = 37.
h.dt = 0.02
Vrest = -73.

spike_times = []

for i,amp in enumerate(amplitudes):
    apc.n = 0
    stim.amp = amp

    h.v_init = Vrest
    h.finitialize(Vrest)
    h.fcurrent()
    h.t = 0.
    h.tstop = stim.delay + stim.dur + 500.
    count = 0
    while h.t < h.tstop:
        h.fadvance()
        if count%1000 == 0:
            print h.t
        count += 1

    if i == 0:
        V = np.zeros((len(amplitudes),len(np.array(rec['vsoma']))))
    V[i,:] = np.array(rec['vsoma'])
    spike_times.append(np.array(rec['spikes']))


import h5utils as h5
h5.save_h5_file('steps.h5', dt=h.dt, V=V, spike_times=spike_times, \
                    swc_file=filename, amplitudes=amplitudes, Rm=Rm, Ra=Ra, Cm=Cm, \
                    stimulus={'dur': stim.dur, 'delay': stim.delay}, temperature=h.celsius)
