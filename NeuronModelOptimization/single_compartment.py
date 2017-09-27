#!/usr/bin/env python

import sys
from neuron import h
import numpy as np
import matplotlib.pyplot as plt
h.load_file('stdrun.hoc')
h.load_file('stdlib.hoc')
h.celsius = 36

def remove_mechanisms(sec, mechs_to_remove):
    mt = h.MechanismType(0)
    for seg in sec:
        for mech in seg:
            if mech.name() in mechs_to_remove:
                mt.select(mech.name())
                mt.remove(sec=sec)
                print('Removed mechanism %s.' % mech.name())

# membrane resistance
Rm = 20e3    # [Ohm cm2]
# membrane capacitance
Cm = 1      # [uF/cm2]
# membrane time constant
tau = Rm * Cm * 1e-3
# desired input resistance
Rin = 50e6   # [Ohm]
# necessary area
area = Rm/Rin*1e8  # [um2]
print('Area: %.0f um2.' % area)

# create the single compartment model
soma = h.Section()
soma.diam = np.sqrt(area/np.pi)
soma.L = soma.diam
soma.cm = Cm
soma.Ra = 100
print('Radius: %.0f um2.' % (soma.diam/2))

# insert the passive conductance
soma.insert('pas')
soma.e_pas = -70.
soma.g_pas = 1./Rm

# the stimulus: a negative pulse of current
stim = h.IClamp(soma(0.5))
stim.amp = -0.3
stim.dur = 10
stim.delay = 100
# the recorders
rec = {'t': h.Vector(), 'v': h.Vector()}
rec['t'].record(h._ref_t)
rec['v'].record(soma(0.5)._ref_v)

# run the simulation
h.tstop = stim.dur + 3*stim.delay
h.t = 0
h.v_init = soma.e_pas
for key in rec:
    rec[key].resize(0)
h.run()

# plot the results with an estimate of the membrane time constant obtained from the parameters of the neuron
t = np.arange(0,stim.delay,h.dt)
dV = np.min(np.array(rec['v'])) - soma.e_pas
V = soma.e_pas + dV * np.exp(-t/tau)
print('Input resistance: %.0f MOhm.\nMembrane time constant: %.1f ms.\n' % (Rin*1e-6,tau))

plt.figure()
plt.plot(rec['t'],rec['v'],'k',lw=2)
plt.plot(t+stim.dur+stim.delay,V,'r',lw=1)
plt.axis([stim.delay-20, stim.delay+stim.dur+tau*5, soma.e_pas+dV-1, soma.e_pas+1])
plt.xlabel('Time (ms)')
plt.ylabel('Vm (mV)')
#plt.show()

# insert fast sodium and delayed rectifier potassium channels
soma.insert('nas')
soma.insert('kdr')
soma.ek = -80
soma.ena = 55
soma.gbar_nas = 0.01
soma.ar_nas = 0.7
soma.gkdrbar_kdr = 0.003

# run the simulation
stim.dur = 1000
stim.amp = 0.2
h.tstop = stim.dur + 3*stim.delay
h.t = 0
h.v_init = soma.e_pas
for key in rec:
    rec[key].resize(0)
h.run()

# plot the results
plt.figure()
plt.plot(rec['t'],rec['v'],'k',lw=1)
plt.xlabel('Time (ms)')
plt.ylabel(r'$Vm$ (mV)')
#plt.show()

tref = np.array(rec['t'])
Vref = np.array(rec['v'])

# insert muscarinic potassium current responsible for spike frequency adaptation
soma.insert('km')
soma.gbar_km = 0.0003

rec['m_km'] = h.Vector()
rec['m_km'].record(soma(0.5)._ref_m_km)

# run the simulation
h.t = 0
h.v_init = soma.e_pas
for key in rec:
    rec[key].resize(0)
h.run()

Im = soma.gbar_km * np.array(rec['m_km']) * (np.array(rec['v'])-soma.ek)*1e3

# plot the results
plt.figure()
plt.subplot(2,1,1)
plt.plot(tref,Vref,color=[1,.7,.7],lw=0.5,label=r'$V_{ref}$')
plt.plot(rec['t'],rec['v'],'k',lw=1,label=r'$V$')
plt.ylabel(r'$Vm$ (mV)')
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(rec['t'],Im,'k',lw=1)
plt.xlabel('Time (ms)')
plt.ylabel(r'$I_{KM}$ (pA)')
#plt.show()

# insert calcium dynamics
soma.insert('cacum')
soma.eca = 120
soma.cai0_cacum = 50e-6
for seg in soma:
    seg.cacum.depth = min([0.1,seg.diam/2])
soma.tau_cacum = 100

# add a recorder for the internal calcium concentration
rec['cai'] = h.Vector()
rec['ica'] = h.Vector()
rec['cai'].record(soma(0.5)._ref_cai)
rec['ica'].record(soma(0.5)._ref_ica)

# insert low-threshold T-type calcium current
soma.insert('cat')
soma.gcatbar_cat = 0.0003

stim.amp = 0.3
# run the simulation
h.t = 0
h.v_init = soma.e_pas
for key in rec:
    rec[key].resize(0)
h.run()

# plot the results
plt.figure()
plt.subplot(2,1,1)
plt.plot(rec['t'],rec['v'],'k',lw=1)
plt.ylabel(r'$Vm$ (mV)')
plt.subplot(2,1,2)
plt.plot(rec['t'],rec['cai'],'b',lw=1,label=r'$Ca_i$')
plt.plot(rec['t'],rec['ica'],'k',lw=1,label=r'$I_{Ca}$')
plt.legend(loc='best')
plt.xlabel('Time (ms)')
plt.show()
