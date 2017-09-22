import neuron
from neuron import h

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pickle 
import math
import nrnutils as nrn

#----------------------------somatic compartment-------------------------

soma = h.Section() # generates somatic compartment

# somatic geometry / properties
soma.L = 30 # somatic length
soma.nseg = 1 # single segment
soma.diam = 30 # somatic diameter
soma.cm = 1 # somatic capacitnace
soma.Ra = 100 # axial resistance

#--------------somatic conductances-----------------
soma.insert ('pas')  # adds leak channel
soma(0.5).pas.g = 0.00025 # leak conductance (S/cm^2)
soma(0.5).pas.e = -65 # leak reversal (mV)
#---------------------------------------------------
soma.insert ('hh2') # adds hodgkin huxley Na & K channels
hh2_mech = soma(0.5).hh2 # refers to hh2 MOD file describing activation & deactivation mechanisms
soma.ek = -80 # potassium reversal
soma.ena = 50 # sodium reversal
hh2_mech.vtraub = -55 # ?
hh2_mech.gnabar = 0.05 # maximum sodium conductance
hh2_mech.gkbar = 0.005 # maximum potassium conductance
#---------------------------------------------------
soma.insert ('im') # adds M current
h.taumax_im = 1000 # maximum decay time constant
im_mech = soma(0.5).im # refers to mechanisms described in Mod file
im_mech.gkbar = 1e-5 # maximum conductance
#---------------------------------------------------
soma.insert ('cad') # adds calcium dynamics (single exponential decay)
soma(0.5).cad.depth = 1
soma(0.5).cad.taur= 5
soma(0.5).cad.cainf = 2.4e-4
soma(0.5).cad.kt = 0
#---------------------------------------------------
soma.insert ('it') # adds T-type calcium channel
soma.cai = 2.4e-4 # internal calcium (mM)
soma.cao = 2 # external calcium (mM)
soma.eca = 120 # calcium reversal (mV)
soma.gcabar_it = 5e-4 # maximum conductance
#---------------------------------------------------
soma.insert ('ical') # adds L-type calcium channel
soma.cai = 2.4e-4 # internal calcium (mM)
soma.cao = 2 # external calcium (mM)
soma.eca = 120 # calcium reversal (mV)
soma.gcabar_ical = 2e-4 # maximum conductance
#---------------------------------------------------
soma.insert ('KahpM95') # adds calcium activated potassium conductance (I_ahp)
soma.cai = 50e-6 # internal calcium [] required for activation?
soma.gbar_KahpM95 = 0.01 # maximum conductance
#---------------------------------------------------
soma.insert ('kd') # adds slowly inactivating potassium conductance (K_d)
soma.ek = -80
soma.gkdbar_kd = 1e-4 # maximum conductance
#---------------------------------------------------
soma.insert ('napinst')
soma.ena = 50
soma.gbar_napinst = 1e-4

#------------------add current clamp recording electrode-----------------------
iclamp = h.IClamp(soma(0.5)) # designates iclamp recording site to middle of soma
iclamp.delay = 250  # delay for iclamp current injection (ms)
iclamp.dur= 500  # duration for iclamp current injection (ms)
iclamp.amp = .125 # amplitude of iclamp current injection 

#---------parameters to recored------->(voltage and time)
vrec = h.Vector()
vrec.record(soma(0.5)._ref_v)
trec =h.Vector()
trec.record(h._ref_t)

h.celsius = 35 # simulation temperature 
h.finitialize(-65) # initialization conditions
neuron.run(1000) # simulation run time (ms)

t = np.array(trec)
v = np.array(vrec)

plt.plot(t,v) # plots voltage over time
plt.show() # shows plot

Rm = 1e2 / (soma(0.5).pas.g*soma.L*soma.diam*math.pi) # (MOhm)

print('')
print('          cell properties          ')
print('-----------------------------------')
print('  length = %.0f um' % soma.L)
print('  diameter = %.0f um' % soma.diam)
print(' input resistance  = %3.0f MOhm' % Rm)
print('')

#I,f = nrn.computefIcurve(soma(0.5),[.1, .5, .05], 500, 250)
#plt.plot(I,f)
#plt.show() 


