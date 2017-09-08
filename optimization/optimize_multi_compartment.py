#!/usr/bin/env python

import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import bluepyopt as bpop
import bluepyopt.ephys as ephys
import CA3opt

swc_filename = '/Users/daniele/Postdoc/Research/Janelia/SWCs/FINAL/thorny/DH070813-.Edit.scaled.converted.simplified.swc'
cell_name = 'CA3_RS'
filenames = {'morphology': swc_filename, 'parameters': 'parameters_passive.json',
             'features': 'features_passive.json', 'mechanisms': 'mechanisms_passive.json',
             'protocols': 'protocols_passive.json'}

evaluator = CA3opt.evaluator.create('CA3_RS', filenames, simplify_morphology=False)
print(evaluator.cell_model)

optimal_params = {'g_pas.all': 5e-5,
                  'e_pas.all': -70.,
                  'cm.basal': 1.1,
                  'cm.apical': 1.1,
                  'gbar_nas.somatic': 0.1,
                  'ar_nas.somatic': 0.5,
                  'gkdrbar_kdr.somatic': 0.01,
                  'gbar_km.somatic': 0.005}

optimal_response = evaluator.run_protocols(protocols=evaluator.fitness_protocols.values(), param_values=optimal_params)

t = np.array(optimal_response['Step1.soma.v']['time'])
V = np.array(optimal_response['Step1.soma.v']['voltage'])

t0 = 500
idx, = np.where((t>t0) & (t<t0+500))
x = t[idx] - t0
y = V[idx]-V[idx[-1]]
popt,pcov = curve_fit(lambda x,a,tau: a*np.exp(-x/tau), x, y, p0=(y[0],20))

print('RMP: %.3f mV' % V[-1])
print('dV: %.3f mV' % np.min(V))
print('tau: %.3f ms' % popt[1])

#plt.plot(t,V,'k')
#plt.xlabel('Time (ms)')
#plt.ylabel(r'$V_m$ (mV)')
#plt.show()
#sys.exit(0)

optimisation = bpop.optimisations.DEAPOptimisation(evaluator=evaluator,offspring_size=30)
final_pop,hall_of_fame,logs,hist = optimisation.run(max_ngen=50)
 
best_ind = hall_of_fame[0]
best_ind_dict = evaluator.param_dict(best_ind)

print('The best values obtained with the optimization are:')
print('         g_pas.all     = %9.1e S/cm2' % best_ind_dict['g_pas.all'])
print('         e_pas.all     = %9.5f mV' % best_ind_dict['e_pas.all'])
print('            cm.apical  = %9.5f uF/cm2' % best_ind_dict['cm.apical'])
print('            cm.basal   = %9.5f uF/cm2' % best_ind_dict['cm.basal'])
print('      gbar_nas.somatic = %9.1e nS/cm2' % best_ind_dict['gbar_nas.somatic'])
print('        ar_nas.somatic = %9.5f' % best_ind_dict['ar_nas.somatic'])
print('   gkdrbar_kdr.somatic = %9.1e nS/cm2' % best_ind_dict['gkdrbar_kdr.somatic'])
print('       gbar_km.somatic = %9.1e nS/cm2' % best_ind_dict['gbar_km.somatic'])

#### let's simulate the optimal protocol
responses = evaluator.run_protocols(protocols=evaluator.fitness_protocols.values(), param_values=best_ind_dict)

#### let's plot the results
plt.plot(t,V,'r',linewidth=2)
plt.plot(responses['Step1.soma.v']['time'], responses['Step1.soma.v']['voltage'],'k',linewidth=1)
plt.xlabel('Time (ms)')
plt.ylabel(r'$V_m$ (mV)')
plt.axis([0,2000,-90,-50])
plt.show()
