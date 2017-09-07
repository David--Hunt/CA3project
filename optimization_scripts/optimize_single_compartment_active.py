#!/usr/bin/env python

import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from neuron import h
import bluepyopt as bpop
import bluepyopt.ephys as ephys

#import logging
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

h.load_file('stdrun.hoc')
h.cvode_active(1)
h.celsius = 36

PLOT = False

def plot_responses(resp):
    plt.figure()
    for name,tv in resp.iteritems():
        plt.plot(tv['time'],tv['voltage'],label=name.split('.')[0],lw=1)
    plt.legend(loc='best')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$V_m$ (mV)')
    plt.show()

# membrane resistance
Rm = 20e3                 # [Ohm cm2]
# membrane capacitance
cm = 1                    # [uF/cm2]
# membrane time constant
tau = Rm * cm * 1e-3      # [ms] theoretical value
tau = 19.2                # [ms] measured value
# desired input resistance
Rin = 500                 # [MOhm]
# necessary area
area = Rm/Rin*1e2         # [um2]
area_cm = area*1e-8       # [cm2]
# necessary radius assuming a cylindrical section
radius = np.sqrt(area/np.pi)/2
# leak conductance reversal
El = -70.                 # [mV]
# axial resistance
Ra = 100.                 # [Ohm cm]
# hyperpolarizing step of current
dI = -0.02                # [nA]
# expected deflection
dV = Rin * dI             # [mV]
# membrane time constant
tau = Rm*cm*1e-3          # [ms]
# current necessary to have 1 spike
rheobase = 0.02 # 0.0163
# how many times the rheobase should be multiplied to have 5 spikes
n_rheobase = 0.025/rheobase # 0.0175/rheobase

swc_file = 'single_compartment.swc'

param_values = {'Ra': Ra, 'cm': cm, 'ena': 55., 'ek': -80.}
optimal_params = {'g_pas': 1/Rm, 'e_pas': El,
                  'gbar_nas': 0.01, 'ar_nas': 0.7,
                  'gkdrbar_kdr': 0.003,
                  'gbar_km': 0.0003}
param_bounds = {'g_pas': [(1./Rm)/2,(1./Rm)*2], 'e_pas': [El-3.,El+3.],
                'ar_nas': [0,1], 'gbar_nas': [optimal_params['gbar_nas']/2,optimal_params['gbar_nas']*2],
                'gkdrbar_kdr': [optimal_params['gkdrbar_kdr']/2,optimal_params['gkdrbar_kdr']*2],
                'gbar_km': [optimal_params['gbar_km']/2,optimal_params['gbar_km']*2]}

def run_optimal_model():
    # create the single compartment model
    soma = h.Section()
    soma.diam = radius*2
    soma.L = soma.diam
    soma.cm = cm
    soma.Ra = Ra
    print('Area: %.0f um2.' % area)
    print('Radius: %.0f um2.' % radius)

    with open(swc_file,'w') as fid:
        fid.write('1 1 -%.2f 0.0 0.0 %.2f -1\n' % (radius,radius))
        fid.write('2 1 0.0 0.0 0.0 %.2f -1\n' % radius)
        fid.write('3 1 %.2f 0.0 0.0 %.2f -1\n' % (radius,radius))
        
    # insert the passive conductance
    soma.insert('pas')
    soma.e_pas = optimal_params['e_pas']
    soma.g_pas = 1./Rm
    # insert fast sodium and delayed rectifier potassium channels
    soma.insert('nas')
    soma.ena = param_values['ena']
    soma.gbar_nas = optimal_params['gbar_nas']
    soma.ar_nas = optimal_params['ar_nas']
    soma.insert('kdr')
    soma.ek = param_values['ek']
    soma.gkdrbar_kdr = optimal_params['gkdrbar_kdr']
    soma.insert('km')
    soma.gbar_km = optimal_params['gbar_km']
    
    # the stimulus: a negative pulse of current
    stim = h.IClamp(soma(0.5))
    stim.amp = dI
    stim.dur = 1000
    stim.delay = 250
    
    # the recorders
    rec = {'t': h.Vector(), 'v': h.Vector(), 'tspikes': h.Vector()}
    rec['t'].record(h._ref_t)
    rec['v'].record(soma(0.5)._ref_v)
    apc = h.APCount(soma(0.5))
    apc.record(rec['tspikes'])
    
    # run the simulation
    h.tstop = stim.dur + 2*stim.delay
    h.t = 0
    h.v_init = soma.e_pas
    for key in rec:
        rec[key].resize(0)
    h.run()
    
    # plot the results with an estimate of the membrane time constant obtained from the parameters of the neuron
    T = np.array(rec['t'])
    idx = np.where((T > stim.delay+stim.dur) & (T < stim.delay+stim.dur+200))
    t = T[idx] - (stim.delay+stim.dur)
    V = np.array(rec['v'])[idx]
    offset = V[-1]
    V = offset - V
    popt,pcov = curve_fit(lambda x,a,tau: a*np.exp(-x/tau), t, V, p0=(V[0],20))
    V = offset - popt[0]*np.exp(-t/popt[1])
    print('Input resistance: %.0f MOhm.\nMembrane time constant: %.1f ms.\n' % (Rin,popt[1]))

    if PLOT:
        plt.figure()
        plt.plot(rec['t'],rec['v'],'k',lw=1)
        plt.plot(t+stim.dur+stim.delay,V,'r',lw=1)

    # run the simulation
    nspikes = []
    for amp,col in [(rheobase,'b'),(n_rheobase*rheobase,'m')]:
        apc.n = 0
        stim.amp = amp
        h.t = 0
        h.v_init = soma.e_pas
        for key in rec:
            rec[key].resize(0)
        h.run()
        nspikes.append(apc.n)
        if PLOT:
            # plot the results
            plt.plot(rec['t'],rec['v'],col,lw=1)
            plt.xlabel('Time (ms)')
            plt.ylabel(r'$Vm$ (mV)')

    if PLOT:
        plt.show()

    return nspikes


nspikes = run_optimal_model()

print('The area of the cell is %.1f um2.' % area)
print('Will inject a step of current of %.0f pA.' % (dI*1e3))
print('The goal is to have:')
print('   1) a resting membrane potential of %.1f mV.' % El)
print('   2) a deflection of %.2f mV, which corresponds to an input resistance of %.1f MOhm.' % (dV,Rin))
print('   3) a membrane time constant of %.1f ms.' % tau)
print('   4) %d spike in response to a 1 s-long step of %.1f pA.' % (nspikes[0],rheobase*1e3))
print('   5) %d spikes in response to a 1 s-long step of %.1f pA.' % (nspikes[1],n_rheobase*rheobase*1e3))

print('The optimal values of the parameters are:')
print('        g_pas: %.1e S/cm2' % optimal_params['g_pas'])
print('        e_pas: %.1f mV' % optimal_params['e_pas'])
print('     gbar_nas: %.5f S/cm2' % optimal_params['gbar_nas'])
print('       ar_nas: %.3f' % optimal_params['ar_nas'])
print('  gkdrbar_kdr: %.5f S/cm2' % optimal_params['gkdrbar_kdr'])
print('      gbar_km: %.5f S/cm2' % optimal_params['gbar_km'])

morph = ephys.morphologies.NrnFileMorphology(swc_file)
# other available names are axonal, apical and basal.
somatic_loc = ephys.locations.NrnSeclistLocation('somatic', seclist_name='somatic')

# let's create the mechanisms
mechanisms = {}
for name in ['pas','nas','kdr','km']:
    mechanisms[name] = ephys.mechanisms.NrnMODMechanism(name=name,suffix=name,locations=[somatic_loc])

# let's create parameters
parameters = {}
for name,value in param_values.iteritems():
    parameters[name] = ephys.parameters.NrnSectionParameter(name=name,param_name=name,value=value,
                                                            locations=[somatic_loc],frozen=True)
for name,bounds in param_bounds.iteritems():
    parameters[name] = ephys.parameters.NrnSectionParameter(name=name,param_name=name,bounds=bounds,
                                                            locations=[somatic_loc],frozen=False)

simple_cell = ephys.models.CellModel(name='simple_cell',morph=morph,mechs=mechanisms.values(),
        params=parameters.values())

print('')
print(simple_cell)

### let's create the stimulation protocol
# where current will be injected
soma_loc = ephys.locations.NrnSeclistCompLocation(name='soma',seclist_name='somatic',sec_index=0,comp_x=0.5)

sweep_protocols = []
for name,amp in [('hyperpolarizing',dI),('rheobase',rheobase),('above_rheobase',n_rheobase*rheobase)]:
    stim = ephys.stimuli.NrnSquarePulse(step_amplitude=amp,step_delay=250,step_duration=1000,
                                        location=soma_loc,total_duration=1500)
    rec = ephys.recordings.CompRecording(name='%s.soma.v'%name,location=soma_loc,variable='v')
    sweep_protocols.append(ephys.protocols.SweepProtocol(name, [stim], [rec]))
steps_protocol = ephys.protocols.SequenceProtocol('steps', protocols=sweep_protocols)

nrn = ephys.simulators.NrnSimulator()

if PLOT:
    responses = steps_protocol.run(cell_model=simple_cell, param_values=optimal_params, sim=nrn)
    plot_responses(responses)

efel_features = {'hyperpolarizing': {'voltage_base': {'mean': El, 'std': 1.},
                                     'steady_state_voltage_stimend': {'mean': El+dV, 'std': 1.},
                                     'decay_time_constant_after_stim': {'mean': tau, 'std': 1.}},
                 'rheobase': {'Spikecount': {'mean': nspikes[0], 'std': 0.05*nspikes[0]}},
                 'above_rheobase': {'Spikecount': {'mean': nspikes[1], 'std': 0.05*nspikes[1]}}}

objectives = []

for protocol in sweep_protocols:
    stim_start = protocol.stimuli[0].step_delay
    stim_end = stim_start + protocol.stimuli[0].step_duration
    for efel_feature_name,pars in efel_features[protocol.name].items():
        feature_name = '%s.%s' % (protocol.name, efel_feature_name)
        feature = ephys.efeatures.eFELFeature(feature_name,efel_feature_name=efel_feature_name,
                                              recording_names={'': '%s.soma.v' % protocol.name},
                                              stim_start=stim_start,stim_end=stim_end,
                                              exp_mean=pars['mean'],exp_std=pars['std'])
        objective = ephys.objectives.SingletonObjective(feature_name,feature)
        objectives.append(objective)

score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)

cell_evaluator = ephys.evaluators.CellEvaluator(cell_model=simple_cell,
                                                param_names=param_bounds.keys(),
                                                fitness_protocols={steps_protocol.name: steps_protocol},
                                                fitness_calculator=score_calc,sim=nrn)

print(cell_evaluator.evaluate_with_dicts(optimal_params))

optimisation = bpop.optimisations.DEAPOptimisation(evaluator=cell_evaluator,offspring_size=50)

final_pop,hall_of_fame,logs,hist = optimisation.run(max_ngen=20)

best_ind = hall_of_fame[0]
best_ind_dict = cell_evaluator.param_dict(best_ind)

print(cell_evaluator.evaluate_with_dicts(best_ind_dict))

best_tau = param_values['cm']/best_ind_dict['g_pas']*1e-3
best_Rin = 1e-6/(best_ind_dict['g_pas']*area_cm)

print('The best values obtained with the optimization are:')
print('        e_pas: %.1f mV' % best_ind_dict['e_pas'])
print('        g_pas: %.1e S/cm2' % best_ind_dict['g_pas'])
print('     gbar_nas: %.5f S/cm2' % best_ind_dict['gbar_nas'])
print('       ar_nas: %.3f' % best_ind_dict['ar_nas'])
print('  gkdrbar_kdr: %.5f S/cm2' % best_ind_dict['gkdrbar_kdr'])
print('      gbar_km: %.5f S/cm2' % best_ind_dict['gbar_km'])
print('which give the following values:')
print('          RMP: %.1f mV' % best_ind_dict['e_pas'])
print('          Rin: %.1f MOhm' % best_Rin)
print('          tau: ~ %.1f ms' % best_tau)

#### let's simulate the optimal protocol and plot the results
responses = steps_protocol.run(cell_model=simple_cell, param_values=best_ind_dict, sim=nrn)
plot_responses(responses)


