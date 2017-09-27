#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt

from neuron import h
import bluepyopt as bpop
import bluepyopt.ephys as ephys
from bluepyopt.ephys.morphologies import NrnFileMorphology
import dlopt

h.load_file('stdrun.hoc')
h.cvode_active(1)
h.celsius = 34

dI = -0.1                 # [nA]
Rm = 20e3                 # [Ohm cm2]
cm = 1.                   # [uF/cm2]
Ra = 100.                 # [Ohm cm]
El = -70.                 # [mV]
Rin = 100                 # [MOhm]
dV = Rin * dI             # [mV]
tau = Rm*cm*1e-3          # [ms]

swc_filename = '/Users/daniele/Postdoc/Research/Janelia/SWCs/FINAL/thorny/DH070813-.Edit.scaled.converted.simplified.swc'
# morph = dlopt.morphology.SWCFileSimplifiedMorphology(swc_filename)
morph = ephys.morphologies.NrnFileMorphology(swc_filename)
    
locations = []
for loc in ('somatic','axonal','apical','basal'):
    locations.append(ephys.locations.NrnSeclistLocation(loc, seclist_name=loc))

# let's create a passive mechanism
pas_mech = ephys.mechanisms.NrnMODMechanism(name='pas',suffix='pas',locations=locations)

# let's create parameters
Ra_param = ephys.parameters.NrnSectionParameter(name='Ra',param_name='Ra',value=Ra,
                                                locations=locations,frozen=True)
cm_param = ephys.parameters.NrnSectionParameter(name='cm',param_name='cm',bounds=[cm/2,cm*2],
                                                locations=locations,frozen=False)
gpas_param = ephys.parameters.NrnSectionParameter(name='g_pas',param_name='g_pas',bounds=[(1./Rm)/2,(1./Rm)*2],
                                                locations=locations,frozen=False)
El_param = ephys.parameters.NrnSectionParameter(name='e_pas',param_name='e_pas',bounds=[El-10.,El+10.],
                                                locations=locations,frozen=False)

CA3_cell = ephys.models.CellModel(name='CA3_cell',morph=morph,mechs=[pas_mech],
        params=[Ra_param, cm_param, gpas_param, El_param])

print(CA3_cell)

### let's create the stimulation protocol
sweep_protocols = []
# where current will be injected
soma_loc = ephys.locations.NrnSeclistCompLocation(name='soma',seclist_name='somatic',sec_index=0,comp_x=0.5)

stim = ephys.stimuli.NrnSquarePulse(step_amplitude=dI,step_delay=500,step_duration=1000,
                                    location=soma_loc,total_duration=2000)
rec = ephys.recordings.CompRecording(name='step.soma.v',location=soma_loc,variable='v')
sweep_protocols.append(ephys.protocols.SweepProtocol('step', [stim], [rec]))
step_protocols = ephys.protocols.SequenceProtocol('step', protocols=sweep_protocols)

efel_features = {'step': {'voltage_base': {'mean': El, 'std': 1.},
                          'steady_state_voltage_stimend': {'mean': El+dV, 'std': 1.},
                          'decay_time_constant_after_stim': {'mean': tau, 'std': 1.}}}

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

nrn = ephys.simulators.NrnSimulator()
cell_evaluator = ephys.evaluators.CellEvaluator(cell_model=CA3_cell,
                                                param_names=['cm', 'g_pas', 'e_pas'],
                                                fitness_protocols={step_protocols.name: step_protocols},
                                                fitness_calculator=score_calc,sim=nrn)

optimal_params = {'cm': cm, 'g_pas': 1/Rm, 'e_pas': El}
optimisation = bpop.optimisations.DEAPOptimisation(evaluator=cell_evaluator,offspring_size=50)

final_pop,hall_of_fame,logs,hist = optimisation.run(max_ngen=50)

best_ind = hall_of_fame[0]
best_ind_dict = cell_evaluator.param_dict(best_ind)

print('The best values obtained with the optimization are:')
print('   cm: %.1f uF/cm2' % best_ind_dict['cm'])
print('   g_pas: %.1e S/cm2' % best_ind_dict['g_pas'])
print('   e_pas: %.1f mV' % best_ind_dict['e_pas'])
print('which give the following values:')
print('   RMP: %.1f mV' % best_ind_dict['e_pas'])

#### let's simulate the optimal protocol
responses = step_protocols.run(cell_model=CA3_cell, param_values=best_ind_dict, sim=nrn)

#### let's plot the results
plt.plot(responses['step.soma.v']['time'], responses['step.soma.v']['voltage'])
plt.xlabel('Time (ms)')
plt.ylabel(r'$V_m$ (mV)')
plt.show()

