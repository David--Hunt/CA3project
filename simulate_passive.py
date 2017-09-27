#!/usr/bin/env python

import os
import sys
import argparse as arg

def main():
    parser = arg.ArgumentParser(description='Simulate a passive neuron model described by an SWC morphology file.')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('--el', default=-70., type=float, help='Leak reversal potential')
    parser.add_argument('--gl', default=3e-5, type=float, help='Leak conductance')
    parser.add_argument('--ra', default=1.0, type=float, help='Axial resistance')
    parser.add_argument('--cm', default=1.0, type=float, help='Somatic and axonal capacitance')
    parser.add_argument('--cm-basal', default=2.0, type=float, help='Basal dendrite capacitance')
    parser.add_argument('--cm-apical', default=2.0, type=float, help='Apical dendrite capacitance')
    args = parser.parse_args(args=sys.argv[1:])

    if not os.path.isfile(args.swc_file):
        print('%s: %s: no such file.' % (os.path.basename(sys.argv[0]),args.swc_file))
        sys.exit(0)
        
    swc_filename = args.swc_file
    #cell_name = os.path.basename(swc_filename).split('-.')[0].replace('-','_')
        
    dI = -0.1                 # [nA]
    Ra = args.ra
    cm = args.cm
    cm_apical = args.cm_apical
    cm_basal = args.cm_basal
    El = args.el
    gl = args.gl

    import bluepyopt.ephys as ephys

    morph = ephys.morphologies.NrnFileMorphology(swc_filename)
    
    locations = []
    for loc in ('somatic','axonal','apical','basal'):
        locations.append(ephys.locations.NrnSeclistLocation(loc, seclist_name=loc))

    # let's create a passive mechanism
    pas_mech = ephys.mechanisms.NrnMODMechanism(name='pas',suffix='pas',locations=locations)

    # let's create the parameters
    Ra_param = ephys.parameters.NrnSectionParameter(name='Ra',param_name='Ra',value=Ra,
                                                    locations=locations,frozen=True)
    cm_param = ephys.parameters.NrnSectionParameter(name='cm',param_name='cm',value=cm,
                                                    locations=locations[:2],frozen=True)
    cm_apical_param = ephys.parameters.NrnSectionParameter(name='cm.apical',param_name='cm',value=cm_apical,
                                                           locations=[locations[2]],frozen=True)
    cm_basal_param = ephys.parameters.NrnSectionParameter(name='cm.basal',param_name='cm',value=cm_basal,
                                                          locations=[locations[3]],frozen=True)
    gpas_param = ephys.parameters.NrnSectionParameter(name='g_pas',param_name='g_pas',value=gl,
                                                      locations=locations,frozen=True)
    El_param = ephys.parameters.NrnSectionParameter(name='e_pas',param_name='e_pas',value=El,
                                                    locations=locations,frozen=True)

    cell = ephys.models.CellModel(name='cell',morph=morph,mechs=[pas_mech],
                                  params=[Ra_param, cm_param, cm_basal_param, cm_apical_param, gpas_param, El_param])

    print(cell)

    ### let's create the stimulation protocol
    # where current will be injected
    soma_loc = ephys.locations.NrnSeclistCompLocation(name='soma',seclist_name='somatic',sec_index=0,comp_x=0.5)

    stim = ephys.stimuli.NrnSquarePulse(step_amplitude=dI,step_delay=1000,step_duration=1000,
                                        location=soma_loc,total_duration=3000)
    rec = ephys.recordings.CompRecording(name='step.soma.v',location=soma_loc,variable='v')
    step_protocols = ephys.protocols.SequenceProtocol('step', protocols=[ephys.protocols.SweepProtocol('step', [stim], [rec])])

    nrn = ephys.simulators.NrnSimulator()
    nrn.neuron.h.load_file('stdrun.hoc')
    nrn.neuron.h.cvode_active(1)
    nrn.neuron.h.celsius = 34
    
    #### let's simulate the optimal protocol
    responses = step_protocols.run(cell_model=cell, param_values={}, sim=nrn)

    import numpy as np
    from scipy.optimize import curve_fit

    t = np.array(responses['step.soma.v']['time'])
    V = np.array(responses['step.soma.v']['voltage'])
    idx, = np.where((t >= 1000) & (t<1200))
    x = t[idx] - t[idx[0]]
    y = V[idx] - V[idx[-1]]
    popt,pcov = curve_fit(lambda x,a,tau: a*np.exp(-x/tau), x, y, p0=(x[0],10))
    print('Rin = %f MOhm\nTau = %f ms' % ((np.min(V)-V[idx[0]-1])/dI,popt[1]))

    #### let's plot the results
    import matplotlib.pyplot as plt
    plt.plot(t,V,'k')
    plt.plot(t[idx],V[idx],'r')
    plt.plot(t[idx],popt[0]*np.exp(-x/popt[1])+V[idx[-1]],'g')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$V_m$ (mV)')
    plt.show()

if __name__ == '__main__':
    main()
    


