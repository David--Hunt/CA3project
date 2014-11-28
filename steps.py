#!/usr/bin/env python

import sys
import time
import argparse as arg
import numpy as np
from neuron import h
import SWC_neuron as swc
import h5utils as h5

def timestamp():
    return time.strftime('%b %d, %H:%M:%S ', time.localtime(time.time()))

def run_current_steps_protocol(cell, amplitudes, ttran, tstep, dt=0.025, temperature=37., use_cvode=False):
    h.load_file('stdrun.hoc')

    print(timestamp() + '>> Inserting the stimulus...')
    stim = h.IClamp(cell.soma[0](0.5))
    stim.dur = tstep
    stim.delay = ttran

    print(timestamp() + '>> Setting up the recorders...')
    rec = {}
    for lbl in 't','vsoma','spikes':
        rec[lbl] = h.Vector()
    rec['t'].record(h._ref_t)
    rec['vsoma'].record(cell.soma[0](0.5)._ref_v)
    apc = h.APCount(cell.soma[0](0.5))
    apc.record(rec['spikes'])

    h.celsius = temperature
    h.dt = dt

    if use_cvode:
        h.tstop = stim.delay + stim.dur + 500
        h.cvode_active(1)
        h.cvode.atol(1e-6)
        h.cvode.rtol(1e-6)
        h.cvode.maxstep(dt)
    else:
        h.tstop = stim.delay - 20
        print(timestamp() + '>> Evolving the model until %g ms...' % h.tstop)
        sys.stdout.flush()
        h.run()
        print(timestamp() + '>> Saving the state...')
        ss = h.SaveState()
        ss.save()

    t = []
    V = []
    spike_times = []

    for i,amp in enumerate(amplitudes):    
        sys.stdout.write('\r' + timestamp() + '>> Trial [%02d/%02d] ' % (i+1,len(amplitudes)))
        sys.stdout.flush()
        if not use_cvode:
            ss.restore()
        stim.amp = amp
        apc.n = 0
        rec['t'].resize(0)
        rec['vsoma'].resize(0)
        rec['spikes'].resize(0)
        if use_cvode:
            h.t = 0
            h.run()
        else:
            h.continuerun(stim.delay + stim.dur + 500.)
        t.append(np.array(rec['t']))
        V.append(np.array(rec['vsoma']))
        spike_times.append(np.array(rec['spikes']))
    sys.stdout.write('\n')

    if use_cvode:
        return np.array(t), np.array(V), np.array(spike_times)
    else:
        return np.array(V), np.array(spike_times)

def compute_input_resistance(t, V, I, tonset, stimdur, ttran):
    n = len(I)
    Vss = np.zeros(n)
    for i in range(n):
        idx, = np.where((t[i] > tonset+ttran) & (t[i] < tonset+stimdur))
        Vss[i] = np.mean(V[i][idx])
    pp = np.polyfit(I,Vss,1)
    return pp[0],pp[1]

def main():
    #filename = '../morphologies/DH070313-.Edit.scaled.swc'
    #filename = '../morphologies/DH070613-1-.Edit.scaled.swc'
    #amplitudes = np.append(np.arange(-0.3,0.31,0.05),np.arange(0.4,1.51,0.1))

    parser = arg.ArgumentParser(description='Inject steps of current in a model neuron.')
    parser.add_argument('filename', type=str, action='store', help='Path of the file containing the morphology')
    parser.add_argument('-a', '--amplitudes', action='append', nargs=3, type=float,
                        help='Amplitudes of the injected current in the form MIN MAX STEP')
    parser.add_argument('-s', '--simplify', default=5., type=float,
                        help='Minimum distance between nodes after simplification of the morphology (default 5 um)')
    parser.add_argument('--ttran', default=500., type=float,
                        help='Transient duration (default 500 ms)')
    parser.add_argument('--tstep', default=2000., type=float,
                        help='Duration of the current step (default 2000 ms)')
    parser.add_argument('--dt', default=0.025, type=float,
                        help='Integration time step (default 0.025 ms)')
    parser.add_argument('--temperature', default=37., type=float,
                        help='Simulation temperature (default 37 celsius)')
    parser.add_argument('-t','--cell-type', type=str,
                        help='Cell type (either RS or IB)')
    parser.add_argument('-o','--out-file', type=str, help='Output file name')

    args = parser.parse_args()

    amplitudes = []
    for amp in args.amplitudes:
        amplitudes.append(np.linspace(amp[0],amp[1],np.round((amp[1]-amp[0])/amp[2])+1))
    amplitudes = np.array([amp for ampli in amplitudes for amp in ampli])

    # constant parameters
    Rm = {'axon': 100, 'soma': 150, 'dend': 75}
    Ra = {'axon': 100, 'soma': 75, 'dend': 75}
    Cm = {'axon': 1, 'soma': 1, 'dend': 1}

    print(timestamp() + '>> Building the model...')  

    try:
        if args.cell_type.lower() == 'rs':
            cell = swc.RSCell(args.filename,Rm,Ra,Cm,args.simplify)
        elif args.cell_type.lower() == 'ib':
            cell = swc.IBCell(args.filename,Rm,Ra,Cm,args.simplify)
        else:
            print('Unknown cell type: %s. Aborting...' % args.cell_type)
            sys.exit(1)
    except:
        print('You must specify the cell type with the -t,--cell-type option.')
        sys.exit(2)

    start = time.time()
    print(timestamp() + '>> Running the protocol...')
    V,spike_times = run_current_steps_protocol(cell, amplitudes, args.ttran, args.tstep, args.dt, args.temperature)
    stop = time.time()

    print(timestamp() + '>> Saving the results...')
    if args.out_file is None:
        output_file = h5.make_output_filename('steps','.h5')
    else:
        output_file = args.out_file
    try:
        h5.save_h5_file(output_file, dt=args.dt, V=V, spike_times=spike_times,
                        swc_file=args.filename, amplitudes=amplitudes, Rm=Rm, Ra=Ra, Cm=Cm,
                        stimulus={'dur': args.tstep, 'delay': args.ttran}, temperature=args.temperature,
                        simplify=args.simplify, cell_type=args.cell_type, simulation_duration=stop-start)
    except:
        import ipdb
        ipdb.set_trace()

if __name__ == '__main__':
    main()
