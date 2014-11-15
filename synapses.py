
import numpy as np
import pylab as p
from neuron import h

class Synapse (object):
    def __init__(self, weight, delay=1.):
        self.stim = h.VecStim()
        self.nc = h.NetCon(self.stim, self.syn)
        self.nc.weight[0] = weight
        self.nc.delay = delay

    def set_presynaptic_spike_times(self, spike_times):
        self.spike_times = h.Vector(spike_times)
        self.stim.play(self.spike_times)

class BiExponentialSynapse (Synapse):
    def __init__(self, sec, x, E, tau1, tau2, weight, delay=1.):
        self.syn = h.Exp2Syn(sec(x))
        self.syn.tau1 = tau1
        self.syn.tau2 = tau2
        self.syn.e = E
        Synapse.__init__(self, weight, delay)

class AMPASynapse (Synapse):
    def __init__(self, sec, x, E, weight, delay=1.):
        self.syn = h.AMPA_S(sec(x))
        self.Erev = E
        Synapse.__init__(self, weight, delay)

def main():
    import pylab as p
    soma = h.Section()
    soma.insert('pas')
    soma.L = 100
    soma.diam = 100
    weight_min = 0.005
    weight_max = 0.05
    mu = (np.log(weight_min)+np.log(weight_max))/2
    sigma = (np.log(weight_max)-mu)/3
    weights = np.sort(np.exp(np.random.normal(mu,sigma,size=200)))
    synapses = [AMPASynapse(soma, 0.5, 0, w) for w in weights]
    for i,syn in enumerate(synapses):
        syn.set_presynaptic_spike_times([10+i*50])
    rec = {}
    for lbl in 't','v','g':
        rec[lbl] = h.Vector()
    rec['t'].record(h._ref_t)
    rec['v'].record(soma(0.5)._ref_v)
    rec['g'].record(syn.syn._ref_g)
    h.load_file('stdrun.hoc')
    h.v_init = -70
    h.celsius = 37
    h.tstop = len(weights)*50 + 100
    h.run()
    p.subplot(2,1,1)
    p.plot(rec['t'],rec['v'],'k')
    p.ylabel('Voltage (mV)')
    p.subplot(2,1,2)
    p.plot(rec['t'],rec['g'],'r')
    p.xlabel('Time (ms)')
    p.ylabel('Conductance (uS)')
    p.show()

if __name__ == '__main__':
    main()

