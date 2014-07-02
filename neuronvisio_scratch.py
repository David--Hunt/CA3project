
#from neuronvisio.controls import Controls
from neuron import h
import pylab as p

class BallStick:
    def __init__(self):
        self.soma = h.Section('soma')
        self.soma.L = 20
	self.soma.diam = 20
	self.dend = h.Section('dend')
	self.dend.L = 200
	self.dend.diam = 5
	self.dend.nseg = 9
	self.soma.insert('hh')
	self.dend.insert('hh')
	self.soma.connect(self.dend, 0, 1)

n = BallStick()
ic = h.IClamp(n.soma(0.5))
ic.amp = 0.1
ic.dur = 200
ic.delay = 200

#rec = {}
#for lbl in 't','v':
#    rec[lbl] = h.Vector()
#rec['t'].record(h._ref_t)
#rec['v'].record(n.soma(0.5)._ref_v)
#
#h.load_file('stdrun.hoc')
#h.tstop = 600
#h.run()
#
#p.plot(rec['t'],rec['v'],'k')
#p.show()

#controls = Controls()
