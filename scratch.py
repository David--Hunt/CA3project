#from neuronvisio.controls import Controls
#controls = Controls() # start the neuronvisio GUI
#controls.make_animation_screenshots(time_start, time_stop=None, saving_dir='anim')
#ffmpeg -f image2 -r 10 -i %09d.png -sameq anim.mov -pass 2

from neuron import h
from SWC_neuron import *
import pylab as p
import numpy as np

h.load_file('stdrun.hoc')
filename = '../morphologies/DH070313-.Edit.scaled.swc'# designate morphology to be used (SWC file name)
Rm = {'axon': 100, 'soma': 150, 'dend': 75}
Ra = {'axon': 100, 'soma': 75, 'dend': 75}
Cm = {'axon': 1, 'soma': 1, 'dend': 1}
n = RSCell(filename,Rm,Ra,Cm) # designate intrinsic properties (bursting or regular spiking)
ic = n.insert_iclamp(0.5) # amplitude of current injection
rec = {}
for lbl in 't','vsoma','vbasal','vapical','vaxon','vais','gnaf','gnap':
    rec[lbl] = h.Vector()
rec['t'].record(h._ref_t)
rec['vsoma'].record(n.soma(0.5)._ref_v)
dist = 8
sec = n.find_section_by_distance(dist,2)
rec['vais'].record(sec(0.5)._ref_v)
dist = 100
sec = n.find_section_by_distance(dist,2)
rec['vaxon'].record(sec(0.5)._ref_v)
dist = 200
sec = n.find_section_by_distance(dist,3)
rec['vbasal'].record(sec(0.5)._ref_v)
dist = 300
sec = n.find_section_by_distance(dist,4)
rec['vapical'].record(sec(0.5)._ref_v)
rec['gnaf'].record(n.soma(0.5).hh2._ref_gna)
rec['gnap'].record(n.soma(0.5).napinst._ref_gna)

h.celsius = 35
h.tstop = 30
print('Running the simulation...')
h.run()

#---------------------plotting the outputs-----------------------------

p.figure()
#p.subplot(2,1,1)
p.plot(rec['t'],rec['vsoma'],'k',label='Soma')
#p.plot(rec['t'],rec['vais'],'r',label='AIS')
#p.plot(rec['t'],rec['vaxon'],'g',label='Axon')
#p.plot(rec['t'],rec['vbasal'],'b',label='Basal')
#p.plot(rec['t'],rec['vapical'],'m',label='Apical')
#p.legend(loc='best')
#p.subplot(2,1,2)
#INaf = np.array(rec['gnaf'])*(np.array(rec['vsoma']) - n.soma.ena)
#INap = np.array(rec['gnap'])*(np.array(rec['vsoma']) - n.soma.ena)
#p.plot(rec['t'],INaf,'k',label='INa,f')
#p.plot(rec['t'],INap,'r',label='INa,p')
#p.legend(loc='best')
p.show()


