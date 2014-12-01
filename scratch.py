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
#filename = '../morphologies/DH070613-1-.Edit.scaled.swc'
#Rm = {'axon': 100, 'soma': 150, 'dend': 75}
#Ra = {'axon': 100, 'soma': 75, 'dend': 75}
Rm = {'axon': 6e3, 'soma': 6e3, 'dend': 6e3}
Ra = {'axon': 50, 'soma': 150, 'dend': 150}
Cm = {'axon': 1, 'soma': 1, 'dend': 1}

n = RSCell(filename,Rm,Ra,Cm,10,True) # designate intrinsic properties (bursting or regular spiking)
#n = IBCell(filename,Rm,Ra,Cm,10,True) # designate intrinsic properties (bursting or regular spiking)

# insert current stimulus in the soma
ic = h.IClamp(n.soma[0](0.5))
ic.amp = -0.1
ic.dur = 1500
ic.delay = 250

# make the recorders
rec = {}
for lbl in 't','vsoma','vbasal','vapical','vaxon','vais','gnaf','gnap':
    rec[lbl] = h.Vector()
# record time
rec['t'].record(h._ref_t)
# record the voltage at the soma
rec['vsoma'].record(n.soma[0](0.5)._ref_v)

# record the voltage at one of the basal dendrites
dist = 200.
sections,idx = filter(n.basal,n.basal_distances,dist)
sec = pick_section(sections)
rec['vbasal'].record(sec(0.5)._ref_v)

# record the voltage at one of the apical dendrites
dist = 300.
sections,idx = filter(n.apical,n.apical_distances,dist)
sec = pick_section(sections)
rec['vapical'].record(sec(0.5)._ref_v)

# record the voltage at the axon initial segment
dist = 20.
sections,idx = filter(n.axon,n.axon_length,dist)
sec = pick_section(sections)
rec['vais'].record(sec(0.5)._ref_v)

# record the voltage in the axon
dist = 100
sections,idx = filter(n.axon,n.axon_length,dist)
sec = pick_section(sections)
rec['vaxon'].record(sec(0.5)._ref_v)

# record fast and persistent sodium conductances
#rec['gnaf'].record(n.soma[0](0.5).hh2._ref_gna)
#rec['gnap'].record(n.soma[0](0.5).napinst._ref_gna)

h.celsius = 35
h.tstop = 2000
h.cvode_active(1)
h.cvode.maxstep(10)
print('Running the simulation...')
h.run()

#---------------------plotting the outputs-----------------------------

p.figure()
#p.subplot(2,1,1)
p.plot(rec['t'],rec['vsoma'],'k',label='Soma')
p.plot(rec['t'],rec['vais'],'r',label='AIS')
p.plot(rec['t'],rec['vaxon'],'g',label='Axon')
p.plot(rec['t'],rec['vbasal'],'b',label='Basal')
p.plot(rec['t'],rec['vapical'],'m',label='Apical')
p.xlabel('Time (ms)')
p.ylabel('Voltage (mV)')
#p.ylim([-100,50])
p.legend(loc='best')
#p.subplot(2,1,2)
#INaf = np.array(rec['gnaf'])*(np.array(rec['vsoma']) - n.soma[0].ena)
#INap = np.array(rec['gnap'])*(np.array(rec['vsoma']) - n.soma[0].ena)
#p.plot(rec['t'],INaf,'k',label='INa,f')
#p.plot(rec['t'],INap,'r',label='INa,p')
#p.xlabel('Time (ms)')
#p.ylabel('Current (nA)')
#p.legend(loc='best')
p.show()


