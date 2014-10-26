#!/usr/bin/env python

from neuron import h
from SWC_neuron import *
import pylab as p
import numpy as np

h.load_file('stdrun.hoc')
filename = '../morphologies/DH070313-.Edit.scaled.swc'# designate morphology to be used (SWC file name)
Rm = {'axon': 100, 'soma': 150, 'dend': 75}
Ra = {'axon': 100, 'soma': 75, 'dend': 75}
Cm = {'axon': 1, 'soma': 1, 'dend': 1}
cell = RSCell(filename,Rm,Ra,Cm)

perisomatic,idx = cell.filter_sections(cell.apical, cell.apical_distances, 100)
perisomatic_areas = np.array(cell.apical_areas)[idx].tolist()
for i in range(20):
    sec = cell.pick_section(perisomatic, perisomatic_areas)
    sec.push()
    index = int(h.secname().split('_')[-1])
    h.pop_section()
    node = cell.tree.get_node_with_index(index)
    print node.content['p3d'].xyz
