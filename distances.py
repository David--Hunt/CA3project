#!/usr/bin/env python

from neuron import h
from SWC_neuron import *
import pylab as p
import numpy as np

h.load_file('stdrun.hoc')
filename = '../morphologies/DH070313-.Edit.scaled.swc'
#filename = '../morphologies/DH070613-1-.Edit.scaled.swc'
Rm = {'axon': 100, 'soma': 150, 'dend': 75}
Ra = {'axon': 100, 'soma': 75, 'dend': 75}
Cm = {'axon': 1, 'soma': 1, 'dend': 1}
cell = RSCell(filename,Rm,Ra,Cm,10.,True)

try:
    morph = np.loadtxt(cell.simplified_swc_filename)
except:
    morph = np.loadtxt(cell.swc_filename)
p.plot(morph[:,3],-morph[:,2],'k.')

sections = [sec for group in [cell.apical,cell.basal] for sec in group]
areas = [area for group_areas in [cell.apical_areas,cell.basal_areas] for area in group_areas]
distances = [dst for group_dst in [cell.apical_distances,cell.basal_distances] for dst in group_dst]

perisomatic,idx = filter(sections, distances, 50.)
perisomatic_areas = [areas[i] for i in idx]
for i in range(100):
    sec = pick_section(perisomatic, perisomatic_areas)
    node = cell.tree.get_node_with_index(cell.section_index(sec))
    p.plot(node.content['p3d'].xyz[1],-node.content['p3d'].xyz[0],'ro',lw=2)

distal,idx = filter(cell.apical, cell.apical_distances, [300.,500.])
distal_areas = [cell.apical_areas[i] for i in idx]
for i in range(200):
    sec = pick_section(distal, distal_areas)
    node = cell.tree.get_node_with_index(cell.section_index(sec))
    p.plot(node.content['p3d'].xyz[1],-node.content['p3d'].xyz[0],'mo',lw=2)

distal,idx = filter(cell.basal, cell.basal_distances, [100.,500.])
distal_areas = [cell.basal_areas[i] for i in idx]
for i in range(100):
    sec = pick_section(distal, distal_areas)
    node = cell.tree.get_node_with_index(cell.section_index(sec))
    p.plot(node.content['p3d'].xyz[1],-node.content['p3d'].xyz[0],'go',lw=2)

p.show()
