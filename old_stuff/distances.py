#!/usr/bin/env python

from neuron import h
from SWC_neuron import *
import pylab as p
import numpy as np

h.load_file('stdrun.hoc')
#filename = '../morphologies/DH070313-.Edit.scaled_converted.swc'
filename = '../morphologies/DH070613-1-.Edit.scaled_converted.swc'
Rm = {'axon': 100, 'soma': 150, 'dend': 75}
Ra = {'axon': 100, 'soma': 75, 'dend': 75}
Cm = {'axon': 1, 'soma': 1, 'dend': 1}
El = {'axon': -70, 'soma': -70, 'dend': -70}
cell = RSCell(filename,El,Rm,Ra,Cm,0,True)

#try:
#    morph = np.loadtxt(cell.simplified_swc_filename)
#except:
#    morph = np.loadtxt(cell.swc_filename)
#p.plot(morph[:,3],-morph[:,2],'k.')

tree = btmorph.STree2()
tree.read_SWC_tree_from_file(filename)
btmorph.plot_2D_SWC(filename,show_axis=False,color_scheme='default',depth='Y')
    
sections = [sec for group in [cell.apical,cell.basal] for sec in group]
areas = [area for group_areas in [cell.apical_areas,cell.basal_areas] for area in group_areas]
distances = [dst for group_dst in [cell.apical_distances,cell.basal_distances] for dst in group_dst]

dst = 70
dst = 100
#perisomatic,idx = filter(sections, distances, 50.)
perisomatic,idx = filter(cell.apical, cell.apical_distances, dst)
perisomatic_areas = [cell.apical_areas[i] for i in idx]
for i in range(70):
    sec = pick_section(perisomatic, perisomatic_areas)
    node = cell.tree.get_node_with_index(cell.section_index(sec))
    #p.plot(node.content['p3d'].xyz[1],-node.content['p3d'].xyz[0],'ro',lw=2)
    p.plot(node.content['p3d'].xyz[0],node.content['p3d'].xyz[1],'ro',lw=2)

dst = 30
perisomatic,idx = filter(cell.basal, cell.basal_distances, dst)
perisomatic_areas = [cell.basal_areas[i] for i in idx]
for i in range(30):
    sec = pick_section(perisomatic, perisomatic_areas)
    node = cell.tree.get_node_with_index(cell.section_index(sec))
    #p.plot(node.content['p3d'].xyz[1],-node.content['p3d'].xyz[0],'ro',lw=2)
    p.plot(node.content['p3d'].xyz[0],node.content['p3d'].xyz[1],'ro',lw=2)

dst = [150,500]
distal,idx = filter(cell.apical, cell.apical_distances, dst)
distal_areas = [cell.apical_areas[i] for i in idx]
for i in range(100):
    sec = pick_section(distal, distal_areas)
    node = cell.tree.get_node_with_index(cell.section_index(sec))
    #p.plot(node.content['p3d'].xyz[1],-node.content['p3d'].xyz[0],'mo',lw=2)
    p.plot(node.content['p3d'].xyz[0],node.content['p3d'].xyz[1],'mo',lw=2)

dst = [50,500]
distal,idx = filter(cell.basal, cell.basal_distances, dst)
distal_areas = [cell.basal_areas[i] for i in idx]
for i in range(50):
    sec = pick_section(distal, distal_areas)
    node = cell.tree.get_node_with_index(cell.section_index(sec))
    #p.plot(node.content['p3d'].xyz[1],-node.content['p3d'].xyz[0],'go',lw=2)
    p.plot(node.content['p3d'].xyz[0],node.content['p3d'].xyz[1],'go',lw=2)

p.box('off')
p.savefig(filename[:-3]+'.pdf')
p.show()
