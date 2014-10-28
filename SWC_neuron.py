import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools as it
import btmorph
import neuron
from neuron import h

SWC_types = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4}

def distance(origin, end, x=0.5):
    h.distance(sec=origin)
    return h.distance(x, sec=end)

def pick_section(group, areas=None):
    if areas is None:
        tmp = np.cumsum(np.ones(len(group)))
    else:
        tmp = np.cumsum(areas)
    return group[np.where(tmp > np.random.uniform(0,tmp[-1]))[0][0]]

def filter(group, distances, distance_interval):
    try:
        if len(distance_interval) < 2:
            distance_interval = [0,distance_interval[0]]
    except:
        distance_interval = [0,distance_interval]
    sections = []
    indexes = []
    for i,dst in enumerate(distances):
        if dst >= distance_interval[0] and dst <= distance_interval[1]:
            sections.append(group[i])
            indexes.append(i)
    return sections,indexes

def path_length(path):
    distance = 0
    for i in range(len(path)-1):
        distance += np.sqrt(np.sum((path[i].content['p3d'].xyz - path[i+1].content['p3d'].xyz)**2))
    return distance

def simplify_tree(node, min_distance, spare_types=(), removed=None):
    if not node is None:
        if not node.parent is None and len(node.children) == 1 and not node.content['p3d'].type in spare_types:
            dst = np.sqrt(np.sum((node.parent.content['p3d'].xyz - node.children[0].content['p3d'].xyz)**2))
            if dst < min_distance:
                node.parent.children[node.parent.children.index(node)] = node.children[0]
                node.children[0].parent = node.parent
                if not removed is None:
                    removed.append(node.index)
                node = node.parent
        for child in node.children:
            simplify_tree(child, min_distance, spare_types, removed)

class SWCCell(object):
    def __init__(self,f_name,Rm,Ra,Cm=1,min_distance=0.):
        """
        Rm is the input resistance of the cell, in MOhm.
        Cm is the membrane capacitance.
        Ra is the axial resistance of the cell.
        All three variables are dictionaries, with the following keys:
           dend - value to be used for the dendrites
           soma - value to be used for the soma
           axon - value to be used for the axon
        """
        # contains the notorious lambda rule
        h.load_file('stdlib.hoc')
        
        # the path of the SWC file
        self.swc_filename = f_name

        # parameters
        self.Rm = Rm
        self.Cm = Cm
        self.Ra = Ra
        self.min_distance = min_distance

        self.load_morphology()
        self.compute_measures()

    def load_morphology(self):
        # load the tree structure that represents the morphology
        self.tree = btmorph.STree2()
        self.tree.read_SWC_tree_from_file(self.swc_filename,types=range(10))
        if self.min_distance > 0.:
            # simplify the morphology
            total = len(self.tree.get_nodes())
            removed = []
            sys.stdout.write('Simplifying the morphology... ')
            sys.stdout.flush()
            simplify_tree(self.tree.root, self.min_distance, (SWC_types['soma'],), removed)
            sys.stdout.write('removed %d nodes out of %d.\n' % (len(removed),total))
            self.simplified_swc_filename = '.'.join(self.swc_filename.split('.')[:-1]) + \
                '_simplified_%g_um.swc' % self.min_distance
            self.tree.write_SWC_tree_to_file(self.simplified_swc_filename)
        # all the sections, indexed by the corresponding index in the SWC file
        self.sections = {}
        # the soma is composed of just one section
        self.soma = None
        # a list of all the sections that make up the axon
        self.axon = []
        # a list of all the sections that make up the basal dendrites
        self.basal = []
        # a list of all the sections that make up the apical dendrites
        self.apical = []
        # parse the tree!
        for node in self.tree:
            section = h.Section(name='sec_{0}'.format(node.index))
            swc_type = node.content['p3d'].type
            if swc_type == SWC_types['axon']:
                self.axon.append(section)
            elif swc_type == SWC_types['basal']:
                self.basal.append(section)
            elif swc_type == SWC_types['apical']:
                self.apical.append(section)
            if swc_type != SWC_types['soma']:
                # do this for all sections that do not belong to the soma
                pPos = node.parent.content['p3d']
                cPos = node.content['p3d']
                c_xyz = cPos.xyz
                p_xyz = pPos.xyz
                if pPos.type == 1:
                    # the section is connected to the soma
                    pPos = self.tree.root.content['p3d']
                    p_xyz = pPos.xyz
                h.pt3dadd(float(p_xyz[0]),float(p_xyz[1]),float(p_xyz[2]),float(pPos.radius),sec=section)
                h.pt3dadd(float(c_xyz[0]),float(c_xyz[1]),float(c_xyz[2]),float(cPos.radius),sec=section)
                # nseg according to NEURON book; too high in general...
                #section.nseg = int(((section.L/(0.1*h.lambda_f(100))+0.9)/2)*2+1)
                section.nseg = 1
                try:
                    section.connect(self.sections[node.parent.index],1,0)
                except:
                    section.connect(self.sections[self.tree.root.index],1,0)
            elif node.parent is None:
                # the root of the SWC is the soma
                # TODO: make the area of the soma right
                self.soma = section
                self.soma.diam = 8
                self.soma.L = 15
            else:
                # do this for all sections in the soma except the first one
                del section
                continue
            self.sections[node.index] = section
        
    def compute_measures(self):
        # the areas of all sections in the apical dendrites
        self.apical_areas = []
        # the distances of all apical sections from the soma: here the
        # distance is just the Euclidean one, not the path length
        self.apical_distances = []
        for sec in self.apical:
            self.apical_distances.append(self.distance_from_soma(sec))
            self.apical_areas.append(h.area(0.5,sec))

        # the areas of all sections in the basal dendrites
        self.basal_areas = []
        # the distances of all basal sections from the soma
        self.basal_distances = []
        for sec in self.basal:
            self.basal_distances.append(self.distance_from_soma(sec))
            self.basal_areas.append(h.area(0.5,sec))

        # the axon root is the section of type ''axon'' that is closest to the soma
        axon_root = None
        for node in self.tree:
            swc_type = node.content['p3d'].type
            if swc_type == SWC_types['axon']:
                if axon_root is None:
                    axon_root = node
                    length = {axon_root.index: path_length(self.tree.path_between_nodes(axon_root, axon_root.parent))}
                else:
                    length[node.index] = length[node.parent.index] + \
                        np.sqrt(np.sum((node.parent.content['p3d'].xyz - node.content['p3d'].xyz)**2))
        self.axon_length = []
        for sec in self.axon:
            index = self.section_index(sec)
            self.axon_length.append(length[index])

    def distance_from_soma(self, sec, x=0.5):
        return distance(self.soma, sec, x)

    def path_length_to_root(self, node):
        return path_length(self.tree.path_to_root(node))

    def section_index(self, section):
        return int(h.secname(sec=section).split('_')[-1])

    def get_soma(self):
        return self.__soma

    def set_soma(self, value):
        self.__soma = value

    soma = property(get_soma, set_soma)

    def get_axon(self):
        return self.__axon

    def set_axon(self, value):
        self.__axon = value

    axon = property(get_axon, set_axon)

    def get_basal(self):
        return self.__basal

    def set_basal(self, value):
        self.__basal = value

    basal = property(get_basal, set_basal)

    def get_apical(self):
        return self.__apical

    def set_apical(self, value):
        self.__apical = value

    apical = property(get_apical, set_apical)

#------------------------------Mechanisms for Regular Spiking properties------------------------
class RSCell(SWCCell):
    def __init__(self, swc_filename, Rm, Ra, Cm=1., min_distance=0.):
        SWCCell.__init__(self, swc_filename, Rm, Ra, Cm, min_distance)
        self.insert_mechanisms()

    def insert_mechanisms(self):
	##### SOMA
        soma = self.soma
        soma.cm = self.Cm['soma']
        soma.Ra = self.Ra['soma']
        soma.insert('pas') # add passive properties
	soma(0.5).pas.g = 0.00002        #1./(self.Rm['soma']*1000)#0.00001
	soma(0.5).pas.e = -60
	soma.insert('hh2') # add Na+ and K+ HH conductances
	hh2_mech = soma(0.5).hh2
	soma.ek = -90
	soma.ena = 50
	hh2_mech.vtraub = -55
	hh2_mech.gnabar = 0.05
	hh2_mech.gkbar = 0.005
	soma.insert('im') # add M conductances
	h.taumax_im = 1000
	im_mech = soma(0.5).im
	im_mech.gkbar = 1e-3
	soma.insert('cad') # add Ca2+ decay dynamics
	soma(0.5).cad.depth = 1
	soma(0.5).cad.taur= 5
	soma(0.5).cad.cainf = 2.4e-4
	soma(0.5).cad.kt = 0
	soma.insert('it') # add T-type Ca2+ conductance
	soma.cai = 2.4e-4
	soma.cao = 2
	soma.eca = 120
	soma.gcabar_it = 0.002
	soma.insert('ical') # add L-type Ca2+ conductance	
	soma.cai = 2.4e-4
	soma.cao = 2 
	soma.eca = 120
	soma.gcabar_ical = 1e-4
	soma.insert('KahpM95') # add calcium activated potassium conductance (I_ahp)
	soma.cai = 50e-6
	soma.gbar_KahpM95 = 0.03
	soma.insert('kd') # add delay rectifier K+ conductance
	soma.ek = -90
	soma.gkdbar_kd = 1e-5
	soma.insert('napinst') # add persistent Na+ conductance
	soma.ena = 50
	soma.gbar_napinst = 1e-8 	
	##### DENDRITES
	for section in it.chain(self.basal,self.apical):
            section.insert('pas') # add passive properties
	    section.cm = self.Cm['dend']
            section.Ra = self.Ra['dend']
            section.insert('hh2') # add hodgkin huxley Na+ and K+ conductances
            section.insert('cad') # add Ca2+ decay dynamics
            section.insert('it') # add T-type Ca2+ conductance
            section.insert('ical') # add L-type Ca2+ conductance
            section.insert('KahpM95') # add calcium activated potassium conductance
            section.insert('kd') # add delayed rectifier K+ conductance
	    section.ek = -90
            section.ena = 50
            for segment in section:
                segment.pas.g = 0.0005 # 1./(self.Rm['dend']*1000)
		segment.pas.e = -70
                segment.hh2.vtraub = -55
                segment.hh2.gnabar = 0.05
                segment.hh2.gkbar = 0.005
                segment.cad.depth = 1
                segment.cad.taur= 5
                segment.cad.cainf = 2.4e-4
                segment.cad.kt = 0
                segment.cai = 2.4e-4
                segment.cao = 2
                segment.eca = 120
                segment.gcabar_it = 0.002	
                segment.cai = 2.4e-4
                segment.cao = 2 
                segment.eca = 120
                segment.gcabar_ical = 1e-4
                segment.cai = 50e-6
                segment.gbar_KahpM95 = 0.02
		soma.gkdbar_kd = 1e-4

        ##### AXON
        for section,distance in zip(self.axon,self.axon_length):
            section.cm = self.Cm['axon']
            section.Ra = self.Ra['axon']
            section.insert('pas')
            section.insert('hh2')
            for segment in section:
                segment.pas.g = 1./(self.Rm['axon']*1000)
                segment.pas.e = -70
                segment.hh2.vtraub = -55
                segment.hh2.gkbar = 0.006
                segment.hh2.gnabar = 0.05
                if distance < 10:
                    # AIS
                    segment.hh2.gnabar = 0.25


#------------------------------Mechanisms for Bursting properties-------------------------

class IBCell(SWCCell):
    def __init__(self, swc_filename, Rm, Ra, Cm=1., min_distance=0.):
        SWCCell.__init__(self, swc_filename, Rm, Ra, Cm, min_distance)
        self.insert_mechanisms()
    
    def insert_mechanisms(self):
        ##### SOMA
        soma = self.soma
        soma.cm = self.Cm['soma']
        soma.Ra = self.Ra['soma']
        soma.insert('pas') # add passive properties
        soma(0.5).pas.g = 0.00005        #1./(self.Rm['soma']*1000)#0.00001
        soma(0.5).pas.e = -70
        soma.insert('hh2') # add Na+ and K+ HH conductances
        hh2_mech = soma(0.5).hh2
        soma.ek = -80
        soma.ena = 50
        hh2_mech.vtraub = -55
        hh2_mech.gnabar = 0.05
        hh2_mech.gkbar = 0.005
        soma.insert('im') # add M conductances
        h.taumax_im = 1000
        im_mech = soma(0.5).im
        im_mech.gkbar = 1e-5
        soma.insert('cad') # add Ca2+ decay dynamics
        soma(0.5).cad.depth = 1
        soma(0.5).cad.taur= 5
        soma(0.5).cad.cainf = 2.4e-4
        soma(0.5).cad.kt = 0
        soma.insert('it') # add T-type Ca2+ conductance
        soma.cai = 2.4e-4
        soma.cao = 2
        soma.eca = 120
        soma.gcabar_it = 5e-4
        soma.insert('ical') # add L-type Ca2+ conductance
        soma.cai = 2.4e-4
        soma.cao = 2
        soma.eca = 120
        soma.gcabar_ical = 5e-4
        soma.insert('KahpM95') # add calcium activated potassium conductance (I_ahp)
        soma.cai = 50e-6
        soma.gbar_KahpM95 = 0.0007
        soma.insert('kd') # add delay rectifier K+ conductance
        soma.ek = -80
        soma.gkdbar_kd = 1e-5
        soma.insert('napinst') # add persistent Na+ conductance
        soma.ena = 50
        soma.gbar_napinst = 5e-5
        
        ##### DENDRITES
        for section in it.chain(self.basal,self.apical):
            section.insert('pas') # add passive properties
            section.cm = self.Cm['dend']
            section.Ra = self.Ra['dend']
            section.insert('hh2') # add hodgkin huxley Na+ and K+ conductances
            section.insert('cad') # add Ca2+ decay dynamics
            section.insert('it') # add T-type Ca2+ conductance
            section.insert('ical') # add L-type Ca2+ conductance
            section.insert('KahpM95') # add calcium activated potassium conductance
            section.insert('kd') # add delayed rectifier K+ conductance
            section.ek = -80
            section.ena = 50
            for segment in section:
                segment.pas.g = 0.0001 # 1./(self.Rm['dend']*1000)
                segment.pas.e = -70
                segment.hh2.vtraub = -55
                segment.hh2.gnabar = 0.05
                segment.hh2.gkbar = 0.005
                segment.cad.depth = 1
                segment.cad.taur= 5
                segment.cad.cainf = 2.4e-4
                segment.cad.kt = 0
                segment.cai = 2.4e-4
                segment.cao = 2
                segment.eca = 120
                segment.gcabar_it = 1e-3
                segment.cai = 2.4e-4
                segment.cao = 2
                segment.eca = 120
                segment.gcabar_ical = 9e-4
                segment.cai = 50e-6
                segment.gbar_KahpM95 = 0.001
                soma.gkdbar_kd = 1e-5
        
        ##### AXON
        for section,distance in zip(self.axon,self.axon_length):
            section = self.sections[index]
            section.cm = self.Cm['axon']
            section.Ra = self.Ra['axon']
            section.insert('pas')
            section.insert('hh2')
            for segment in section:
                segment.pas.g = 1./(self.Rm['axon']*1000)
                segment.pas.e = -70
                segment.hh2.vtraub = -55
                segment.hh2.gkbar = 0.006
                segment.hh2.gnabar = 0.05
                if distance < 10:
                    # AIS
                    segment.hh2.gnabar = 0.25
