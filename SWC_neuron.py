import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools as it
import btmorph
import neuron
from neuron import h

SWC_types = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4}

def distance(self, origin, end, x=0.5):
    origin.push()
    h.distance()
    end.push()
    dst = h.distance(x)
    h.pop_section()
    h.pop_section()
    return dst

def pick_section(self, group, areas):
    tmp = np.cumsum(areas)
    return group[np.where(tmp > np.random.uniform(0,tmp[-1]))[0][0]]

def filter(self, group, distances, distance_interval):
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

class SWCCell(object):
    def __init__(self,f_name,Rm,Ra,Cm=1):
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
        
        # load the tree structure that represents the morphology
        self.tree = btmorph.STree2()
        self.tree.read_SWC_tree_from_file(f_name,types=range(10))

        self.sections = {}
        self.sections_areas = {}
        
        # the soma is composed of just one section
        self.soma = None
        # a list of all the sections that make up the axon
        self.axon = []
        # a list of all the sections that make up the basal dendrites
        self.basal = []
        # a list of all the sections that make up the apical dendrites
        self.apical = []

        # parameters
        self.Rm = Rm
        self.Cm = Cm
        self.Ra = Ra
        self.load_morphology()
        self.compute_distances()

    def distance_from_soma(self, sec, x=0.5):
        return self.distance(self.soma, sec, x)

    def compute_distances(self):
        # the axon root is the section of type ''axon'' that is closest to the soma
        self.axon_root = None
        self.apical_distances = []
        self.apical_areas = []
        for node in self.tree:
            swc_type = node.content['p3d'].type
            if swc_type == SWC_types['axon']:
                if self.axon_root is None:
                    self.axon_root = node
                    self.axon_length = {self.axon_root.index: self.path_length(
                            self.tree.path_between_nodes(self.axon_root, self.axon_root.parent))}
                else:
                    self.axon_length[node.index] = self.axon_length[node.parent.index] + \
                        np.sqrt(np.sum((node.parent.content['p3d'].xyz - node.content['p3d'].xyz)**2))
        for sec in self.apical:
            self.apical_distances.append(self.distance_from_soma(sec))
            self.apical_areas.append(h.area(0.5,sec))
        self.insert_mechanisms()

    def find_node_by_distance(self, distance, node_type):
        """
        Returns the first node that is at least distance far from the root.
        """
        for node in self.tree:
            if node.content['p3d'].type == node_type:
                if self.find_node_distance_to_root(node) >= distance:
                    return node
        raise Exception('No node is that far from the root.')

    def find_section_by_distance(self, distance, node_type):
        return self.sections[self.find_node_by_distance(distance,node_type).index]

    def find_node_distance_to_root(self, node):
        return self.path_length(self.tree.path_to_root(node))

    def path_length(self, path):
        distance = 0
        for i in range(len(path)-1):
            distance += np.sqrt(np.sum((path[i].content['p3d'].xyz - \
                                        path[i+1].content['p3d'].xyz)**2))
        return distance

    def insert_iclamp(self,amp):
        ic = h.IClamp(self.soma(0.5))
        ic.amp = amp
        ic.dur = 200
        ic.delay = 50
        return ic

    def load_morphology(self):
        for node in self.tree:
            sec = self.make_section(node)
            if not sec is None:
                self.sections[node.index] = sec
                self.sections_areas[node.index] = h.area(0.5,sec)

    def make_section(self,node):
        compartment = h.Section(name='sec_{0}'.format(node.index))
        swc_type = node.content['p3d'].type
        if swc_type == SWC_types['axon']:
            self.axon.append(compartment)
        elif swc_type == SWC_types['basal']:
            self.basal.append(compartment)
        elif swc_type == SWC_types['apical']:
            self.apical.append(compartment)
        if swc_type != SWC_types['soma']:
            # do this for all sections that do not belong to the soma
            pPos = node.parent.content['p3d']
            cPos = node.content['p3d']
            compartment.push()
            c_xyz = cPos.xyz
            p_xyz = pPos.xyz
            if pPos.type == 1:
                # the section is connected to the soma
                pPos = self.tree[1].content['p3d']
                p_xyz = pPos.xyz
            h.pt3dadd(float(p_xyz[0]),float(p_xyz[1]),float(p_xyz[2]),float(pPos.radius))
            h.pt3dadd(float(c_xyz[0]),float(c_xyz[1]),float(c_xyz[2]),float(cPos.radius))
            # nseg according to NEURON book; too high in general...
            compartment.nseg = int(((compartment.L/(0.1*h.lambda_f(100))+0.9)/2)*2+1)
            h.pop_section()
            try:
                compartment.connect(self.sections[node.parent.index],1,0)
            except:
                compartment.connect(self.sections[self.tree.root.index],1,0)
        elif node.parent is None:
            # the root of the SWC is the soma
            # TODO: make the area of the soma right
            compartment.diam = 8
            compartment.L = 15
            self.soma = compartment
        else:
            # do this for all sections in the soma except the first one
            del compartment
            return None
        return compartment
        
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
    def __init__(self, swc_filename, Rm, Ra, Cm=1):
        SWCCell.__init__(self, swc_filename, Rm, Ra, Cm)
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
        for index,distance in self.axon_length.iteritems():
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


#------------------------------Mechanisms for Bursting properties-------------------------

class IBCell(SWCCell):
    def __init__(self, swc_filename, Rm, Ra, Cm=1):
        SWCCell.__init__(self, swc_filename, Rm, Ra, Cm)
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
        for index,distance in self.axon_length.iteritems():
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

