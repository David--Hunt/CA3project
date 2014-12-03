import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools as it
import btmorph
import neuron
from neuron import h

soma_type = 1
axon_type = 2
basal_type = 3
apical_type = 4

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
        h.load_file("stdlib.hoc") # contains the notorious lambda rule
        self.tree = btmorph.STree2()
        self.tree.read_SWC_tree_from_file(f_name,types=range(10))
        self.nodes = self.tree.get_nodes()
        self.sections = {}
        self._soma = []
        self._axon = []
        self._basal = []
        self._apical = []
        self.Rm = Rm
        self.Cm = Cm
        self.Ra = Ra
        self.load_morphology()
        self.axon_root = None
        for node in self.nodes:
            if node.get_content()['p3d'].type == axon_type:
                if self.axon_root is None:
                    self.axon_root = node
                    self.axon_distances = {self.axon_root.get_index(): self.path_length(self.tree.path_between_nodes(node, node.get_parent_node()))}
                else:
                    self.axon_distances[node.get_index()] = self.axon_distances[node.get_parent_node().get_index()] + \
                        np.sqrt(np.sum((node.get_parent_node().get_content()['p3d'].xyz - node.get_content()['p3d'].xyz)**2))
        self.insert_mechanisms()

    def find_node_by_distance(self, distance, node_type):
        """
        Returns the first node that is at least distance far from the root.
        """
        for node in self.nodes:
            if node.get_content()['p3d'].type == node_type:
                if self.find_node_distance_to_root(node) >= distance:
                    return node
        raise Exception('No node is that far from the root.')

    def find_section_by_distance(self, distance, node_type):
        return self.sections[self.find_node_by_distance(distance,node_type).get_index()]

    def find_node_distance_to_root(self, node):
        return self.path_length(self.tree.path_to_root(node))

    def path_length(self, path):
        distance = 0
        for i in range(len(path)-1):
            distance += np.sqrt(np.sum((path[i].get_content()['p3d'].xyz - \
                                        path[i+1].get_content()['p3d'].xyz)**2))
        return distance

    def insert_iclamp(self,amp):
        ic = h.IClamp(self.soma[0](0.5))
        ic.amp = amp
        ic.dur = 200
        ic.delay = 50
        return ic

    def load_morphology(self):
        for node in self.nodes:
            sec = self.make_section(node)
            if not sec is None:
                self.sections[node.get_index()] = sec
        self._soma = [self._soma[0]]

    def make_section(self,node):
        compartment = h.Section(name='sec_{0}'.format(node.get_index()))
        swc_type = node.get_content()['p3d'].type
        if swc_type == 1:
            self._soma.append(compartment)
        elif swc_type == 2:
            self._axon.append(compartment)
        elif swc_type == 3:
            self._basal.append(compartment)
        elif swc_type == 4:
            self._apical.append(compartment)
        if swc_type != 1:
            # do this for all sections that do not belong to the soma
            parent = node.get_parent_node()
            pPos = parent.get_content()['p3d']
            cPos = node.get_content()['p3d']
            compartment.push()
            c_xyz = cPos.xyz
            p_xyz = pPos.xyz
            if pPos.type == 1:
                # the section is connected to the soma
                parent = self.nodes[0]
                pPos = parent.get_content()['p3d']
                p_xyz = pPos.xyz
            h.pt3dadd(float(p_xyz[0]),float(p_xyz[1]),float(p_xyz[2]),float(pPos.radius))
            h.pt3dadd(float(c_xyz[0]),float(c_xyz[1]),float(c_xyz[2]),float(cPos.radius))
            # nseg according to NEURON book; too high in general...
            compartment.nseg = int( ((compartment.L/(0.1*h.lambda_f(100))+0.9)/2)*2+1)
            h.pop_section()
            compartment.connect(self.sections.get(parent.get_index()),1,0)
        elif node.get_parent_node() is None:
            # do this only for the first section of the soma
            # root of SWC tree = soma
            cPos = node.get_content()['p3d']
            compartment.push()
            compartment.diam = 8
            compartment.L = 15
            h.pop_section()
        else:
            # do this for all sections in the soma except the first one
            del compartment
            return None
        return compartment
        
    @property
    def soma(self):
        return self._soma
            
    @property
    def axon(self):
        return self._axon
            
    @property
    def basal(self):
        return self._basal
            
    @property
    def apical(self):
        return self._apical

#------------------------------Mechanisms for Regular Spiking properties------------------------
class RSCell(SWCCell):
    def __init__(self, swc_filename, Rm, Ra, Cm=1):
        SWCCell.__init__(self, swc_filename, Rm, Ra, Cm)
        #self.insert_mechanisms()

    def insert_mechanisms(self):
	##### SOMA
        soma = self.soma[0]
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
        for index,distance in self.axon_distances.iteritems():
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
        soma = self.soma[0]
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
        soma.gbar_KahpM95 = 0.001
        soma.insert('kd') # add delay rectifier K+ conductance
        soma.ek = -80
        soma.gkdbar_kd = 1e-5
        soma.insert('napinst') # add persistent Na+ conductance
        soma.ena = 50
        soma.gbar_napinst = 4e-5
        
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
                segment.gcabar_ical = 1e-3
                segment.cai = 50e-6
                segment.gbar_KahpM95 = 0.002
                soma.gkdbar_kd = 1e-5
        
        ##### AXON
        for index,distance in self.axon_distances.iteritems():
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




