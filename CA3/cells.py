
import sys
import itertools as it
import numpy as np
from neuron import h
from utils import *
import btmorph
h.load_file('stdlib.hoc')

DEBUG = False

__all__ = ['Neuron','SimplifiedNeuron','SWCNeuron','SimplifiedNeuron3D']

class Neuron:
    def __init__(self, parameters, with_active=True):
        self.soma = []
        self.basal = []
        self.proximal = []
        self.distal = []
        self.parameters = parameters
        self.build_morphology()
        self.compute_total_area()
        self.insert_passive_mech()
        self.is_active = with_active
        if self.is_active:
            self.insert_active_mech()

    def build_morphology(self):
        self.make_sections()
        self.setup_membrane_capacitance()
        self.setup_axial_resistance()
        self.setup_topology()
        self.compute_nseg()
        self.connect_sections()

    def make_sections(self):
        raise NotImplementedError()

    def setup_membrane_capacitance(self):
        for sec in self.soma:
            sec.cm = self.parameters['soma']['Cm']
        for sec in self.basal:
            sec.cm = self.parameters['scaling'] * self.parameters['soma']['Cm']
        for sec in self.proximal:
            sec.cm = self.parameters['scaling'] * self.parameters['soma']['Cm']
        for sec in self.distal:
            sec.cm = self.parameters['scaling'] * self.parameters['soma']['Cm']

    def setup_axial_resistance(self):
        for sec in self.soma:
            sec.Ra = self.parameters['soma']['Ra']
        for sec in self.basal:
            sec.Ra = self.parameters['basal']['Ra']
        for sec in self.proximal:
            sec.Ra = self.parameters['proximal']['Ra']
        for sec in self.distal:
            sec.Ra = self.parameters['distal']['Ra']

    def setup_topology(self):
        raise NotImplementedError()

    def compute_nseg(self):
        for sec in h.allsec():
            sec.nseg = int((sec.L/(0.1*h.lambda_f(100,sec=sec))+0.9)/2)*2 + 1
            if DEBUG:
                print('%s has %d segments.' % (h.secname(sec=sec),sec.nseg))

    def connect_sections(self):
        raise NotImplementedError()

    def compute_total_area(self):
        self.total_area = 0
        for sec in h.allsec():
            for seg in sec:
                self.total_area += h.area(seg.x, sec)
        if DEBUG:
            print('Total area: %.0f um^2.' % self.total_area)

    def distance_from_soma(self, sec, x=0.5):
        return distance(self.soma[0], sec, x)

    def compute_measures(self):
        # the areas of all sections in the soma
        self.soma_areas = []
        for sec in self.soma:
            self.soma_areas.append(compute_section_area(sec))

        # the areas of all sections in the proximal apical dendrites
        self.proximal_areas = []
        # the distances of all proximal apical sections from the soma: here the
        # distance is just the Euclidean one, not the path length
        self.proximal_distances = []
        for sec in self.apical:
            self.proximal_distances.append(self.distance_from_soma(sec))
            self.proximal_areas.append(compute_section_area(sec))

        # the areas of all sections in the distal apical dendrites
        self.distal_areas = []
        # the distances of all distal apical sections from the soma
        self.distal_distances = []
        for sec in self.distal:
            self.distal_distances.append(self.distance_from_soma(sec))
            self.distal_areas.append(compute_section_area(sec))

        # the areas of all sections in the basal dendrites
        self.basal_areas = []
        # the distances of all basal sections from the soma
        self.basal_distances = []
        for sec in self.basal:
            self.basal_distances.append(self.distance_from_soma(sec))
            self.basal_areas.append(compute_section_area(sec))
        
        # the areas of the sections in the axon
        self.axon_areas = []
        # the path length of each section in the axon from the root
        self.axon_length = []
        try:
            for sec in self.axon:
                self.axon_areas.append(compute_section_area(sec))
                # here we are assuming that the axon extends vertically from the soma,
                # in which case distance and path length are the same. if this is not the
                # case, the subclass should override this method and implement its own code
                self.axon_length.append(self.distance_from_soma(sec))
        except:
            # there is no axon in this cell
            if DEBUG:
                print('No axon present.')

        self.total_area = np.sum(self.soma_areas) + np.sum(self.apical_areas) + \
            np.sum(self.basal_areas) + np.sum(self.axon_areas)

    def insert_passive_mech(self):
        for sec in h.allsec():
            sec.insert('pas')
        for sec in self.soma:
            sec.e_pas = self.parameters['soma']['El']
            sec.g_pas = 1./self.parameters['soma']['Rm']
        for sec in self.basal:
            sec.e_pas = self.parameters['basal']['El']
            sec.g_pas = self.parameters['scaling']/self.parameters['soma']['Rm']
        for sec in self.proximal:
            sec.e_pas = self.parameters['proximal']['El']
            sec.g_pas = self.parameters['scaling']/self.parameters['soma']['Rm']
        for sec in self.distal:
            sec.e_pas = self.parameters['distal']['El']
            sec.g_pas = self.parameters['scaling']/self.parameters['soma']['Rm']

class SimplifiedNeuron (Neuron):
    def __init__(self, parameters, with_active=True):
        Neuron.__init__(self, parameters, with_active)

    def make_sections(self):
        self.soma = [h.Section(name='soma')]
        self.basal = [h.Section(name='basal-%d' % i) for i in range(2)]
        self.proximal = [h.Section(name='proximal')]
        self.distal = [h.Section(name='distal-%d' % i) for i in range(3)]
        self.axon = [h.Section(name='axon-%d' % i) for i in range(5)]

    def setup_membrane_capacitance(self):
        Neuron.setup_membrane_capacitance(self)
        for sec in self.axon:
            sec.cm = self.parameters['axon']['Cm']

    def setup_axial_resistance(self):
        Neuron.setup_axial_resistance(self)
        for sec in self.axon:
            sec.Ra = self.parameters['axon']['Ra']

    def setup_topology(self):
        self.soma[0].L = 20
        self.soma[0].diam = 20
        self.proximal[0].L = 500
        self.proximal[0].diam = 5
        for i in range(3):
            self.distal[i].L = 200
            self.distal[i].diam = 2
        for d in self.basal:
            d.L = 300
            d.diam = 5
        for i in range(len(self.axon)):
            self.axon[i].L = 20
            self.axon[i].diam = 1

    def connect_sections(self):
        self.proximal[0].connect(self.soma[0], 1, 0)
        for i in range(3):
            self.distal[i].connect(self.proximal[0], 1, 0)
        for d in self.basal:
            d.connect(self.soma[0], 0, 0)
        self.axon[0].connect(self.soma[0], 0, 0)
        for i in range(1,len(self.axon)):
            self.axon[i].connect(self.axon[i-1], 1, 0)
        
    def insert_passive_mech(self):
        Neuron.insert_passive_mech(self)
        for sec in self.axon:
            sec.e_pas = self.parameters['axon']['El']
            sec.g_pas = 1./self.parameters['axon']['Rm']

    def insert_active_mech(self):
        self.insert_fast_Na_and_delayed_rectifier_K()
        self.insert_persistent_Na()
        self.insert_Im()
        self.insert_AHP_K()
        self.insert_K_D()
        self.insert_A_type_K()
        self.insert_Ih()
        self.insert_calcium_dynamics()
        self.insert_calcium_currents()

    def insert_calcium_currents(self):
        for sec in it.chain(self.soma,self.proximal,self.distal,self.basal):
            sec.insert('cat')
            sec.insert('cal')
            sec.insert('can')
            # regular spiking
            sec.gcalbar_cal = 1e-5
            sec.gcatbar_cat = 1e-5
            sec.gcanbar_can = 1e-5
            # bursting
            #sec.gcalbar_cal = 1e-5
            #sec.gcatbar_cat = 5e-4
            #sec.gcanbar_can = 1e-4

    def insert_calcium_dynamics(self):
        for sec in it.chain(self.soma,self.proximal,self.distal,self.basal):
            sec.insert('cacum')
            sec.eca = 120
            for seg in sec:
                seg.cacum.depth = seg.diam/2

    def insert_Ih(self):
        for sec in self.soma:
            sec.insert('hd')
            sec.ghdbar_hd = 1e-6
        # sigmoidally increasing Ih in the dendrites.
        # see Poirazi et al., 2003, Neuron
        h.distance(sec=self.soma[0])
        gbar = 9*self.soma[0].ghdbar_hd
        #gbar = 3*self.soma[0].ghdbar_hd
        halfdist = 280. # [um]
        steepness = 50. # [um]
        for sec in it.chain(self.proximal,self.distal):
            sec.insert('hd')
            for seg in sec:
                seg.hd.ghdbar = self.soma[0].ghdbar_hd + (gbar - self.soma[0].ghdbar_hd) / \
                    (1. + np.exp((halfdist-h.distance(seg.x,sec=sec))/steepness))
                if DEBUG:
                    print('gbar Ih @ x = %g: %g' % (seg.x,seg.hd.ghdbar))

    def insert_K_D(self):
        for sec in self.soma:
            sec.insert('kd')
            sec.gkdbar_kd = 1e-8

    def insert_A_type_K(self):
        for sec in h.allsec():
            sec.insert('kap')
            sec.gkabar_kap = 0.001
            if sec in self.axon:
                sec.sh_kap = 0

    def insert_AHP_K(self):
        for sec in self.soma:
            sec.insert('KahpM95')
            sec.gbar_KahpM95 = 0.03

    def insert_Im(self):
        for sec in self.soma:
            sec.insert('km')
            sec.gbar_km = 0.002

    def insert_persistent_Na(self):
        for sec in h.allsec():
            if sec in self.soma or sec in self.axon:
                sec.insert('napinst')
                # regular spiking
                sec.gbar_napinst = 1e-8
            else:
                max_dist = 100.
                h.distance(sec=self.soma[0])
                if h.distance(0, sec=sec) < max_dist:
                    if DEBUG:
                        print('Inserting persistent sodium conductance in %s.' % h.secname(sec=sec))
                    sec.insert('napinst')
                    for seg in sec:
                        seg.napinst.gbar = max(self.soma[0].gbar_napinst * (max_dist-h.distance(seg.x,sec=sec))/max_dist,0)
                        if DEBUG and seg.napinst.gbar > 0:
                            print('g_Na @ x = %g: %g' % (seg.x,seg.napinst.gbar))

    def insert_fast_Na_and_delayed_rectifier_K_bis(self):
        for sec in it.chain(self.soma,self.axon):
            sec.insert('na3')
            sec.insert('kdr')
            sec.ena = 55
            sec.ek = -90
            sec.gbar_na3 = 0.022
            sec.gkdrbar_kdr = 0.01
            sec.sh_na3 = 10
            if sec is self.axon[0]:
                sec.gbar_na3 = 0.05
        max_dist = 100. + self.soma[0].L
        for sec in it.chain(self.proximal,self.distal,self.basal):
            h.distance(sec=self.soma[0])
            if h.distance(0.,sec=sec) < max_dist:
                if DEBUG:
                    print('Inserting HH conductance in %s.' % h.secname(sec=sec))
                sec.insert('na3')
                sec.insert('kdr')
                sec.ena = 55
                sec.ek = -90
                for seg in sec:
                    seg.na3.gbar = max(self.soma[0].gbar_na3 * (max_dist-h.distance(seg.x,sec=sec))/max_dist,0)
                    seg.na3.sh = 10
                    if DEBUG and seg.na3.gbar > 0:
                        print('g_Na @ x = %g: %g' % (seg.x,seg.na3.gbar))

    def insert_fast_Na_and_delayed_rectifier_K(self):
        for sec in it.chain(self.soma,self.axon):
            sec.insert('hh2')
            sec.ena = 55
            sec.ek = -90
            sec.gnabar_hh2 = 0.05
            sec.gkbar_hh2 = 0.005
            if sec is self.axon[0]:
                sec.gnabar_hh2 = 0.25
        # sigmoidally decreasing sodium in the dendrites
        # see Kim et al., 2012, Nat. Neurosci.
        h.distance(sec=self.soma[0])
        gbar = 0.1*self.soma[0].gnabar_hh2
        halfdist = 200. # [um]
        steepness = 30. # [um]
        for sec in it.chain(self.proximal,self.distal,self.basal):
            sec.insert('hh2')
            for seg in sec:
                seg.hh2.gkbar = 0.5 * self.soma[0].gkbar_hh2
                seg.hh2.gnabar = gbar + (self.soma[0].gnabar_hh2 - gbar) / \
                    (1. + np.exp((h.distance(seg.x,sec=sec) - halfdist)/steepness))
                if DEBUG:
                    print('gbar I_Na @ x = %g: %g' % (seg.x,seg.hh2.gnabar))
        # linearly decreasing sodium in the dendrites
        #max_dist = 100. + self.soma[0].L
        #for sec in it.chain(self.proximal,self.distal,self.basal):
        #    h.distance(sec=self.soma[0])
        #    if h.distance(0.,sec=sec) < max_dist:
        #        if DEBUG:
        #            print('Inserting HH conductance in %s.' % h.secname(sec=sec))
        #        sec.insert('hh2')
        #        sec.ena = 55
        #        sec.ek = -90
        #        for seg in sec:
        #            seg.hh2.gnabar = max(self.soma[0].gnabar_hh2 * (max_dist-h.distance(seg.x,sec=sec))/max_dist,0)
        #            if DEBUG and seg.hh2.gnabar > 0:
        #                print('g_Na @ x = %g: %g' % (seg.x,seg.hh2.gnabar))

class SWCNeuron (SimplifiedNeuron):
    def __init__(self, parameters, with_active=True, convert_to_3pt_soma=True):
        if convert_to_3pt_soma:
            self.swc_filename = '.'.join(parameters['swc_filename'].split('.')[:-1]) + '_converted.swc'
            convert_morphology(parameters['swc_filename'], self.swc_filename)
        else:
            self.swc_filename = swc_filename
        SimplifiedNeuron.__init__(self, parameters, with_active)

    def make_sections(self):
        # load the tree structure that represents the morphology
        self.tree = btmorph.STree2()
        self.tree.read_SWC_tree_from_file(self.swc_filename,types=range(10))
        # all the sections, indexed by the corresponding index in the SWC file
        self.sections = {}
        # a list of all the sections that make up the soma
        self.soma = []
        # a list of all the sections that make up the axon
        self.axon = []
        # a list of all the sections that make up the basal dendrites
        self.basal = []
        # a (temporary) list of all the sections that make up the apical dendrites
        self.apical = []
        # a list of all the sections that make up the proximal apical dendrites
        self.proximal = []
        # a list of all the sections that make up the distal apical dendrites
        self.distal = []
        # parse the tree!
        for node in self.tree:
            if node is self.tree.root:
                continue
            # make the section
            self.sections[node.index] = h.Section(name='sec_{0}'.format(node.index))
            section = self.sections[node.index]
            # set the geometry
            if not node.parent is None:
                pPos = node.parent.content['p3d']
                cPos = node.content['p3d']
                c_xyz = cPos.xyz
                p_xyz = pPos.xyz
                h.pt3dclear(sec=section)
                h.pt3dadd(float(p_xyz[0]),float(p_xyz[1]),float(p_xyz[2]),float(pPos.radius),sec=section)
                h.pt3dadd(float(c_xyz[0]),float(c_xyz[1]),float(c_xyz[2]),float(cPos.radius),sec=section)
            # assign it to the proper region
            swc_type = node.content['p3d'].type
            if swc_type == SWC_types['soma']:
                self.soma.append(section)
                h.distance(sec=self.soma[0])
            elif swc_type == SWC_types['axon']:
                self.axon.append(section)
            elif swc_type == SWC_types['basal']:
                self.basal.append(section)
            elif swc_type == SWC_types['apical']:
                self.apical.append(section)
                
    def setup_topology(self):
        pass

    def connect_sections(self):
        for node in self.tree:
            if node is self.tree.root:
                continue
            try:
                self.sections[node.index].connect(self.sections[node.parent.index],1,0)
            except:
                if not self.sections[node.index] is self.soma[0]:
                    self.sections[node.index].connect(self.soma[0],1,0)
        # now that the sections are connected we can subdivide those in the apical
        # dendrite in proximal and distal
        h.distance(sec=self.soma[0])
        for sec in self.apical:
            if h.distance(0.5, sec=sec) < self.parameters['proximal_limit']:
                self.proximal.append(sec)
            else:
                self.distal.append(sec)

    def compute_measures(self):
        Neuron.compute_measures(self)
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
        # the path length of each section in the axon from the root
        self.axon_length = []
        for sec in self.axon:
            index = self.section_index(sec)
            self.axon_length.append(length[index])

class SimplifiedNeuron3D (SimplifiedNeuron):
    def __init__(self, parameters, with_active=True):
        SimplifiedNeuron.__init__(self, parameters, with_active)

    def setup_topology(self):
        h.pt3dadd(0, 0, 0, 20, sec=self.soma[0])
        h.pt3dadd(0, 0, 20, 20, sec=self.soma[0])

        h.pt3dadd(0, 0, 20, 10, sec=self.proximal[0])
        h.pt3dadd(0, 0, 520, 7, sec=self.proximal[0])

        h.pt3dadd(0, 0, 520, 7, sec=self.distal[0])
        h.pt3dadd(0, -140, 650, 3, sec=self.distal[0])

        h.pt3dadd(0, 0, 520, 7, sec=self.distal[1])
        h.pt3dadd(0, 0, 710, 3, sec=self.distal[1])

        h.pt3dadd(0, 0, 520, 7, sec=self.distal[2])
        h.pt3dadd(0, 140, 650, 3, sec=self.distal[2])

        h.pt3dadd(0, 0, 0, 12, sec=self.basal[0])
        h.pt3dadd(0, -212, -212, 7, sec=self.basal[0])

        h.pt3dadd(0, 0, 0, 12, sec=self.basal[1])
        h.pt3dadd(0, 212, -212, 7, sec=self.basal[1])

        h.pt3dadd(0, 0, 0, 5, sec=self.axon[0])
        h.pt3dadd(0, 0, -10, 3, sec=self.axon[0])
        h.pt3dadd(0, 0, -10, 3, sec=self.axon[1])
        h.pt3dadd(0, 0, -20, 1, sec=self.axon[1])
        
        for i in range(2,len(self.axon)):
            h.pt3dadd(0, 0, -20-(i-2)*20, 1, sec=self.axon[i])
            h.pt3dadd(0, 0, -20-(i-1)*20, 1, sec=self.axon[i])

def run_step(amplitude=-0.1):
    parameters = {'scaling': 2, 'soma': {'Cm': 1, 'Ra': 100, 'El': -70, 'Rm': 10e3},
                  'proximal': {'Ra': 500, 'El': -70}, 'distal': {'Ra': 500, 'El': -70},
                  'basal': {'Ra': 500, 'El': -70}, 'axon': {'Cm': 1, 'Ra': 50, 'El': -70, 'Rm': 10e3},
                  'proximal_limit': 100, 'swc_filename': '../morphologies/DH070313-.Edit.scaled.swc'}
    n = SimplifiedNeuron(parameters,with_active=False)
    #n = SWCNeuron(parameters,with_active=False)
    rec = make_voltage_recorders(n)
    stim = h.IClamp(n.soma[0](0.5))
    stim.delay = 200
    stim.amp = amplitude
    stim.dur = 500
    run(tend=1000,V0=-70)
    import pylab as p
    p.plot(rec['t'],rec['vsoma'],'k',label='Soma')
    p.plot(rec['t'],rec['vproximal'],'r',label='Proximal')
    p.plot(rec['t'],rec['vdistal'],'g',label='Distal')
    p.plot(rec['t'],rec['vbasal'],'b',label='Basal')
    p.xlabel('Time (ms)')
    p.ylabel('Membrane voltage (mV)')
    p.legend(loc='best')
    p.show()

if __name__ == '__main__':
    run_step()
