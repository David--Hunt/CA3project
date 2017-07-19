
import os
import sys
import itertools as it
import numpy as np
from neuron import h
from utils import *
import btmorph
h.load_file('stdlib.hoc')

DEBUG = False

__all__ = ['Neuron','SimplifiedNeuron','ThornyNeuron','AThornyNeuron','SWCNeuron','SimplifiedNeuron3D',
           'taper_section','neuron_factory']

# convert from pS/um2 (human-friendly) to S/cm2 (NEURON unit)
PSUM2_TO_SCM2 = 1e-4

def taper_section(sec, max_diam, min_diam):
    if max_diam < 0 or min_diam < 0:
        raise Exception('Diameters must be positive')
    for seg in sec:
        seg.diam = max_diam - seg.x * (max_diam - min_diam)

class Neuron:
    def __init__(self, parameters, with_axon=True):
        self.parameters = parameters
        self.has_axon = with_axon
        self.soma = []
        self.basal = []
        self.proximal = []
        self.distal = []
        if self.has_axon:
            self.axon = []
        self.build_morphology()
        self.compute_measures()
        self.compute_total_area()
        self.insert_passive_mech()
        self.insert_active_mech()

    def save_properties(self, filename=None):
        if filename is None:
            filename = h5.make_output_filename(self.__class__.__name__+'_properties','.h5')
        h5.save_h5_file(filename, parameters=self.parameters,
                        has_axon=self.has_axon, neuron_type=self.__class__.__name__)

    def build_morphology(self):
        self.make_sections()
        self.setup_membrane_capacitance()
        self.setup_axial_resistance()
        self.adjust_dimensions()
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
        if self.has_axon:
            for sec in self.axon:
                sec.cm = self.parameters['axon']['Cm']

    def setup_axial_resistance(self):
        for sec in self.soma:
            sec.Ra = self.parameters['soma']['Ra']
        for sec in self.basal:
            sec.Ra = self.parameters['basal']['Ra']
        for sec in self.proximal:
            sec.Ra = self.parameters['proximal']['Ra']
        for sec in self.distal:
            sec.Ra = self.parameters['distal']['Ra']
        if self.has_axon:
            for sec in self.axon:
                sec.Ra = self.parameters['axon']['Ra']

    def adjust_dimensions(self):
        pass

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
        for sec in self.proximal:
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

        self.total_area = np.sum(self.soma_areas) + np.sum(self.proximal_areas) + \
            np.sum(self.distal_areas) + np.sum(self.basal_areas)

        if self.has_axon:
            # the areas of the sections in the axon
            self.axon_areas = []
            # the path length of each section in the axon from the root
            self.axon_lengths = []
            for sec in self.axon:
                self.axon_areas.append(compute_section_area(sec))
                # here we are assuming that the axon extends vertically from the soma,
                # in which case distance and path length are the same. if this is not the
                # case, the subclass should override this method and implement its own code
                self.axon_lengths.append(self.distance_from_soma(sec))
            self.total_area += np.sum(self.axon_areas)

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
        if self.has_axon:
            for sec in self.axon:
                sec.e_pas = self.parameters['axon']['El']
                sec.g_pas = 1./self.parameters['soma']['Rm']

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

    def get_proximal(self):
        return self.__proximal

    def set_proximal(self, value):
        self.__proximal = value

    proximal = property(get_proximal, set_proximal)

    def get_distal(self):
        return self.__distal

    def set_distal(self, value):
        self.__distal = value

    distal = property(get_distal, set_distal)

class SimplifiedNeuron (Neuron):
    def __init__(self, parameters, with_axon=True):
        Neuron.__init__(self, parameters, with_axon)

    @classmethod
    def n_somatic_sections(cls):
        return 1

    @classmethod
    def n_basal_sections(cls):
        return 1

    @classmethod
    def n_proximal_sections(cls):
        return 1

    @classmethod
    def n_distal_sections(cls):
        return 1

    @classmethod
    def n_axonal_sections(cls):
        # axon hillock, axon initial segment and axon proper
        return 3

    def build_morphology(self):
        Neuron.build_morphology(self)
        if self.has_axon:
            # see Bahl et al., J Neurosci Methods, 2012
            self.axon[0].nseg = 5
            self.axon[1].nseg = 5
            taper_section(self.axon[0], 3.5, 2.)
            taper_section(self.axon[1], 2., 1.5)

    def make_sections(self):
        self.soma = [h.Section(name='soma') for i in range(self.n_somatic_sections())]
        self.basal = [h.Section(name='basal') for i in range(self.n_basal_sections())]
        self.proximal = [h.Section(name='proximal') for i in range(self.n_proximal_sections())]
        self.distal = [h.Section(name='distal') for i in range(self.n_distal_sections())]
        if self.has_axon:
            self.axon = [h.Section(name='hillock'), h.Section(name='AIS'), h.Section(name='axon')]

    def adjust_dimensions(self):
        if 'area' in self.parameters['soma']:
            self.parameters['soma']['diam'] = np.sqrt(self.parameters['soma']['area']/np.pi)
            self.parameters['soma']['L'] = self.parameters['soma']['diam'] / self.n_somatic_sections()
        if not 'diam' in self.parameters['basal']:
            self.parameters['basal']['diam'] = self.parameters['basal']['area'] / \
            (self.n_basal_sections()*np.pi*self.parameters['basal']['L'])
        if not 'diam' in self.parameters['proximal']:
            self.parameters['proximal']['diam'] = self.parameters['proximal']['area'] / \
            (self.n_proximal_sections()*np.pi*self.parameters['proximal']['L'])
        if not 'diam' in self.parameters['distal']:
            self.parameters['distal']['diam'] = self.parameters['distal']['area'] / \
            (self.n_distal_sections()*np.pi*self.parameters['distal']['L'])

    def setup_topology(self):
        self.soma[0].L = self.parameters['soma']['L']
        self.soma[0].diam = self.parameters['soma']['diam']
        self.proximal[0].L = self.parameters['proximal']['L']
        self.proximal[0].diam = self.parameters['proximal']['diam']
        for d in self.distal:
            d.L = self.parameters['distal']['L']
            d.diam = self.parameters['distal']['diam']
        for d in self.basal:
            d.L = self.parameters['basal']['L']
            d.diam = self.parameters['basal']['diam']
        if self.has_axon:
            self.axon[0].L = 20
            self.axon[0].diam = 3.5 # temporary value
            self.axon[1].L = 25
            self.axon[1].diam = 2.0 # temporary value
            self.axon[2].L = 500
            self.axon[2].diam = 1.5

    def connect_sections(self):
        self.proximal[0].connect(self.soma[0], 1, 0)
        for d in self.distal:
            d.connect(self.proximal[0], 1, 0)
        for d in self.basal:
            d.connect(self.soma[0], 0, 0)
        if self.has_axon:
            self.axon[0].connect(self.soma[0], 0, 0)
            for i in range(1,len(self.axon)):
                self.axon[i].connect(self.axon[i-1], 1, 0)
        
    def insert_active_mech(self):
        try:
            self.insert_fast_Na_and_delayed_rectifier_K()
            print('Inserted fast sodium and delayed rectifier potassium currents.')
        except KeyError:
            if DEBUG:
                print('Not inserting fast sodium and delayed rectifier potassium currents.')
        try:
            self.insert_calcium_dynamics()
            print('Inserted calcium dynamics.')
        except KeyError:
            if DEBUG:
                print('Not inserting calcium dynamics.')
        try:
            self.insert_persistent_Na()
            print('Inserted persistent sodium current.')
        except KeyError:
            if DEBUG:
                print('Not inserting persistent sodium current.')
        try:
            self.insert_Im()
        except KeyError:
            if DEBUG:
                print('Not inserting Im.')
        try:
            self.insert_AHP_K()
            print('Inserted potassium current responsible for AHP.')
        except KeyError:
            if DEBUG:
                print('Not inserting potassium current responsible for AHP.')
        try:
            self.insert_K_D()
        except KeyError:
            if DEBUG:
                print('Not inserting K-D current.')
        try:
            self.insert_A_type_K()
        except KeyError:
            if DEBUG:
                print('Not inserting A-type potassium current.')
        try:
            self.insert_Ih()
        except KeyError:
            if DEBUG:
                print('Not inserting Ih.')
        try:
            self.insert_calcium_current('cal')
        except KeyError:
            if DEBUG:
                print('Not inserting L-type calcium current.')
        try:
            self.insert_calcium_current('cat')
        except KeyError:
            if DEBUG:
                print('Not inserting T-type calcium current.')
        try:
            self.insert_calcium_current('can')
        except KeyError:
            if DEBUG:
                print('Not inserting N-type calcium current.')

    def compute_gbar_at_position(self, distance_from_soma, parameters, max_distance=None):
        if parameters['dend_mode'] == 'passive':
            return 0.
        # the gbar at one of the extrema, i.e. the soma
        if 'gbar_soma' in parameters:
            gbar_0 = parameters['gbar_soma']
        elif 'gbar' in parameters:
            gbar_0 = parameters['gbar']
        else:
            raise Exception('No gbar or gbar_soma in the parameters for mechanism %s' % mech_name)
        # the gbar at the other extremum, i.e. the distal tip
        if 'gbar_distal' in parameters:
            gbar_1 = parameters['gbar_distal']
        elif 'scaling_dend' in parameters:
            gbar_1 = gbar_0 * parameters['scaling_dend']
        else:
            gbar_1 = 0.
        if parameters['dend_mode'] == 'constant':
            g = gbar_1
        elif parameters['dend_mode'] == 'linear':
            g = gbar_0 + distance_from_soma/max_distance * (gbar_1 - gbar_0)
        elif parameters['dend_mode'] == 'exponential':
            g = gbar_1 + (gbar_0 - gbar_1) * np.exp(-distance_from_soma/parameters['lambda'])
        elif parameters['dend_mode'] == 'sigmoidal':
            g = gbar_1 + (gbar_0 - gbar_1) / (1. + np.exp((distance_from_soma - parameters['half_dist'])/parameters['lambda']))
        return max(g*PSUM2_TO_SCM2,0)

    def insert_fast_Na_and_delayed_rectifier_K(self):
        self.parameters['nat']
        # sodium and potassium in the soma and axon (if present)
        if self.has_axon:
            sections = [sec for sec in it.chain(self.soma,self.axon)]
        else:
            sections = self.soma
        for sec in sections:
            sec.insert('hh2')
            sec.ena = 55
            sec.ek = -90
            if sec in self.soma:
                sec.vtraub_hh2 += self.parameters['nat']['vtraub_offset_soma']
            sec.gnabar_hh2 = self.parameters['nat']['gbar_soma'] * PSUM2_TO_SCM2
            sec.gkbar_hh2 = self.parameters['kdr']['gbar_soma'] * PSUM2_TO_SCM2
            if self.has_axon:
                if sec is self.axon[0]:
                    sec.gnabar_hh2 = self.parameters['nat']['gbar_hillock'] * PSUM2_TO_SCM2
                    sec.vtraub_hh2 += self.parameters['nat']['vtraub_offset_hillock']
                elif sec is self.axon[1]:
                    sec.gnabar_hh2 = self.parameters['nat']['gbar_ais'] * PSUM2_TO_SCM2
                    sec.vtraub_hh2 += self.parameters['nat']['vtraub_offset_ais']
                elif sec in self.axon:
                    sec.gnabar_hh2 = self.parameters['nat']['gbar_soma'] * PSUM2_TO_SCM2
                    sec.vtraub_hh2 += self.parameters['nat']['vtraub_offset_soma']

        # sodium and potassium in the dendrites
        if len(self.basal) > 0:
            h.distance(sec=self.soma[0])
            if 'max_dist' in self.parameters['nat']:
                max_dist_Na = self.parameters['nat']['max_dist']
            else:
                max_dist_Na = h.distance(1, sec=self.basal[-1])
            if 'max_dist' in self.parameters['kdr']:
                max_dist_K = self.parameters['kdr']['max_dist']
            else:
                max_dist_K = h.distance(1, sec=self.basal[-1])
            for sec in self.basal:
                sec.insert('hh2')
                sec.ena = 55
                sec.ek = -90
                sec.vtraub_hh2 += self.parameters['nat']['vtraub_offset_soma']
                for seg in sec:
                    dst = h.distance(seg.x,sec=sec)
                    seg.hh2.gnabar = self.compute_gbar_at_position(dst, self.parameters['nat'], max_dist_Na)
                    seg.hh2.gkbar = self.compute_gbar_at_position(dst, self.parameters['kdr'], max_dist_K)
                    if DEBUG:
                        print('gbar INa @ x = %g: %g' % (dst,seg.hh2.gnabar))
                        print('gbar IK @ x = %g: %g' % (dst,seg.hh2.gkbar))
        if len(self.proximal)+len(self.distal) > 0:
            if 'max_dist' in self.parameters['nat']:
                max_dist_Na = self.parameters['nat']['max_dist']
            else:
                max_dist_Na = h.distance(1, sec=self.distal[-1])
            if 'max_dist' in self.parameters['kdr']:
                max_dist_K = self.parameters['kdr']['max_dist']
            else:
                max_dist_K = h.distance(1, sec=self.distal[-1])
            for sec in it.chain(self.proximal,self.distal):
                sec.insert('hh2')
                sec.ena = 55
                sec.ek = -90
                sec.vtraub_hh2 += self.parameters['nat']['vtraub_offset_soma']
                for seg in sec:
                    dst = h.distance(seg.x,sec=sec)
                    seg.hh2.gnabar = self.compute_gbar_at_position(dst, self.parameters['nat'], max_dist_Na)
                    seg.hh2.gkbar = self.compute_gbar_at_position(dst, self.parameters['kdr'], max_dist_K)
                    if DEBUG:
                        print('gbar INa @ x = %g: %g' % (dst,seg.hh2.gnabar))
                        print('gbar IK @ x = %g: %g' % (dst,seg.hh2.gkbar))

    def insert_persistent_Na(self):
        self.parameters['nap']['gbar']
        if self.has_axon:
            sections = [sec for sec in it.chain(self.soma,self.axon)]
        else:
            sections = self.soma
        for sec in sections:
            sec.insert('nap')
            sec.gbar_nap = self.parameters['nap']['gbar'] * PSUM2_TO_SCM2
        #for sec in h.allsec():
        #    if sec in self.soma or (self.has_axon and sec in self.axon):
        #        sec.insert('napinst')
        #        # regular spiking
        #        sec.gbar_napinst = gbar
        #    else:
        #        # FIX: choose which strategy to use for persistent sodium in the dendrites
        #        max_dist += self.soma[0].L
        #        h.distance(sec=self.soma[0])
        #        if h.distance(0, sec=sec) < max_dist:
        #            if DEBUG:
        #                print('Inserting persistent sodium conductance in %s.' % h.secname(sec=sec))
        #            sec.insert('napinst')
        #            for seg in sec:
        #                seg.napinst.gbar = max(self.soma[0].gbar_napinst * (max_dist-h.distance(seg.x,sec=sec))/max_dist,0)
        #                if DEBUG and seg.napinst.gbar > 0:
        #                    print('g_Na @ x = %g: %g' % (seg.x,seg.napinst.gbar))

    def insert_Im(self):
        self.parameters['km']['gbar']
        for sec in self.soma:
            sec.insert('km')
            sec.gbar_km = self.parameters['km']['gbar'] * PSUM2_TO_SCM2

    def insert_AHP_K(self):
        self.parameters['kahp']['gbar']
        for sec in self.soma:
            sec.insert('KahpM95')
            sec.gbar_KahpM95 = self.parameters['kahp']['gbar'] * PSUM2_TO_SCM2
            try:
                h.b0_KahpM95 = 1. / (self.parameters['kahp']['tau'] * h.q10_KahpM95**(0.1*(h.celsius-24)))
            except:
                pass

    def insert_K_D(self):
        self.parameters['kd']['gbar']
        for sec in self.soma:
            sec.insert('kd')
            sec.gkdbar_kd = self.parameters['kd']['gbar'] * PSUM2_TO_SCM2

    def insert_A_type_K(self):
        self.parameters['kap']['gbar']
        if self.has_axon:
            sections = [sec for sec in it.chain(self.soma,self.axon)]
        else:
            sections = self.soma
        for sec in sections:
            sec.insert('kap')
            sec.gkabar_kap = self.parameters['kap']['gbar'] * PSUM2_TO_SCM2
        #for sec in h.allsec():
        #    sec.insert('kap')
        #    sec.gkabar_kap = gbar
        #    if self.has_axon and sec in self.axon:
        #        sec.sh_kap = 0

    def insert_Ih(self):
        self.parameters['ih']['gbar_soma']
        for sec in self.soma:
            sec.insert('hd')
            sec.ghdbar_hd = self.parameters['ih']['gbar_soma'] * PSUM2_TO_SCM2
        # sigmoidally increasing Ih in the dendrites.
        # see Poirazi et al., 2003, Neuron
        h.distance(sec=self.soma[0])
        for sec in it.chain(self.proximal,self.distal):
            sec.insert('hd')
            for seg in sec:
                seg.hd.ghdbar = self.compute_gbar_at_position(h.distance(seg.x,sec=sec), self.parameters['ih'])
                if DEBUG:
                    print('gbar Ih @ x = %g: %g' % (h.distance(seg.x,sec=sec),seg.hd.ghdbar))

    def insert_calcium_dynamics(self):
        self.parameters['ca']
        for sec in it.chain(self.soma,self.proximal,self.distal,self.basal):
            sec.insert('cacum')
            sec.eca = 120
            sec.cai0_cacum = 50e-6
            for seg in sec:
                seg.cacum.depth = min([0.1,seg.diam/2])
            try:
                sec.tau_cacum = self.parameters['ca']['tau']
            except:
                pass

    def insert_calcium_current(self, label):
        if label == 'cat':
            self.parameters[label]['gbar_distal']
            for sec in self.distal:
                sec.insert(label)
                sec.__setattr__('g{0}bar_{0}'.format(label), self.parameters[label]['gbar_distal'])
        else:
            self.parameters[label]['gbar']
            for sec in it.chain(self.soma,self.proximal,self.distal,self.basal):
                sec.insert(label)
                sec.__setattr__('g{0}bar_{0}'.format(label), self.parameters[label]['gbar'])

class SingleCompartmentNeuron (SimplifiedNeuron):
    def __init__(self, parameters, with_axon=False):
        for sec in 'basal','proximal','distal':
            try:
                parameters['soma']['area'] += parameters[sec]['area']
            except:
                pass
        SimplifiedNeuron.__init__(self, parameters, False)

    @classmethod
    def n_somatic_sections(cls):
        return 1

    @classmethod
    def n_basal_sections(cls):
        return 0

    @classmethod
    def n_proximal_sections(cls):
        return 0

    @classmethod
    def n_distal_sections(cls):
        return 0

    @classmethod
    def n_axonal_sections(cls):
        return 0

    def adjust_dimensions(self):
        if 'area' in self.parameters['soma']:
            self.parameters['soma']['diam'] = np.sqrt(self.parameters['soma']['area']/np.pi)
            self.parameters['soma']['L'] = self.parameters['soma']['diam'] / self.n_somatic_sections()

    def setup_topology(self):
        self.soma[0].L = self.parameters['soma']['L']
        self.soma[0].diam = self.parameters['soma']['diam']
    
    def connect_sections(self):
        pass

class AThornyNeuron (SimplifiedNeuron):
    def __init__(self, parameters, with_axon=True):
        SimplifiedNeuron.__init__(self, parameters, with_axon)

    @classmethod
    def n_basal_sections(cls):
        return 2

    @classmethod
    def n_distal_sections(cls):
        return 2

    def connect_sections(self):
        self.proximal[0].connect(self.soma[0], 1, 0)
        for d in self.distal:
            d.connect(self.proximal[0], 1, 0)
        for d in self.basal:
            d.connect(self.soma[0], 0, 0)
        if self.has_axon:
            self.axon[0].connect(self.soma[0], 0, 0)
            for i in range(1,len(self.axon)):
                self.axon[i].connect(self.axon[i-1], 1, 0)
        
class ThornyNeuron (SimplifiedNeuron):
    def __init__(self, parameters, with_axon=True):
        SimplifiedNeuron.__init__(self, parameters, with_axon)

    @classmethod
    def n_basal_sections(cls):
        return 4

    @classmethod
    def n_distal_sections(cls):
        return 6

    def adjust_dimensions(self):
        SimplifiedNeuron.adjust_dimensions(self)
        if 'area' in self.parameters['distal']:
            self.parameters['distal']['L'] /= 2
            self.parameters['distal']['diam'] *= 2

    def connect_sections(self):
        self.proximal[0].connect(self.soma[0], 1, 0)
        self.distal[0].connect(self.proximal[0], 1, 0)
        self.distal[1].connect(self.proximal[0], 1, 0)
        self.distal[2].connect(self.distal[0], 1, 0)
        self.distal[3].connect(self.distal[0], 1, 0)
        self.distal[4].connect(self.distal[1], 1, 0)
        self.distal[5].connect(self.distal[1], 1, 0)
        for d in self.basal:
            d.connect(self.soma[0], 0, 0)
        if self.has_axon:
            self.axon[0].connect(self.soma[0], 0, 0)
            for i in range(1,len(self.axon)):
                self.axon[i].connect(self.axon[i-1], 1, 0)

class SWCNeuron (SimplifiedNeuron):
    def __init__(self, parameters, with_axon=True, convert_to_3pt_soma=False):
        if convert_to_3pt_soma:
            self.swc_filename = '.'.join(parameters['swc_filename'].split('.')[:-1]) + '_converted.swc'
            convert_morphology(parameters['swc_filename'], self.swc_filename)
        else:
            self.swc_filename = parameters['swc_filename']
        parameters['swc_filename'] = os.path.abspath(self.swc_filename)
        parameters['convert_to_3pt_soma'] = convert_to_3pt_soma
        SimplifiedNeuron.__init__(self, parameters, with_axon)

    def make_sections(self):
        # load the tree structure that represents the morphology
        self.tree = btmorph.STree2()
        self.tree.read_SWC_tree_from_file(self.swc_filename,types=range(10))
        # all the sections, indexed by the corresponding index in the SWC file
        self.sections = {}
        # a (temporary) list of all the sections that make up the apical dendrites
        self.apical = []
        # parse the tree!
        for node in self.tree:
            if node is self.tree.root or (not self.has_axon and node.content['p3d'].type == SWC_types['axon']):
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
                h.pt3dadd(float(p_xyz[0]),float(p_xyz[1]),float(p_xyz[2]),2*float(pPos.radius),sec=section)
                h.pt3dadd(float(c_xyz[0]),float(c_xyz[1]),float(c_xyz[2]),2*float(cPos.radius),sec=section)
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
                
    def adjust_dimensions(self):
        pass

    def setup_topology(self):
        pass

    def connect_sections(self):
        for node in self.tree:
            if node is self.tree.root or (not self.has_axon and node.content['p3d'].type == SWC_types['axon']):
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
        del self.apical

    def compute_measures(self):
        Neuron.compute_measures(self)
        self.compute_path_lengths()

    def compute_path_lengths(self):
        axon_root = None
        for node in self.tree:
            if node.content['p3d'].type != SWC_types['axon']:
                # the path length is referred to the root of the tree
                if node is self.tree.root:
                    length = {node.index: 0.}
                else:
                    length[node.index] = length[node.parent.index] + \
                        np.sqrt(np.sum((node.parent.content['p3d'].xyz - node.content['p3d'].xyz)**2))
            else:
                # let's be a little more precise for the axon.
                # the axon root is the section of type ''axon'' that is closest to the soma:
                # we will measure axon length from there instead of from the root of the tree
                if axon_root is None:
                    axon_root = node
                    axon_length = {axon_root.index: path_length(self.tree.path_between_nodes(axon_root, axon_root.parent))}
                else:
                    axon_length[node.index] = axon_length[node.parent.index] + \
                        np.sqrt(np.sum((node.parent.content['p3d'].xyz - node.content['p3d'].xyz)**2))
        self.soma_lengths = []
        for sec in self.soma:
            self.soma_lengths.append(length[self.section_index(sec)])
        self.basal_lengths = []
        for sec in self.basal:
            self.basal_lengths.append(length[self.section_index(sec)])
        self.proximal_lengths = []
        for sec in self.proximal:
            self.proximal_lengths.append(length[self.section_index(sec)])
        self.distal_lengths = []
        for sec in self.distal:
            self.distal_lengths.append(length[self.section_index(sec)])
        if self.has_axon:
            self.axon_lengths = []
            for sec in self.axon:
                self.axon_lengths.append(axon_length[self.section_index(sec)])

    def path_length_to_root(self, node):
        return path_length(self.tree.path_to_root(node))

    def section_index(self, section):
        return int(h.secname(sec=section).split('_')[-1])

class SimplifiedNeuron3D (SimplifiedNeuron):
    def __init__(self, parameters):
        SimplifiedNeuron.__init__(self, parameters)

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

def neuron_factory(filename):
    data = h5.load_h5_file(filename)
    if data['neuron_type'] == 'SimplifiedNeuron':
        cls = SimplifiedNeuron
    elif data['neuron_type'] == 'AThornyNeuron':
        cls = AThornyNeuron
    elif data['neuron_type'] == 'ThornyNeuron':
        cls = ThornyNeuron
    elif data['neuron_type'] == 'SWCNeuron':
        cls = SWCNeuron
    elif data['neuron_type'] == 'SimplifiedNeuron3D':
        cls = SimplifiedNeuron3D
    else:
        raise Exception('Unknown neuron model: %s.' % data['neuron_type'])
    return cls(data['parameters'], data['has_axon'])

def run_step_single(amplitude=0.125):
    parameters = {'soma': {'Cm': 1., 'Ra': 100., 'El': -70., 'area': 40000},
                  'swc_filename': '../../morphologies/DH070613-1-.Edit.scaled.swc'}

    Rin = 50e6
    parameters['soma']['Rm'] = Rin * parameters['soma']['area'] * 1e-8
    # fast sodium current
    parameters['nat'] = {'gbar_soma': 100, 'vtraub_offset_soma': -5.}
    # delayed rectifier potassium
    parameters['kdr'] = {'gbar_soma': 30}
    # calcium dynamics
    #parameters['ca'] = {'tau': 100} # [ms]
    # persistent sodium
    #parameters['nap'] = {'gbar': 5}
    # muscarinic potassium
    #parameters['km'] = {'gbar': 20}
    # ahp potassium
    #parameters['kahp'] = {'gbar': 300, 'tau': 100}
    # K-D
    #parameters['kd'] = {'gbar': 1e-4}
    # A-type potassium
    #parameters['kap'] = {'gbar': 10}
    # Ih current
    #parameters['ih'] = {'gbar_soma': 1e-2}

    n = SingleCompartmentNeuron(parameters)
    #n.save_properties()
    #h.topology()
    #print('gnabar = %f\ngkbar = %f\n' % (n.soma[0].gnabar_hh2,n.soma[0].gkbar_hh2))
    rec = make_voltage_recorders(n)
    #rec['ahp'] = h.Vector()
    #rec['ahp'].record(n.soma[0](0.5)._ref_gkahp_KahpM95)
    stim = h.IClamp(n.soma[0](0.5))
    stim.delay = 100
    stim.amp = amplitude
    stim.dur = 1000
    run(tend=stim.dur+stim.delay*3,V0=-70)

    #t = np.arange(0,stim.delay,h.dt)
    #dV = np.min(np.array(rec['vsoma'])) - n.soma[0].e_pas
    #tau = n.soma[0].cm * parameters['soma']['area']*1e-8 * Rin * 1e-3
    #V = n.soma[0].e_pas + dV * np.exp(-t/tau)
    #print('Input resistance: %.0f MOhm.\nMembrane time constant: %.1f ms.\n' % (Rin*1e-6,tau))

    import pdb; pdb.set_trace()
    
    import pylab as p
    p.plot(rec['t'],rec['vsoma'],'k',label='Soma',lw=2)
    #p.plot(t+stim.dur+stim.delay,V,'r',lw=1)
    #p.plot(rec['t'],rec['ahp'],'k',label='Soma',lw=1)
    p.xlabel('Time (ms)')
    p.ylabel('Membrane voltage (mV)')
    p.legend(loc='best')
    p.show()

def run_step(amplitude=0.4):
    parameters = {'scaling': 0.5,
                  'soma': {'Cm': 1., 'Ra': 100., 'El': -70., 'Rm': 10e3, 'area': 1500},
                  'proximal': {'Ra': 500., 'El': -70., 'L': 500., 'diam': 5.},
                  'distal': {'Ra': 500., 'El': -70., 'L': 200., 'area': 1500.},
                  'basal': {'Ra': 500., 'El': -70., 'L': 300., 'diam': 5.}}
    # the passive properties of the axon are the same as the soma
    parameters['axon'] = parameters['soma'].copy()
    # fast sodium current
    with_axon = True
    #parameters['nat'] = {'gbar_soma': 50, 'gbar_distal': 5, 'lambda': 50, 'dend_mode': 'exponential'}
    parameters['nat'] = {'gbar_soma': 100, 'dend_mode': 'linear'}
    #parameters['nat'] = {'gbar_soma': 100, 'scaling_dend': 0.2, 'dend_mode': 'constant'}
    #parameters['nat'] = {'gbar_soma': 100, 'dend_mode': 'passive'}
    if with_axon:
        parameters['nat']['gbar_hillock'] = 1000
        parameters['nat']['gbar_ais'] = 2500
    # delayed rectifier potassium
    #parameters['kdr'] = {'gbar_soma': 50, 'gbar_distal': 20, 'lambda': 50, 'dend_mode': 'exponential'}
    parameters['kdr'] = {'gbar_soma': 20, 'dend_mode': 'linear'}
    # persistent sodium
    #parameters['nap'] = {'gbar': 1e-4}
    # muscarinic potassium
    #parameters['km'] = {'gbar': 20}
    # ahp potassium
    #parameters['kahp'] = {'gbar': 300}
    # K-D
    #parameters['kd'] = {'gbar': 1e-4}
    # A-type potassium
    #parameters['kap'] = {'gbar': 10}
    # Ih current
    #parameters['ih'] = {'gbar_soma': 1e-2, 'scaling_dend': 10., 'half_dist': 100., 'lambda': 30., 'dend_mode': 'sigmoidal'}

    n = SimplifiedNeuron(parameters,with_axon=with_axon)
    #n = AThornyNeuron(parameters,with_axon=True)
    #n = ThornyNeuron(parameters,with_axon=True)
    #n = SWCNeuron(parameters,with_axon=False,convert_to_3pt_soma=False)
    #n.save_properties()
    h.topology()
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
    if n.has_axon:
        p.plot(rec['t'],rec['vhillock'],'c',label='Hillock')
        p.plot(rec['t'],rec['vais'],'m',label='AIS')
        p.plot(rec['t'],rec['vaxon'],'y',label='Axon')
    p.xlabel('Time (ms)')
    p.ylabel('Membrane voltage (mV)')
    p.legend(loc='best')
    p.show()

if __name__ == '__main__':
    run_step_single()
