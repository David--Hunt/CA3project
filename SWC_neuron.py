import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools as it
import btmorph
import neuron
from neuron import h
h.load_file('stdrun.hoc')

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

def simplify_tree(node, min_distance, spare_types=(SWC_types['soma'],SWC_types['axon']), removed=None):
    if not node is None:
        if not node.parent is None and len(node.children) == 1 and not node.content['p3d'].type in spare_types:
            length = path_length([node.parent,node,node.children[0]])
            if length < min_distance:
                node.parent.children[node.parent.children.index(node)] = node.children[0]
                node.children[0].parent = node.parent
                if not removed is None:
                    removed.append(node.index)
                node = node.parent
        for child in node.children:
            simplify_tree(child, min_distance, spare_types, removed)

def convert_morphology(filename_in, filename_out):
    import numpy as np
    original_morphology = np.loadtxt(filename_in)
    idx, = np.where(original_morphology[:,1] == SWC_types['soma'])
    soma_ids = original_morphology[idx,0]
    center = np.mean(original_morphology[idx,2:5],axis=0)
    original_morphology[:,2:5] -= center
    radius = np.mean(np.sqrt(np.sum(original_morphology[idx,2:5]**2,axis=1)))
    converted_morphology = [[1, SWC_types['soma'], 0, 0, 0, radius, -1],
                            [2, SWC_types['soma'], 0, -radius, 0, radius, 1],
                            [3, SWC_types['soma'], 0, radius, 0, radius, 1]]
    for entry in original_morphology:
        if entry[1] != SWC_types['soma']:
            if entry[-1] in soma_ids:
                entry[-1] = 1
            converted_morphology.append(entry.tolist())
    np.savetxt(filename_out, converted_morphology, '%g')

def compute_section_area(section):
    a = 0.
    for segment in section:
        a += h.area(segment.x, sec=section)
    return a

class SWCCell(object):
    def __init__(self, f_name, El, Rm, Ra, Cm, min_distance=0., convert_to_3pt_soma=True):
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
        if convert_to_3pt_soma:
            self.swc_filename = '.'.join(f_name.split('.')[:-1]) + '_converted.swc'
            convert_morphology(f_name, self.swc_filename)
        else:
            self.swc_filename = f_name

        # parameters
        self.El = El
        self.Rm = Rm
        self.Cm = Cm
        self.Ra = Ra
        self.min_distance = min_distance

        self.load_morphology()
        self.compute_measures()
        self.insert_passive_mech()
        self.insert_active_mech()

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
            simplify_tree(self.tree.root, self.min_distance, (SWC_types['soma'],SWC_types['axon']), removed)
            sys.stdout.write('removed %d nodes out of %d.\n' % (len(removed),total))
            self.simplified_swc_filename = '.'.join(self.swc_filename.split('.')[:-1]) + \
                '_simplified_%g_um.swc' % self.min_distance
            self.tree.write_SWC_tree_to_file(self.simplified_swc_filename)
        # all the sections, indexed by the corresponding index in the SWC file
        self.sections = {}
        # a list of all the sections that make up the soma
        self.soma = []
        # a list of all the sections that make up the axon
        self.axon = []
        # a list of all the sections that make up the basal dendrites
        self.basal = []
        # a list of all the sections that make up the apical dendrites
        self.apical = []
        # parse the tree!
        for node in self.tree:
            if node is self.tree.root:
                continue
            section = h.Section(name='sec_{0}'.format(node.index))
            swc_type = node.content['p3d'].type
            if swc_type == SWC_types['soma']:
                section.cm = self.Cm['soma']
                section.Ra = self.Ra['soma']
                self.soma.append(section)
            elif swc_type == SWC_types['axon']:
                section.cm = self.Cm['axon']
                section.Ra = self.Ra['axon']
                self.axon.append(section)
            elif swc_type == SWC_types['basal']:
                section.cm = self.Cm['dend']
                section.Ra = self.Ra['dend']
                self.basal.append(section)
            elif swc_type == SWC_types['apical']:
                section.cm = self.Cm['dend']
                section.Ra = self.Ra['dend']
                self.apical.append(section)
            if not node.parent is None:
                pPos = node.parent.content['p3d']
                cPos = node.content['p3d']
                c_xyz = cPos.xyz
                p_xyz = pPos.xyz
                h.pt3dclear(sec=section)
                h.pt3dadd(float(p_xyz[0]),float(p_xyz[1]),float(p_xyz[2]),float(pPos.radius),sec=section)
                h.pt3dadd(float(c_xyz[0]),float(c_xyz[1]),float(c_xyz[2]),float(cPos.radius),sec=section)
                # nseg according to NEURON book; too high in general...
                #section.nseg = int(((section.L/(0.1*h.lambda_f(100))+0.9)/2)*2+1)
                section.nseg = int((section.L/(0.1*h.lambda_f(100))+0.9)/2)*2 + 1
                try:
                    section.connect(self.sections[node.parent.index],1,0)
                except:
                    if not section is self.soma[0]:
                        section.connect(self.soma[0],1,0)
            self.sections[node.index] = section

    def compute_measures(self):
        # the areas of all sections in the soma
        self.soma_areas = []
        for sec in self.soma:
            self.soma_areas.append(compute_section_area(sec))

        # the areas of all sections in the apical dendrites
        self.apical_areas = []
        # the distances of all apical sections from the soma: here the
        # distance is just the Euclidean one, not the path length
        self.apical_distances = []
        for sec in self.apical:
            self.apical_distances.append(self.distance_from_soma(sec))
            self.apical_areas.append(compute_section_area(sec))

        # the areas of all sections in the basal dendrites
        self.basal_areas = []
        # the distances of all basal sections from the soma
        self.basal_distances = []
        for sec in self.basal:
            self.basal_distances.append(self.distance_from_soma(sec))
            self.basal_areas.append(compute_section_area(sec))

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
        # the areas of the sections in the axon
        self.axon_areas = []
        # the path length of each section in the axon from the root
        self.axon_length = []
        for sec in self.axon:
            self.axon_areas.append(compute_section_area(sec))
            index = self.section_index(sec)
            self.axon_length.append(length[index])

        self.total_area = np.sum(self.soma_areas) + np.sum(self.apical_areas) + \
            np.sum(self.basal_areas) + np.sum(self.axon_areas)

    def insert_passive_mech(self):
        #Rm = self.Rm['soma'] * 1e6 * self.total_area * 1e-8   # ohm cm^2
        print('Total area: %.0f um^2.' % self.total_area)
        #print('Rm = %.2f kOhm cm^2.' % (Rm*1e-3))
        #print('g_pas = %f S cm^-2.' % (1./Rm))
        for sec in h.allsec():
            sec.insert('pas')
            sec.e_pas = self.El
            if sec in self.soma:
                sec.g_pas = 1./self.Rm['soma']
            elif sec in self.axon:
                sec.g_pas = 1./self.Rm['axon']
            else:
                sec.g_pas = 1./self.Rm['dend']

    def insert_active_mech(self):
        """ Must be implemented by the subclasses. """
        pass

    def distance_from_soma(self, sec, x=0.5):
        return distance(self.soma[0], sec, x)

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
    def __init__(self, swc_filename, Rm, Ra, Cm, min_distance=0., convert_to_3pt_soma=True):
        SWCCell.__init__(self, swc_filename, -65., Rm, Ra, Cm, min_distance, convert_to_3pt_soma)
        
    def insert_active_mech(self):
	##### SOMA
        for sec in self.soma:
            #sec.insert('pas') # add passive properties
            #sec.g_pas = 0.00002        #1./(self.Rm['soma']*1000)#0.00001
            #sec.e_pas = -60
            sec.insert('hh2') # add Na+ and K+ HH conductances
            sec.ek = -90
            sec.ena = 50
            sec.vtraub_hh2 = -55
            sec.gnabar_hh2 = 0.05
            sec.gkbar_hh2 = 0.005
            sec.insert('im') # add M conductances
            h.taumax_im = 1000
            sec.gkbar_im = 1e-3
            sec.insert('cad') # add Ca2+ decay dynamics
            sec.depth_cad = 1
            sec.taur_cad= 5
            sec.cainf_cad = 2.4e-4
            sec.kt_cad = 0
            sec.insert('it') # add T-type Ca2+ conductance
            sec.cai = 2.4e-4
            sec.cao = 2
            sec.eca = 120
            sec.gcabar_it = 0.002
            sec.insert('ical') # add L-type Ca2+ conductance	
            sec.gcabar_ical = 1e-4
            sec.insert('KahpM95') # add calcium activated potassium conductance (I_ahp)
            #sec.cai = 50e-6 # !!! Ca^i is just one value per section
            sec.gbar_KahpM95 = 0.03
            sec.insert('kd') # add delay rectifier K+ conductance
            sec.ek = -90
            sec.gkdbar_kd = 1e-5
            sec.insert('napinst') # add persistent Na+ conductance
            sec.ena = 50
            sec.gbar_napinst = 1e-8

	##### DENDRITES
	for sec in it.chain(self.basal,self.apical):
            #sec.insert('pas') # add passive properties
            #sec.g_pas = 0.0005 # 1./(self.Rm['dend']*1000)
            #sec.e_pas = -70
            sec.insert('hh2') # add hodgkin huxley Na+ and K+ conductances
	    sec.ek = -90
            sec.ena = 50
            sec.vtraub_hh2 = -55
            sec.gnabar_hh2 = 0.05
            sec.gkbar_hh2 = 0.005
            sec.insert('cad') # add Ca2+ decay dynamics
            sec.depth_cad = 1
            sec.taur_cad = 5
            sec.cainf_cad = 2.4e-4
            sec.kt_cad = 0
            sec.insert('it') # add T-type Ca2+ conductance
            sec.cai = 2.4e-4
            sec.cao = 2
            sec.eca = 120
            sec.gcabar_it = 0.002	
            sec.insert('ical') # add L-type Ca2+ conductance
            sec.gcabar_ical = 1e-4
            sec.insert('KahpM95') # add calcium activated potassium conductance
            #sec.cai = 50e-6 # !!! Ca^i is just one value per section
            sec.gbar_KahpM95 = 0.02
            sec.insert('kd') # add delayed rectifier K+ conductance
            sec.gkdbar_kd = 1e-4

        ##### AXON
        for sec,distance in zip(self.axon,self.axon_length):
            #sec.insert('pas')
            #sec.g_pas = 1./(self.Rm['axon']*1000)
            #sec.e_pas = -70
            sec.insert('hh2')
            sec.vtraub_hh2 = -55
            sec.gkbar_hh2 = 0.006
            sec.gnabar_hh2 = 0.05
            if distance < 10:
                # AIS
                sec.gnabar_hh2 = 0.25


#------------------------------Mechanisms for Bursting properties-------------------------
class IBCell(SWCCell):
    def __init__(self, swc_filename, Rm, Ra, Cm, min_distance=0., convert_to_3pt_soma=True):
        SWCCell.__init__(self, swc_filename, -65., Rm, Ra, Cm, min_distance, convert_to_3pt_soma)
    
    def insert_active_mech(self):
        ##### SOMA
        for sec in self.soma:
            #sec.insert('pas') # add passive properties
            #sec.g_pas = 0.00005        #1./(self.Rm['soma']*1000)#0.00001
            #sec.e_pas = -70
            sec.insert('hh2') # add Na+ and K+ HH conductances
            sec.ek = -80
            sec.ena = 50
            sec.vtraub_hh2 = -55
            sec.gnabar_hh2 = 0.05
            sec.gkbar_hh2 = 0.005
            sec.insert('im') # add M conductances
            h.taumax_im = 1000
            sec.gkbar_im = 1e-5
            sec.insert('cad') # add Ca2+ decay dynamics
            sec.depth_cad = 1
            sec.taur_cad = 5
            sec.cainf_cad = 2.4e-4
            sec.kt_cad = 0
            sec.insert('it') # add T-type Ca2+ conductance
            sec.cai = 2.4e-4
            sec.cao = 2
            sec.eca = 120
            sec.gcabar_it = 5e-4
            sec.insert('ical') # add L-type Ca2+ conductance
            sec.gcabar_ical = 5e-4
            sec.insert('KahpM95') # add calcium activated potassium conductance (I_ahp)
            #sec.cai = 50e-6
            sec.gbar_KahpM95 = 0.0007
            sec.insert('kd') # add delay rectifier K+ conductance
            sec.gkdbar_kd = 1e-5
            sec.insert('napinst') # add persistent Na+ conductance
            sec.gbar_napinst = 5e-5

        ##### DENDRITES
        for sec in it.chain(self.basal,self.apical):
            #sec.insert('pas') # add passive properties
            #sec.g_pas = 0.0001 # 1./(self.Rm['dend']*1000)
            #sec.e_pas = -70
            sec.insert('hh2') # add hodgkin huxley Na+ and K+ conductances
            sec.ek = -80
            sec.ena = 50
            sec.vtraub_hh2 = -55
            sec.gnabar_hh2 = 0.05
            sec.gkbar_hh2 = 0.005
            sec.insert('cad') # add Ca2+ decay dynamics
            sec.cai = 2.4e-4
            sec.cao = 2
            sec.eca = 120
            sec.depth_cad = 1
            sec.taur_cad = 5
            sec.cainf_cad = 2.4e-4
            sec.kt_cad = 0
            sec.insert('it') # add T-type Ca2+ conductance
            sec.gcabar_it = 1e-3
            sec.insert('ical') # add L-type Ca2+ conductance
            sec.gcabar_ical = 9e-4
            sec.insert('KahpM95') # add calcium activated potassium conductance
            #sec.cai = 50e-6
            sec.gbar_KahpM95 = 0.001
            sec.insert('kd') # add delayed rectifier K+ conductance
            sec.gkdbar_kd = 1e-5
        
        ##### AXON
        for sec,distance in zip(self.axon,self.axon_length):
            #sec.insert('pas')
            #sec.g_pas = 1./(self.Rm['axon']*1000)
            #sec.e_pas = -70
            sec.insert('hh2')
            sec.vtraub_hh2 = -55
            sec.gkbar_hh2 = 0.006
            sec.gnabar_hh2 = 0.05
            if distance < 10:
                # AIS
                sec.gnabar_hh2 = 0.25

#### so far this model is not working
class CA3RS(SWCCell):
    def __init__(self, swc_filename, Rm, Ra, Cm, min_distance=0., convert_to_3pt_soma=True):
        """
        Hippocampal CA3 pyramidal neuron model from the paper 
        M. Migliore, E. Cook, D.B. Jaffe, D.A. Turner and D. Johnston,
        Computer simulations of morphologically reconstructed CA3 hippocampal neurons,
        J. Neurophysiol. 73, 1157-1168 (1995).
        """
        SWCCell.__init__(self, swc_filename, -65., Rm, Ra, Cm, min_distance, convert_to_3pt_soma)
        ### regular spiking
	self.gna = 0.015
	self.gkdr = 0.03
	self.gka = 0.001
	self.gkm = 0.0001
        self.gpas = 0.00018 # 1./60000.
        ### bursting
        #self.gna = 0.015
	#self.gkdr = 0.009
	#self.gka = 0.0001
	#self.gkm = 0.00002
        self.insert_mechanisms()

    def insert_mechanisms(self):
        print('Inserting active mechanisms...')
        for sec in self.axon:
            sec.Ra = 200
            sec.insert('pas')
            for seg in sec:
                seg.pas.e = -65.
                seg.pas.g = self.gpas

        # insert mechanisms that are present everywhere
        for sec in it.chain(self.soma,self.basal,self.apical):
            sec.Ra = 200.
            sec.insert('pas')
            #sec.insert('cadifus')
            #sec.insert('cal')
            #sec.insert('can')
            #sec.insert('cat')
            #sec.insert('kahp')
            #sec.insert('cagk')
            for seg in sec:
                seg.pas.e = -65.
                seg.pas.g = self.gpas
            #    seg.cal.gcalbar = 0.0025
            #    seg.can.gcanbar = 0.0025
            #    seg.cat.gcatbar = 0.00025
            #    seg.kahp.gkahpbar = 0.0004
            #    seg.cagk.gkbar = 0.00055
            #    seg.cao = 2.
            #    seg.cai = 50e-6
            #    seg.ek = -91
                seg.v = -65

        count = 0
        max_dist = 1000
        for sec in it.chain(self.soma,
                            filter(self.basal,self.basal_distances,max_dist)[0],
                            filter(self.apical,self.apical_distances,max_dist)[0]):
            sec.insert('nahh')
            sec.insert('borgkdr')
            #sec.insert('borgka')
            #sec.insert('borgkm')
            for seg in sec:
                seg.nahh.gnabar = self.gna
                seg.borgkdr.gkdrbar = self.gkdr
            #    seg.borgka.gkabar = self.gka
            #    seg.borgkm.gkmbar = self.gkm
                seg.ek = -91.
                seg.ena = 50.
            count += 1

        print('Added sodium channels to %d sections (%d are in the soma).' % (count,len(self.soma)))
	h.finitialize(-65.)
        h.fcurrent()
        #for sec in it.chain(self.soma,self.basal,self.apical):
        #    if h.ismembrane('nahh',sec=sec):
        #        for seg in sec:
        #            seg.pas.e = seg.v + (seg.ina+seg.ik+seg.ica)/seg.pas.g
        #    else:
        #        for seg in sec:
        #            seg.pas.e = seg.v + (seg.ik+seg.ica)/seg.pas.g
        h.cvode.re_init()

class CA3b(SWCCell):
    def __init__(self, swc_filename, gbars, Rm, Ra, Cm, kmultp=0.02, axonm=5, min_distance=0., convert_to_3pt_soma=True):
        """
        Hippocampal CA3b pyramidal cell model from the paper
        Hemond, P., Epstein, D., Boley, A., Migliore, M., Ascoli, G. A., & Jaffe, D. B. (2008).
        Distinct classes of pyramidal cells exhibit mutually exclusive firing patterns in hippocampal area CA3b.
        Hippocampus, 18(4), 411-424. doi:10.1002/hipo.20404
        """
        SWCCell.__init__(self, swc_filename, -65., Rm, Ra, Cm, min_distance, convert_to_3pt_soma)
        self.gbars = self.fill_with_default_gbars(gbars)
        self.somatic_mech = ['kap','cacum']
        self.axonal_mech = ['kap']
        self.dendritic_mech = ['kap','cacum']
        for k,v in gbars.iteritems():
            if v > 0:
                if not k in self.somatic_mech:
                    self.somatic_mech.append(k)
                if k in ('na3','kdr') and not k in self.axonal_mech:
                    self.axonal_mech.append(k)
                if not k in self.dendritic_mech and k != 'km':
                    self.dendritic_mech.append(k)
        self.Vrest = -64
        self.Rm = 25370.
        self.Cm = 1.41
        self.RaAll = 150
        self.kmultp = kmultp
        self.axonm = axonm
        self.insert_mechanisms()

    def fill_with_default_gbars(self, gbars):
        gc = 1e-5
        defaults = {'na': 0.022, 'kdr': 0.005, 'kc': 5e-5, 'km': 0.017, 'kd': 0.0, 'ahp': 0.0001,
                 'cal': gc, 'can': gc, 'cat': gc, 'hd': 0.00001}
        for k,v in defaults.iteritems():
            if not k in gbars:
                gbars[k] = v
        gbars['na3'] = gbars['na']
        gbars['KahpM95'] = gbars['ahp']
        gbars['cagk'] = gbars['kc']
        gbars.pop('na')
        gbars.pop('ahp')
        gbars.pop('kc')
        return gbars

    def insert_mechanisms(self):
        print('Inserting active mechanisms...')
        sys.stdout.write('Somatic mechanisms: ')
        print(self.somatic_mech)
        sys.stdout.write('Dendritic mechanisms: ')
        print(self.dendritic_mech)
        sys.stdout.write('Axonal mechanisms: ')
        print(self.axonal_mech)
        for sec in self.soma:
            for mech in self.somatic_mech:
                sec.insert(mech)
            for seg in sec:
                seg.cacum.depth = seg.diam/2
        for sec in self.axon:
            for mech in self.axonal_mech:
                sec.insert(mech)
        for sec in it.chain(self.apical,self.basal):
            for mech in self.dendritic_mech:
                sec.insert(mech)
            for seg in sec:
                seg.cacum.depth = seg.diam/2
        #### not sure whether this is correct
        h.ehd_hd = -30.
        for sec in h.allsec():
            sec.insert('pas')
            sec.v = self.Vrest
            sec.e_pas = self.Vrest
            sec.g_pas = 1./self.Rm
            sec.Ra = self.RaAll
            sec.cm = self.Cm
            sec.ek = -90.
            sec.ena = 55.
            if sec in self.axon:
                sec.Ra = self.RaAll/3
            #if h.ismembrane('hd',sec=sec):
            #    sec.ehd_hd = -30
            if h.ismembrane('cal',sec=sec):
                sec.gcalbar_cal = self.gbars['cal']
                sec.gcanbar_can = self.gbars['can']
                sec.gcatbar_cat = self.gbars['cat']
            if h.ismembrane('cagk',sec=sec):
                sec.gbar_cagk = self.gbars['cagk']
            if h.ismembrane('KahpM95',sec=sec):
                sec.gbar_KahpM95 = self.gbars['KahpM95']

        for sec in self.axon:
            sec.gbar_na3 = self.gbars['na3'] * self.axonm
            sec.gkdrbar_kdr = self.gbars['kdr']
            sec.gkabar_kap = self.kmultp
            sec.sh_kap = 0

        for sec in self.soma:
            sec.ghdbar_hd = self.gbars['hd']
            sec.gbar_na3 = self.gbars['na3']
            sec.gkdrbar_kdr = self.gbars['kdr']
            sec.gkabar_kap = self.kmultp
            if h.ismembrane('km',sec=sec):
                sec.gbar_km = self.gbars['km']
            if h.ismembrane('kd',sec=sec):
                sec.gkdbar_kd = self.gbars['kd']

        for sec in it.chain(self.basal,self.apical):
            sec.ghdbar_hd = self.gbars['hd']
            sec.gbar_na3 = self.gbars['na3']
            sec.gkdrbar_kdr = self.gbars['kdr']
            sec.gkabar_kap = self.kmultp

	h.finitialize(self.Vrest)
        h.fcurrent()
	h.finitialize(self.Vrest)
        for sec in h.allsec():
            for seg in sec:
                if h.ismembrane('cal',sec=sec):
                    seg.pas.e = seg.v + (seg.hd.i + seg.ina + seg.ik + seg.ica)/seg.pas.g
                else:
                    seg.pas.e = seg.v + (seg.ina + seg.ik)/seg.pas.g

	h.cvode.re_init()


def test_CA3b(fig='9d'):
    gbars = {'9b': {},
             '9c': {'kdr': 0.01, 'kc': 0., 'cal': 0., 'can': 0., 'cat': 0.},
             '9d': {'kdr': 0.01, 'kc': 0., 'cal': 0., 'can': 0., 'cat': 0., 'km': 0.},
             '9e': {'kdr': 0.01, 'kc': 0., 'km': 0., 'kd': 0.0011, 'ahp': 0.}}
    if not fig in gbars:
        sys.exit(1)
    h.cvode_active(1)
    h.cvode.atol(1e-6)
    h.cvode.rtol(1e-6)
    cell = CA3b(swc_filename = '../morphologies/DH070313-.Edit.scaled.swc',
                gbars = gbars[fig], kmultp=0.02, axonm=5, Rm = {'axon': 100, 'soma': 150, 'dend': 75},
                Ra = {'axon': 100, 'soma': 75, 'dend': 75},
                min_distance = 10.)
    stim = h.IClamp(cell.soma[0](0.5))
    stim.delay = 100
    stim.dur = 500
    stim.amp = 0.3
    rec = {'t': h.Vector(), 'v': h.Vector()}
    rec['t'].record(h._ref_t)
    rec['v'].record(cell.soma[0](0.5)._ref_v)
    h.tstop = 700
    h.celsius = 36
    h.run()
    import pylab as p
    p.plot(rec['t'],rec['v'],'k')
    p.show()

def main():
    test_CA3b()

if __name__ == '__main__':
    main()
