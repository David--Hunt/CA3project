#!/usr/bin/env python

from bluepyopt import ephys
from bluepyopt.ephys.morphologies import NrnFileMorphology
from neuron import h
import btmorph

__all__ = ['SWCFileSimplifiedMorphology']

class SWCFileSimplifiedMorphology(NrnFileMorphology):
    """SWCFileSimplifiedMorphology"""

    SWC_types = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4}
    SWC_types_inverse = {v:k for k,v in SWC_types.iteritems()}

    def __init__(self, morphology_path, do_replace_axon=False,
                 do_set_nseg=True, comment='', replace_axon_hoc=None):
        super(SWCFileSimplifiedMorphology, self).__init__(morphology_path,
                                                          do_replace_axon, do_set_nseg,
                                                          comment, replace_axon_hoc)

    def instantiate(self, sim=None, icell=None):
        tree = btmorph.STree2()
        tree.read_SWC_tree_from_file(self.morphology_path,types=range(10))

        # all the sections
        self.sections = []
        sections_map = {}
        sections_connections = []
        # parse the tree!
        for node in tree:
            if node is tree.root:
                section = sim.neuron.h.Section(name='{0}_{1}'.format(
                    SWCFileSimplifiedMorphology.SWC_types_inverse[node.content['p3d'].type],node.index))
                self.sections.append(section)
                sections_map[node.index] = len(self.sections)-1
                sim.neuron.h.pt3dclear(sec=section)
                icell.somatic.append(sec=section)
            elif len(node.children) == 1 and len(node.parent.children) > 1:
                # the parent of the current node is a branching point: start a new section
                section = sim.neuron.h.Section(name='{0}_{1}'.format(
                    SWCFileSimplifiedMorphology.SWC_types_inverse[node.content['p3d'].type],node.index))
                self.sections.append(section)
                sections_connections.append((len(self.sections)-1,sections_map[node.parent.index]))
                sections_map[node.index] = len(self.sections)-1
                sim.neuron.h.pt3dclear(sec=section)
                # assign it to the proper region
                swc_type = node.content['p3d'].type
                if swc_type == SWCFileSimplifiedMorphology.SWC_types['soma']:
                    icell.somatic.append(sec=section)
                    #sim.neuron.h.distance(sec=soma[0]) ### THIS IS WRONG!
                elif swc_type == SWCFileSimplifiedMorphology.SWC_types['axon']:
                    icell.axonal.append(sec=section)
                elif swc_type == SWCFileSimplifiedMorphology.SWC_types['basal']:
                    icell.basal.append(sec=section)
                elif swc_type == SWCFileSimplifiedMorphology.SWC_types['apical']:
                    icell.apical.append(sec=section)
            else:
                sections_map[node.index] = sections_map[node.parent.index]
                section = self.sections[sections_map[node.parent.index]]
            xyz = node.content['p3d'].xyz
            sim.neuron.h.pt3dadd(float(xyz[0]),float(xyz[1]),float(xyz[2]),2*float(node.content['p3d'].radius),sec=section)
        
        for i,j in sections_connections:
            self.sections[i].connect(self.sections[j], 1, 0)

        for sec in self.sections:
            icell.all.append(sec=sec)

        ### the following code is taken from the instantiate method in the parent class
        # TODO Set nseg should be called after all the parameters have been
        # set
        # (in case e.g. Ra was changed)
        if self.do_set_nseg:
            self.set_nseg(icell)

        # TODO replace these two functions with general function users can
        # specify
        if self.do_replace_axon:
            self.replace_axon(sim=sim, icell=icell)
            
    @staticmethod
    def set_nseg(icell):
        """Set the nseg of each section using the lambda rule"""
        nseg = 0
        for section in icell.all:
            section.nseg = int((section.L/(0.1*h.lambda_f(100,sec=section))+0.9)/2)*2 + 1
            nseg += section.nseg


