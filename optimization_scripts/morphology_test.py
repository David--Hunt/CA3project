#!/usr/bin/env python

from bluepyopt import ephys
from bluepyopt.ephys.morphologies import NrnFileMorphology
from neuron import h
from dlmorph import SWCFileSimplifiedMorphology
import sys

# the Neuron simulator
sim = ephys.simulators.NrnSimulator()

swc_filename = '/Users/daniele/Postdoc/Research/Janelia/SWCs/FINAL/thorny/DH070813-.Edit.scaled.converted.swc'
if len(sys.argv) == 2 and sys.argv[1] == 'new':
    morpho = SWCFileSimplifiedMorphology(swc_filename)
else:
    morpho = NrnFileMorphology(swc_filename)

locations = []
for loc in ('somatic','axonal','apical','basal'):
    locations.append(ephys.locations.NrnSeclistLocation(loc, seclist_name=loc))

# let's create a passive mechanism
pas_mech = ephys.mechanisms.NrnMODMechanism(name='pas',suffix='pas',locations=locations)

# let's create a cell model
CA3_cell = ephys.models.CellModel(name='CA3_cell',morph=morpho,mechs=[pas_mech])

# instantiate the cell model
CA3_cell.instantiate(sim)




