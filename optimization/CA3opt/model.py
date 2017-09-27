"""Run simple cell optimisation"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
# pylint: disable=R0914

import os
import json

import bluepyopt.ephys as ephys
from . import morphology

__all__ = ['define_mechanisms','define_parameters','define_morphology','create_cell']

def define_mechanisms(config_dir='config',mechanisms_file='mechanisms.json'):
    """Define mechanisms"""

    mech_definitions = json.load(open(os.path.join(config_dir,mechanisms_file)))
    
    mechanisms = []
    for sectionlist, channels in mech_definitions.items():
        seclist_loc = ephys.locations.NrnSeclistLocation(
            sectionlist,
            seclist_name=sectionlist)
        for channel in channels:
            mechanisms.append(ephys.mechanisms.NrnMODMechanism(
                name='%s.%s' % (channel, sectionlist),
                mod_path=None,
                suffix=channel,
                locations=[seclist_loc],
                preloaded=True))

    return mechanisms


def define_parameters(config_dir='config',parameters_file='parameters.json'):
    """Define parameters"""

    param_configs = json.load(open(os.path.join(config_dir,parameters_file)))
    parameters = []

    for param_config in param_configs:
        if 'value' in param_config:
            frozen = True
            value = param_config['value']
            bounds = None
        elif 'bounds' in param_config:
            frozen = False
            bounds = param_config['bounds']
            value = None
        else:
            raise Exception(
                'Parameter config has to have bounds or value: %s'
                % param_config)

        if param_config['type'] == 'global':
            parameters.append(
                ephys.parameters.NrnGlobalParameter(
                    name=param_config['param_name'],
                    param_name=param_config['param_name'],
                    frozen=frozen,
                    bounds=bounds,
                    value=value))
        elif param_config['type'] in ['section', 'range']:
            if param_config['dist_type'] == 'uniform':
                scaler = ephys.parameterscalers.NrnSegmentLinearScaler()
            elif param_config['dist_type'] == 'exp':
                scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                    distribution=param_config['dist'])
            seclist_loc = ephys.locations.NrnSeclistLocation(
                param_config['sectionlist'],
                seclist_name=param_config['sectionlist'])

            name = '%s.%s' % (param_config['param_name'],
                              param_config['sectionlist'])

            if param_config['type'] == 'section':
                parameters.append(
                    ephys.parameters.NrnSectionParameter(
                        name=name,
                        param_name=param_config['param_name'],
                        value_scaler=scaler,
                        value=value,
                        frozen=frozen,
                        bounds=bounds,
                        locations=[seclist_loc]))
            elif param_config['type'] == 'range':
                parameters.append(
                    ephys.parameters.NrnRangeParameter(
                        name=name,
                        param_name=param_config['param_name'],
                        value_scaler=scaler,
                        value=value,
                        frozen=frozen,
                        bounds=bounds,
                        locations=[seclist_loc]))
        else:
            raise Exception(
                'Param config type has to be global, section or range: %s' %
                param_config)

    return parameters


def define_morphology(morphology_file, simplify_morphology):
    """Define morphology"""
    extension = morphology_file.split('.')[-1]
    if extension == 'asc':
        simplify_morphology = False
    if simplify_morphology:
        return morphology.SWCFileSimplifiedMorphology(morphology_file, do_replace_axon=True, do_set_nseg=True)    
    return ephys.morphologies.NrnFileMorphology(morphology_file, do_replace_axon=True, do_set_nseg=True)


def create_cell(cell_name, morphology_file, simplify_morphology=True, config_dir='config',
                mechanisms_file='mechanisms.json', parameters_file='parameters.json'):
    """Create cell model"""
    return ephys.models.CellModel(cell_name,
                                  morph=define_morphology(morphology_file, simplify_morphology),
                                  mechs=define_mechanisms(config_dir, mechanisms_file),
                                  params=define_parameters(config_dir, parameters_file))
