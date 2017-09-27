#!/usr/bin/env python

import os
import sys
import time
import shutil
import pickle
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
import bluepyopt
import dlopt


def get_responses(evaluator, individuals, filename=None):
    responses = []
    for individual in individuals:
        responses.append(evaluator.run_protocols(protocols=evaluator.fitness_protocols.values(),
                                                 param_values=evaluator.param_dict(individual)))
    if not filename is None:
        with open(filename,'w') as fid:
            pickle.dump(responses,fid)
    return responses


def save_individuals(individuals, filename):
    with open(filename,'w') as fid:
        pickle.dump(individuals,fid)

        
def main():
    parser = arg.ArgumentParser(description='Optimize a multi-compartment neuron model.')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-c', '--config-dir', default='config', type=str, help='folder containing the configuration files')
    parser.add_argument('-s', '--suffix', default=None, type=str, help='suffix for configuration files')
    parser.add_argument('--cell-name', default=None, type=str, help='cell name')
    parser.add_argument('--parameters', default='parameters.json', type=str, help='parameters')
    parser.add_argument('--features', default='features.json', type=str, help='features')
    parser.add_argument('--mechanisms', default='mechanisms.json', type=str, help='mechanisms')
    parser.add_argument('--protocols', default='protocols.json', type=str, help='protocols')
    parser.add_argument('-p', '--population-size', default=100, type=int, help='population size')
    parser.add_argument('-g', '--num-generation', default=100, type=int, help='number of generations')
    args = parser.parse_args(args=sys.argv[1:])

    if args.swc_file.lower() in ('thorny','rs'):
        swc_filename = '/Users/daniele/Postdoc/Research/Janelia/SWCs/FINAL/thorny/DH070813-.Edit.scaled.converted.swc'
    elif args.swc_file.lower() in ('a-thorny','ib'):
        swc_filename = '/Users/daniele/Postdoc/Research/Janelia/SWCs/FINAL/a-thorny/DH070613-1-.Edit.scaled.converted.swc'
    else:
        swc_filename = args.swc_file

    cell_name = args.cell_name
    if cell_name is None:
        cell_name = os.path.basename(swc_filename).split('-.')[0].replace('-','_')
        
    if args.suffix is None:
        suffix = ''
    else:
        suffix = '_' + args.suffix
        
    filenames = {'morphology': swc_filename,
                 'parameters': ''.join(args.parameters.split('.')[:-1]) + suffix + '.' + args.parameters.split('.')[-1],
                 'features': ''.join(args.features.split('.')[:-1]) + suffix + '.' + args.features.split('.')[-1],
                 'mechanisms': ''.join(args.mechanisms.split('.')[:-1]) + suffix + '.' + args.mechanisms.split('.')[-1],
                 'protocols': ''.join(args.protocols.split('.')[:-1]) + suffix + '.' + args.protocols.split('.')[-1]}

    for f in filenames.values():
        if not os.path.isfile(f) and not os.path.isfile(args.config_dir + '/' + f):
            print('%s: %s: no such file.' % (os.path.basename(sys.argv[0]),f))
            sys.exit(1)

    now = time.gmtime(time.time())
    output_folder = '%d%02d%02d%02d%02d%0d_%s' % (now.tm_year,now.tm_mon,now.tm_mday,
                                                  now.tm_hour,now.tm_min,now.tm_sec,
                                                  cell_name)
    os.mkdir(output_folder, 0755)

    for f in filenames.values():
        if os.path.isfile(f):
            shutil.copy(f,output_folder)
        elif os.path.isfile(args.config_dir+'/'+f):
            shutil.copy(args.config_dir+'/'+f, output_folder)
        else:
            print('%s: %s: cannot find file.' % (os.path.basename(sys.argv[0]),f))

    evaluator = dlopt.evaluator.create(cell_name, filenames, config_dir=args.config_dir)
    print(evaluator.cell_model)

    optimisation = bluepyopt.optimisations.DEAPOptimisation(evaluator=evaluator,offspring_size=args.population_size)
    final_pop,hall_of_fame,logbook,history = optimisation.run(max_ngen=args.num_generation)
 
    best_ind = hall_of_fame[0]
    best_ind_dict = evaluator.param_dict(best_ind)

    print('The best values obtained with the optimization are:')
    for k,v in best_ind_dict.iteritems():
        print('%30s = %9.1e' % (k,v))

    #### let's simulate the responses of the final population
    responses = get_responses(evaluator, hall_of_fame, output_folder+'/hall_of_fame_responses.pkl')
    get_responses(evaluator, final_pop, output_folder+'/final_population_responses.pkl')
    save_individuals(hall_of_fame, output_folder+'/hall_of_fame.pkl')
    save_individuals(final_pop, output_folder+'/final_population.pkl')
    with open(output_folder+'/evaluator.pkl','w') as fid:
        pickle.dump(evaluator,fid)
    with open(output_folder+'/logbook.pkl','w') as fid:
        pickle.dump(logbook,fid)
    with open(output_folder+'/history.pkl','w') as fid:
        pickle.dump(history,fid)
    
    #### let's plot the results
    for resp in responses:
        for k,v in resp.iteritems():
            if 'soma' in k:
                plt.plot(v['time'],v['voltage'],linewidth=1)
    for k,v in responses[0].iteritems():
        if 'soma' in k:
            plt.plot(v['time'],v['voltage'],'k',linewidth=2,label='Best individual')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$V_m$ (mV)')
    plt.legend(loc='best')
    plt.show()
    
    #params = {name: [] for name in evaluator.param_names}
    #for individual in final_pop:
    #    for k,v in evaluator.param_dict(individual):
    #        params[k].append(v)
    #cnt = 0.75
    #for k,v in params.iteritems():
    #    plt.plot(cnt+0.5*rnd(size=len(v)),v,'ko')
    #    cnt += 1
    #plt.show()

if __name__ == '__main__':
    main()
