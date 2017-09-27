#!/usr/bin/env python

import os
import sys
import argparse as arg
import numpy as np
from numpy.linalg import svd
import btmorph

# the name of this script
progname = os.path.basename(sys.argv[0])
# the standard SWC types
SWC_types = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4, 'soma_contour': 16}

############################################################
###                       CONVERT                        ###
############################################################

def make_node(entry,parent):
    node = btmorph.SNode2(entry[0])
    p3d = btmorph.P3D2(np.array(entry[2:5]),entry[5],entry[1])
    node.set_content({'p3d': p3d})
    node.set_parent(parent)
    return node


def convert():
    parser = arg.ArgumentParser(description='Convert a morphology to the 3-point soma convention usable by NEURON',
                                prog=progname+' convert')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of existing output file')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    args = parser.parse_args(args=sys.argv[2:])
    
    f_in = args.swc_file
    if args.output is None:
        f_out = f_in.split('.swc')[0] + '.converted.swc'
    else:
        f_out = args.output

    if not os.path.isfile(f_in):
        print('%s: %s: no such file.' % (progname,f_in))
        sys.exit(0)

    if os.path.isfile(f_out) and not args.force:
        print('%s exists: use the -f option to force overwrite.' % f_out)
        sys.exit(0)

    entry_to_string = lambda e: '%d @ (%.3f,%.3f,%.3f), R=%.3f, type=%d, parent=%d' % \
                      (e[0],e[2],e[3],e[4],e[5],e[1],e[6])
    
    morpho_in = np.loadtxt(f_in)

    print('Fixing the morphology.')
    # we first change the type of those points that are of type soma but have parent
    # and children of a different type
    while True:
        done = True
        soma_idx, = np.where(morpho_in[:,1] == SWC_types['soma'])
        soma_ids = morpho_in[soma_idx,0]
        for entry in morpho_in[soma_idx,:]:
            if not entry[-1] in soma_ids and entry[-1] != -1:
                done = False
                parent_idx, = np.where(morpho_in[:,0] == entry[-1])
                print(entry_to_string(entry) + ' >> parent not in soma (type = %d).' % \
                      morpho_in[parent_idx,1])
                children_idx, = np.where(morpho_in[:,-1] == entry[0])
                print('%d children present' % len(children_idx))
                children_types = morpho_in[children_idx,1]
                if not np.any(children_types == SWC_types['soma']):
                    print('None of the children are of type soma: convert entry %d to type %d.' % \
                          (entry[0],morpho_in[parent_idx,1]))
                    morpho_in[morpho_in[:,0] == entry[0],1] = morpho_in[parent_idx,1]
                else:
                    print('At least one child of type soma: do not know what to do.')
                    import pdb; pdb.set_trace()
        if done:
            break

    # center and rotate the morphology
    print('Centering and rotating the morphology.')
    idx, = np.where(morpho_in[:,1] != SWC_types['axon'])
    xyz = morpho_in[:,2:5]
    xyz_centered = xyz - np.mean(xyz[idx,:],axis=0)
    U,S,V = svd(xyz_centered[idx,:])
    xyz_rot = np.dot(xyz_centered,np.transpose(V))
    xyz_rot = xyz_rot[:,(1,0,2)]
    morpho_in[:,2:5] = xyz_rot

    # convert to three-point soma representation
    print('Converting to three-point soma.')
    soma_idx, = np.where(morpho_in[:,1] == SWC_types['soma'])
    if soma_idx.size == 0:
        soma_contour = True
        soma_idx, = np.where(morpho_in[:,1] == SWC_types['soma_contour'])
        soma_type = SWC_types['soma_contour']
        print('The contour of the soma contains %d points.' % soma_idx.size)
    else:
        soma_contour = False
        soma_type = SWC_types['soma']
    soma_ids = morpho_in[soma_idx,0]
    center = np.mean(morpho_in[soma_idx,2:5],axis=0)
    print('The center of the soma is at (%.3f,%.3f,%.3f).' % (center[0],center[1],center[2]))
    morpho_in[:,2:5] -= center
    radius = np.mean(np.sqrt(np.sum(morpho_in[soma_idx,2:5]**2,axis=1)))

    # three-point soma representation
    morpho_out = [[1, SWC_types['soma'], 0, 0, 0, radius, -1],
                  [2, SWC_types['soma'], 0, -radius, 0, radius, 1],
                  [3, SWC_types['soma'], 0, radius, 0, radius, 1]]

    for entry in morpho_in:
        if entry[1] != soma_type:
            # do not add entries that are part of the soma in the original morphology
            if entry[-1] in soma_ids or (soma_contour and entry[-1] == -1):
                if soma_contour and entry[-1] == -1:
                    print(entry_to_string(entry))
                # if an entry's parent belonged to the soma, attach it to the root
                entry[-1] = 1
            morpho_out.append(entry)

    # convert to a numpy array for easier handling
    morpho_out = np.array(morpho_out)

    # create the empty tree
    tree = btmorph.STree2()

    # a dictionary of nodes indexed by their id
    nodes = {-1: None}

    # build the tree
    print('Building the tree to sort the entries.')
    for entry in morpho_out:
        parent = nodes[entry[-1]]
        node = make_node(entry,parent)
        nodes[node.index] = node
        if parent is None:
            tree.set_root(node)
        else:
            parent.add_child(node)
    
    # save the tree to file
    print('Saving the tree to file %s.' % f_out)
    fid = open(f_out,'w')
    for index,node in enumerate(tree):
        node.index = index+1
        xyz = node.content['p3d'].xyz
        if node.parent is None:
            index = -1
        else:
            index = node.parent.index
        fid.write('%d %d %g %g %g %g %d\n' % (node.index,node.content['p3d'].type,
                                              xyz[0], xyz[1], xyz[2],
                                              node.content['p3d'].radius, index))
    fid.close()

############################################################
###                        BUILD                         ###
############################################################

def create_empty_template(template_name,seclist_names=None,secarray_names=None):
    '''create an hoc template named template_name for an empty cell'''
    
    objref_str = 'objref this, CellRef'
    newseclist_str = ''
    
    if seclist_names:
        for seclist_name in seclist_names:
            objref_str += ', %s' % seclist_name
            newseclist_str += \
                              '             %s = new SectionList()\n' % seclist_name

    create_str = ''
    if secarray_names:
        create_str = 'create '
        create_str += ', '.join(
            '%s[1]' % secarray_name
            for secarray_name in secarray_names)
        create_str += '\n'

    template = '''\
    begintemplate %(template_name)s
    %(objref_str)s
    proc init() {\n%(newseclist_str)s
    forall delete_section()
    CellRef = this
    }

    gid = 0

    proc destroy() {localobj nil
    CellRef = nil
    }

    %(create_str)s
    endtemplate %(template_name)s
    ''' % dict(template_name=template_name, objref_str=objref_str,
               newseclist_str=newseclist_str,
               create_str=create_str)

    return template


def build():
    parser = arg.ArgumentParser(description='Use a converted morphology to build a cell in NEURON',
                                prog=progname+' build')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-v','--verbose', action='store_true', help='Be verbose')
    args = parser.parse_args(args=sys.argv[2:])

    if not os.path.isfile(args.swc_file):
        print('%s: %s: no such file.' % (progname,args.swc_file))
        sys.exit(1)

    output_file = args.swc_file.split('.swc')[0] + '.log'

    from neuron import h
    
    cell_name = 'CA3_cell'
    seclist_names = ['all', 'somatic', 'basal', 'apical', 'axonal', 'myelinated']
    secarray_names = ['soma', 'dend', 'apic', 'axon', 'myelin']
    template = create_empty_template(cell_name,seclist_names,secarray_names)
    h(template)
    template_function = getattr(h, cell_name)
    cell = template_function()

    h.load_file('stdlib.hoc')
    h.load_file('stdrun.hoc')
    h.load_file('import3d.hoc')

    import3d = h.Import3d_SWC_read()
    import3d.input(args.swc_file)
    gui = h.Import3d_GUI(import3d, 0)
    gui.instantiate(cell)

    regions = ('soma','axon','basal','apical')
    nsections = {r: 0 for r in regions}
    npoints = {r: 0 for r in regions}
    lengths = {r: 0 for r in regions}
    areas = {r: 0 for r in regions}
    for reg in regions:
        region = getattr(cell,reg)
        for sec in region:
            nsections[reg] += 1
            npoints[reg] += h.n3d(sec=sec)
            area = 0
            for seg in sec:
                area += h.area(seg.x,sec)
            if args.verbose:
                print(sec.name())
                for i in range(int(h.n3d(sec=sec))):
                    pt= np.array([h.x3d(i,sec=sec),h.y3d(i,sec=sec),h.z3d(i,sec=sec)])
                    print('   [%d] (%.2f,%.2f,%.2f)' % (i+1,pt[0],pt[1],pt[2]))
                print('      L = %.2f um.' % sec.L)
                print('   area = %.2f um2.' % area)
            lengths[reg] += sec.L
            areas[reg] += area

    print('')
    for reg in regions:
        print('-----------------------------------------')
        print('[%s]' % reg.upper())
        print('        number of sections: %d' % nsections[reg])
        print('          number of points: %d' % npoints[reg])
        print('    total dendritic length: %.0f um' % lengths[reg])
        print('        total surface area: %.0f um2' % areas[reg])

    print('=========================================\n')
    print('Total number of sections: %d.' % np.sum([v for v in nsections.values()]))
    print('Total number of points: %d.' % np.sum([v for v in npoints.values()]))
    print('Total dendritic length: %.0f um' % np.sum([v for v in lengths.values()]))
    print('Total surface area: %.0f um2' % np.sum([v for v in areas.values()]))


############################################################
###                       SIMPLIFY                       ###
############################################################


def copy_node(node_in, index=None):
    if index == None:
        index = node_in.index
    node_out = btmorph.SNode2(index)
    p3d_in = node_in.content['p3d']
    p3d_out = btmorph.P3D2(p3d_in.xyz, p3d_in.radius, p3d_in.type)
    node_out.set_content({'p3d': p3d_out})
    return node_out


def parse_branch(node, parent, max_branch_length=np.inf):
    branch = [node]
    if node.content['p3d'].type != SWC_types['soma']:
        branch_length = 0
        while len(node.children) == 1 and branch_length < max_branch_length:
            node = node.children[0]
            branch.append(node)
            branch_length += np.sqrt(np.sum((branch[-2].content['p3d'].xyz-branch[-1].content['p3d'].xyz)**2))
    else:
        xyz = branch[-1].content['p3d'].xyz
        print('Node (%f,%f,%f) belongs to soma.' % (xyz[0],xyz[1],xyz[2]))
    end_point = copy_node(branch[-1])
    end_point.set_parent(parent)
    parent.add_child(end_point)
    for child in node.children:
        parse_branch(child, end_point, max_branch_length)


def plot_tree(tree,marker='-',color=None):
    cmap = {idx: col for idx,col in zip(SWC_types.values(),'krbg')}
    for node in tree:
        if not node.parent is None:
            x = [node.content['p3d'].xyz[0],node.parent.content['p3d'].xyz[0]]
            y = [node.content['p3d'].xyz[1],node.parent.content['p3d'].xyz[1]]
            lw = node.content['p3d'].radius
            if color is None:
                c = cmap[node.content['p3d'].type]
            else:
                c = color
            plt.plot(x,y,c+marker,linewidth=lw)


def simplify():
    parser = arg.ArgumentParser(description='Simplify a converted morphology stored in an SWC file.',
                                prog=progname+' simplify')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-l', '--max-length', default=40, type=int, help='maximum path length for section')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    args = parser.parse_args(args=sys.argv[2:])

    if not os.path.isfile(args.swc_file):
        print('%s: no such file.' % args.swc_file)
        sys.exit(1)

    if args.output is None:
        output_file = args.swc_file.split('.swc')[0] + '.simplified.%d_um.swc' % args.max_length
        
    # the original tree
    full_tree = btmorph.STree2()
    full_tree.read_SWC_tree_from_file(args.swc_file,types=range(10))
    print('There are %d nodes in the full representation of the morphology.' % len(full_tree.get_nodes()))

    # the simplified tree
    tree = btmorph.STree2()

    # the two trees have the same root
    tree.set_root(copy_node(full_tree.root))

    # start the recursion with the children of the root
    for node in full_tree.root.children:
        parse_branch(node,tree.root,args.max_length)

    print('There are %d nodes in the simplified representation of the morphology.' % len(tree.get_nodes()))

    fid = open(output_file,'w')
    for index,node in enumerate(tree):
        node.index = index+1
        xyz = node.content['p3d'].xyz
        if node.parent is None:
            index = -1
        else:
            index = node.parent.index
        fid.write('%d %d %f %f %f %f %d\n' % (node.index,node.content['p3d'].type,
                                              xyz[0], xyz[1], xyz[2],
                                              node.content['p3d'].radius, index))
    fid.close()
    #tree.write_SWC_tree_to_file(output_file)



############################################################
###                         PLOT                         ###
############################################################


def plot():
    parser = arg.ArgumentParser(description='Plot a converted morphology',
                                prog=progname+' plot')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    args = parser.parse_args(args=sys.argv[2:])
    
    f_in = args.swc_file
    if args.output is None:
        f_out = f_in.split('.swc')[0] + '.pdf'
    else:
        f_out = args.output

    if not os.path.isfile(f_in):
        print('%s: %s: no such file.' % (progname,f_in))
        sys.exit(0)

    import neurom.viewer
    import matplotlib.pyplot as plt
    neurom.viewer.draw(neurom.load_neuron(f_in))
    plt.show()



############################################################
###                         HELP                         ###
############################################################


def help():
    if len(sys.argv) > 2 and sys.argv[2] in commands:
        cmd = sys.argv[2]
        sys.argv = [sys.argv[0], cmd, '-h']
        commands[cmd]()
    else:
        print('Usage: %s <command> [<args>]' % progname)
        print('')
        print('Available commands are:')
        print('   convert        Convert a morphology to the 3-point soma convention usable by NEURON')
        print('   build          Use a converted morphology to build a cell in NEURON and obtain some statistics')
        print('   plot           Plot a converted morphology file')
        print('   simplify       Simplify a converted morphology')
        print('')
        print('Type \'%s help <command>\' for help about a specific command.' % progname)

############################################################
###                         MAIN                         ###
############################################################


# all the commands currently implemented
commands = {'help': help, 'convert': convert, 'build': build, 'plot': plot, 'simplify': simplify}

def main():
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        commands['help']()
        sys.exit(0)
    if not sys.argv[1] in commands:
        print('%s: %s is not a recognized command. See \'%s --help\'.' % (progname,sys.argv[1],progname))
        sys.exit(1)
    commands[sys.argv[1]]()


if __name__ == '__main__':
    main()


