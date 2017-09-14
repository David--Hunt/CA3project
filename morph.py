#!/usr/bin/env python

import os
import sys
import btmorph
import matplotlib.pyplot as plt

#--------analyzing and visualizing morphology--------

def main():
    if len(sys.argv) == 1:
        print('Usage: %s file.swc' % os.path.basename(sys.argv[0]))
        sys.exit(1)
    # get the filename passed as an argument by the user
    swc_file = sys.argv[1]
    tree = btmorph.STree2()
    tree.read_SWC_tree_from_file(swc_file)
    stats = btmorph.BTStats(tree)
    # get the total length
    total_length = stats.total_length()
    print('total_length = %f um.' % total_length)
    # get the max degree
    max_degree = stats.degree_of_node(tree.root)
    print('max_degree = %d.' % max_degree)
    # generate dendrogram
    btmorph.plot_dendrogram(swc_file)
    # generate 2D_projection
    btmorph.plot_2D_SWC(swc_file,show_axis=False,color_scheme='default',depth='Y')
    plt.show()

if __name__ == '__main__':
    main()

