import btmorph
import numpy
import matplotlib.pyplot as plt

#--------analyzing and visualizing morphology--------

swc_tree= btmorph.STree2()
swc_tree.read_SWC_tree_from_file('SWCfiles/DH071013C2-.Edit.scaled.swc')
swc_tree.read_SWC_tree_from_file('SWCfiles/DH071013C2-.Edit.scaled.swc')
stats = btmorph.BTStats(swc_tree)

# get the total length
total_length = stats.total_length()
print 'total_length = %f' % total_length

# get the max degree
max_degree = stats.degree_of_node(swc_tree.get_root())
print 'max_degree = %f' % max_degree

#generate dendrogram
btmorph.plot_dendrogram('SWCfiles/DH071013C2-.Edit.scaled.swc')

#generate 2D_projection
btmorph.plot_2D_SWC('SWCfiles/DH071013C2-.Edit.scaled.swc',show_axis=False,color_scheme='default',depth='Y')

plt.show()
