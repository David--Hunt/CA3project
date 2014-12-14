
from neuron import h
import numpy as np

__all__ = ['psth','generate_poisson_spike_times','make_voltage_recorders','run',
           'distance','pick_section','filter','path_length','simplify_tree',
           'convert_morphology','compute_section_area','SWC_types']

SWC_types = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4}

def psth(spks, binwidth, interval=None):
    if interval is None:
        interval = [min([t[0] for t in spks]), max([t[-1] for t in spks])]
    edges = np.arange(interval[0],interval[1]+binwidth/2,binwidth)
    ntrials = len(spks)
    count = np.zeros((ntrials,len(edges)-1))
    for i,t in enumerate(spks):
        count[i,:] = np.histogram(t,edges)[0]
    if ntrials > 1:
        nu = np.sum(count,0) / (ntrials*binwidth)
    else:
        nu = count / (ntrials*binwidth)
    return nu,edges,count

def generate_poisson_spike_times(rate, tend):
    n = rate*tend
    isi = -np.log(np.random.uniform(size=n))/rate
    spike_times = np.cumsum(isi)
    return spike_times,isi

def make_voltage_recorders(n):
    rec = {}
    for lbl in 't','vsoma','vproximal','vbasal','vdistal':
        rec[lbl] = h.Vector()
    rec['t'].record(h._ref_t)
    rec['vsoma'].record(n.soma[0](0.5)._ref_v)
    rec['vproximal'].record(n.proximal[0](0.5)._ref_v)
    rec['vdistal'].record(n.distal[1](0.5)._ref_v)
    rec['vbasal'].record(n.basal[0](0.5)._ref_v)
    return rec

def run(tend,V0=-65,temperature=36):
    h.load_file('stdrun.hoc')
    h.v_init = V0
    h.celsius = temperature
    h.cvode_active(1)
    h.cvode.maxstep(10)
    h.tstop = tend
    h.run()
        
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

