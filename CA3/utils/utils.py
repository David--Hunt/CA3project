
import sys
from neuron import h
import numpy as np
import time

__all__ = ['findpeaks', 'extractAPPeak', 'extractAPThreshold', 'extractAPHalfWidth',
           'extractAPAHP', 'extractAPEnd', 'extractAPADP',
           'run_current_steps_protocol','psth','generate_poisson_spike_times','make_voltage_recorders','run',
           'distance','pick_section','filter','path_length','simplify_tree',
           'convert_morphology','compute_section_area','SWC_types','timestamp']

SWC_types = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4}

timestamp = lambda: time.strftime('%b %d, %H:%M:%S', time.localtime(time.time()))

def findpeaks(x, min_peak_height=None, min_peak_distance=None):
    locs = np.intersect1d(np.where(x[1:] > x[:-1])[0]+1,np.where(x[:-1] > x[1:])[0])
    if not min_peak_height is None:
        locs = locs[x[locs] > min_peak_height]
    if not min_peak_distance is None:
        i = 1
        while i < len(locs):
            if (locs[i] - locs[i-1]) < min_peak_distance:
                locs = np.delete(locs, i)
            else:
                i += 1
    return x[locs],locs

def adjust_threshold(V, threshold):
    if threshold is None:
        threshold = np.max(V, axis=1) - 40
        threshold[threshold < -40] = 0
    elif np.isscalar(threshold):
        threshold += np.zeros(V.shape[0])
    return threshold

def extractAPPeak(T, V, threshold=None, min_distance=5):
    if len(V.shape) == 1:
        V = np.array([V])
    nexp = V.shape[0]
    threshold = adjust_threshold(V, threshold)
    mpd = np.round(min_distance/np.diff(T[:2]))
    tpeak = [np.array([]) for i in range(nexp)]
    vpeak = [np.array([]) for i in range(nexp)]
    for i in range(nexp):
        if np.max(V[i,:]) > threshold[i]:
            pks,locs = findpeaks(V[i,:], min_peak_height=threshold[i], min_peak_distance=mpd)
            tpeak[i] = T[locs]
            vpeak[i] = pks
    return tpeak,vpeak

def extractAPThreshold(T, V, threshold=None, tpeak=None, model=True):
    if len(V.shape) == 1:
        V = np.array([V])
    nexp = V.shape[0]
    threshold = adjust_threshold(V, threshold)
    if tpeak is None:
        tpeak,_ = extractAPPeak(T,V,threshold)
    dt = np.diff(T[:2])
    nspks = map(len, tpeak)
    tthresh = [np.nan + np.zeros(n) for n in nspks]
    vthresh = [np.nan + np.zeros(n) for n in nspks]
    for i in range(nexp):
        for j in range(nspks[i]):
            idx = np.where((T > tpeak[i][j]-3) & (T <= tpeak[i][j]))
            t = T[idx]
            v = np.squeeze(V[i,idx])
            dvdt = (v[2:] - v[:-2]) / (2*dt)
            if model:
                th = np.min([25,np.max(dvdt)/2])
                k = np.where(dvdt > th)[0][0]
                tthresh[i][j] = t[k]
                vthresh[i][j] = v[k]
            else:
                baseline = 0.2 * np.max(dvdt)
                d2vdt2 = (dvdt[2:] - dvdt[:-2]) / (2*dt)
                d3vdt3 = (d2vdt2[2:] - d2vdt2[:-2]) / (2*dt)
                pks,locs = findpeaks(d3vdt3)
                for k in np.flipud(locs):
                    if v[k+3] <= threshold[i] and dvdt[k+1] <= baseline:
                        break
                tthresh[i][j] = t[k+3]
                vthresh[i][j] = v[k+3]
    return tthresh,vthresh

def extractAPHalfWidth(T, V, threshold=None, tpeak=None, Vpeak=None, tthresh=None, Vthresh=None, interp=True):
    if len(V.shape) == 1:
        V = np.array([V])
    nexp = V.shape[0]
    threshold = adjust_threshold(V, threshold)
    if tpeak is None or Vpeak is None:
        tpeak,Vpeak = extractAPPeak(T,V,threshold)
    if tthresh is None or Vthresh is None:
        tthresh,Vthresh = extractAPThreshold(T,V,threshold,tpeak)
    nspks = map(len,tpeak)
    Vhalf = [(Vp+Vth)/2 for Vp,Vth in zip(Vpeak,Vthresh)]
    interval = [np.zeros((2,n)) for n in nspks]
    for i in range(nexp):
        for j in range(nspks[i]):
            idx, = np.where((T >= tthresh[i][j]) & (T<=tpeak[i][j]))
            below, = np.where(V[i,idx] < Vhalf[i][j])
            if interp:
                interval[i][0,j] = np.polyval(np.polyfit(V[i,idx[below[-1]:below[-1]+2]],T[idx[below[-1]:below[-1]+2]],1),Vhalf[i][j])
            else:
                Vhalf[i][j] = V[i,idx[below[-1]]]
                interval[i][0,j] = T[idx[below[-1]]]
            idx, = np.where((T >= tpeak[i][j]) & (T<=tpeak[i][j]+2))
            above, = np.where(V[i,idx] > Vhalf[i][j])
            if interp:
                try:
                    interval[i][1,j] = np.polyval(np.polyfit(V[i,idx[above[-1]:above[-1]+2]],T[idx[above[-1]:above[-1]+2]],1),Vhalf[i][j])
                except:
                    interval[i][1,j] = T[idx[above[-1]]]
            else:
                interval[i][1,j] = T[idx[above[-1]]]
    width = [np.squeeze(np.diff(x,n=1,axis=0)) for x in interval]
    return Vhalf,width,interval

def extractAPAHP(T, V, max_ahp_dur=5, threshold=None, tpeak=None, tthresh=None, tadp=None):
    if len(V.shape) == 1:
        V = np.array([V])
    if tpeak is None:
        tpeak,_ = extractAPPeak(T, V, threshold)
    if tthresh is None:
        tthresh,_ = extractAPThreshold(T, V, threshold, tpeak)
    nexp = V.shape[0]
    nspks = map(len,tpeak)
    tahp = [np.nan + np.zeros(n) for n in nspks]
    Vahp = [np.nan + np.zeros(n) for n in nspks]
    for i in range(nexp):
        for j in range(nspks[i]-1):
            stop = np.min([tthresh[i][j+1]-tpeak[i][j],max_ahp_dur])
            if not tadp is None:
                stop = np.min([stop,tadp[i][j]-tpeak[i][j]])
            idx, = np.where((T >= tpeak[i][j]) & (T <= tpeak[i][j]+stop))
            k = np.argmin(V[i,idx])
            tahp[i][j] = T[idx[k]]
            Vahp[i][j] = V[i,idx[k]]
        if tadp is None:
            idx, = np.where((T >= tpeak[i][-1]) & (T <= tpeak[i][-1]+max_ahp_dur))
        else:
            idx, = np.where((T >= tpeak[i][-1]) & (T <= tadp[i][-1]))
        k = np.argmin(V[i,idx])
        tahp[i][-1] = T[idx[k]]
        Vahp[i][-1] = V[i,idx[k]]
    return tahp,Vahp

def extractAPEnd(T, V, threshold=None, tpeak=None, tthresh=None, Vthresh=None, tahp=None):
    from copy import deepcopy
    if len(V.shape) == 1:
        V = np.array([V])
    if tpeak is None:
        tpeak,_ = extractAPPeak(T, V, threshold)
    if tthresh is None or Vthresh is None:
        tthresh,Vthresh = extractAPThreshold(T, V, threshold, tpeak)
    if tahp is None:
        tahp,_ = extractAPAHP(T, V, threshold=threshold, tthresh=tthresh, tpeak=tpeak)
    nexp = V.shape[0]
    nspks = map(len,tpeak)
    dt = np.diff(T[:2])
    tend = [np.nan + np.zeros(n) for n in nspks]
    Vend = deepcopy(Vthresh)
    for i in range(nexp):
        for j in range(nspks[i]):
            if j < nspks[i]-1 and Vend[i][j] < Vthresh[i][j+1]:
                Vend[i][j] = Vthresh[i][j+1]
            idx, = np.where((T>tpeak[i][j]+0.5) & (T<tahp[i][j]))
            k = np.where(V[i,idx] <= Vend[i][j])[0][0]
            tend[i][j] = T[idx[k]]
            Vend[i][j] = V[i,idx[k]]
    return tend,Vend

def extractAPADP(T, V, max_adp_dur=15, threshold=None, tthresh=None, tahp=None):
    if len(V.shape) == 1:
        V = np.array([V])
    if tthresh is None:
        tthresh,_ = extractAPThreshold(T, V, threshold)
    if tahp is None:
        tahp,_ = extractAPAHP(T, V, threshold=threshold, tthresh=tthresh)
    nexp = V.shape[0]
    nspks = map(len,tthresh)
    dt = np.diff(T[:2])
    tadp = [np.nan + np.zeros(n) for n in nspks]
    Vadp = [np.nan + np.zeros(n) for n in nspks]
    for i in range(nexp):
        for j in range(nspks[i]-1):
            stop = np.min([tthresh[i][j+1]-tahp[i][j],max_adp_dur])
            idx, = np.where((T>tahp[i][j]) & (T<tahp[i][j]+stop))
            k = np.argmax(V[i,idx])
            tadp[i][j] = T[idx[k]]
            Vadp[i][j] = V[i,idx[k]]
        idx, = np.where((T>tahp[i][-1]) & (T<tahp[i][-1]+max_adp_dur))
        k = np.argmax(V[i,idx])
        tadp[i][-1] = T[idx[k]]
        Vadp[i][-1] = V[i,idx[k]]
    return tadp,Vadp

def run_current_steps_protocol(cell, amplitudes, ttran, tstep, tafter=500., temperature=36., max_step=10., V0=-70.):
    h.load_file('stdrun.hoc')

    print(timestamp() + '>> Inserting the stimulus...')
    stim = h.IClamp(cell.soma[0](0.5))
    stim.dur = tstep
    stim.delay = ttran

    print(timestamp() + '>> Setting up the recorders...')
    rec = make_voltage_recorders(cell)
    rec['spikes'] = h.Vector()
    apc = h.APCount(cell.soma[0](0.5))
    apc.record(rec['spikes'])

    h.celsius = temperature
    h.tstop = stim.delay + stim.dur + tafter
    h.cvode_active(1)
    h.cvode.atol(1e-6)
    h.cvode.rtol(1e-6)
    h.cvode.maxstep(max_step)

    t = []
    V = []
    spike_times = []

    for i,amp in enumerate(amplitudes):    
        sys.stdout.write('\r' + timestamp() + '>> Trial [%02d/%02d] ' % (i+1,len(amplitudes)))
        sys.stdout.flush()
        stim.amp = amp
        apc.n = 0
        rec['t'].resize(0)
        rec['vsoma'].resize(0)
        rec['spikes'].resize(0)
        h.t = 0
        h.v_init = V0
        h.run()
        t.append(np.array(rec['t']))
        V.append(np.array(rec['vsoma']))
        spike_times.append(np.array(rec['spikes']))
    sys.stdout.write('\n')

    return np.array(t), np.array(V), np.array(spike_times)

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
    try:
        rec['vproximal'].record(n.proximal[0](0.5)._ref_v)
    except:
        rec.pop('vproximal')
    try:
        rec['vdistal'].record(n.distal[0](0.5)._ref_v)
    except:
        rec.pop('vdistal')
    try:
        rec['vbasal'].record(n.basal[0](0.5)._ref_v)
    except:
        rec.pop('vbasal')
    if n.has_axon:
        for i,lbl in enumerate(['vhillock','vais','vaxon']):
            rec[lbl] = h.Vector()
            if lbl == 'vaxon':
                rec[lbl].record(n.axon[i](1)._ref_v)
            else:
                rec[lbl].record(n.axon[i](0.5)._ref_v)
    return rec

def run(tend,V0=-65,temperature=36):
    h.load_file('stdrun.hoc')
    h.v_init = V0
    h.celsius = temperature
    h.cvode_active(1)
    h.cvode.maxstep(10)
    h.cvode.rtol(1e-6)
    h.cvode.atol(1e-6)
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

