from neuron import h
import numpy as np

__all__ = ['removeSpikes','extractSpikes',
           'connectWithTMGSynapse', 'makeSectionWithMechanisms',
           'makeRS', 'makeLTS',
           'makeFSIN', 'makeL5PC', 'makePC', 'makeWB', 'makeHH',
           'makeIclamp','makeNoisyIclamp',
           'makeGclamp','makeNoisyGclamp',
           'makePeriodicPulse', 'makeAPcount',
           'makeRecorders', 'run', 'continuerun', 'computefIcurve',
           'computeInputResistance','computeCapacitance','optimizeF']

import sys
import pylab as p

def removeSpikes(dt,Vin,threshold,window,in_place=False):
    t = np.arange(Vin.shape[-1])*dt
    spikes = extractSpikes(dt,Vin,threshold)
    if in_place:
        Vout = Vin
    else:
        Vout = Vin.copy()
    wndw = np.arange(-np.round(window[0]/dt),np.round(window[1]/dt))
    ndx = [np.array([]) for i in range(Vin.shape[0])]
    for i,trace in enumerate(Vout):
        for spk in spikes[i]:
            idx = np.round(spk/dt) + wndw
            idx = idx[idx>=0]
            idx = idx[idx<len(trace)]
            ndx[i] = np.append(ndx[i],idx)
        ndx[i] = ndx[i].astype(int)
        Vout[i,ndx[i]] = np.nan
    return Vout,ndx

def extractSpikes(dt,V,threshold):
    spikes = []
    for trace in V:
        x = (trace-threshold)[:-1] 
        y = (trace-threshold)[1:]
        idx = np.intersect1d(np.nonzero(x<0)[0],np.nonzero(y>=0)[0]) + 1
        spikes.append(idx*dt)
    return spikes

def connectWithTMGSynapse(pre, post, E, taus, U, delay=1, weight=0.01):
    pre.sec.push()
    syn = h.tmgsyn(post)
    syn.e = E
    syn.tau_1 = taus['1']
    syn.tau_rec = taus['rec']
    syn.tau_facil = taus['facil']
    syn.U = U
    conn = h.NetCon(pre._ref_v, syn)
    conn.weight[0] = weight
    conn.delay = delay
    conn.threshold = 0
    h.pop_section()
    return syn,conn

def makeSectionWithMechanisms(mechanisms, L=20, diam=20):
    sec = h.Section()
    sec.L = L
    sec.diam = diam
    for m in mechanisms:
        sec.insert(m)
    return sec

def makeLTS(L=96, diam=96, type='deterministic'):
    t = type.lower()
    if t == 'deterministic' or t == 'det' or t == 'd':
        t = 'd'
    elif t == 'stochastic' or t == 'stoch' or t == 's':
        t = 's'
    else:
        print('Unknown type [%s].\n' % type)
        return None
    # an LTS neuron is a RS with some slight changes in parameters
    # and an additional low-threshold calcium current, IT
    sec = makeRS(L, diam, type)
    sec(0.5).pas.e = -60
    sec(0.5).pas.g = 1e-5
    if t == 's':
        sec(0.5).im_cn.gkbar = 3e-5
    sec.insert('cad')
    sec(0.5).cad.depth = 1
    sec(0.5).cad.taur = 5
    sec(0.5).cad.cainf = 2.4e-4
    sec(0.5).cad.kt = 0
    sec.insert('it')
    sec.cai = 2.4e-4
    sec.cao = 2
    sec.eca = 120
    sec.gcabar_it = 4e-4

    ### modifications - start
    #sec.e_pas = -50
    #sec.g_pas = 1e-5
    #sec.gnabar_hh2 = 0.05
    #sec.gkbar_hh2 = 0.005
    #sec.gkbar_im = 3e-5
    #sec.gcabar_it = 4e-4
    ### modifications - end

    return sec

def makeRS(L=96, diam=96, type='deterministic'):
    t = type.lower()
    if t == 'deterministic' or t == 'det' or t == 'd':
        t = 'd'
    elif t == 'stochastic' or t == 'stoch' or t == 's':
        t = 's'
    else:
        print('Unknown type [%s].\n' % type)
        return None
    sec = h.Section()
    sec.Ra = 100
    sec.nseg = 1
    sec.L = L
    sec.diam = diam
    sec.cm = 1
    sec.insert('pas')
    sec(0.5).pas.e = -70
    sec(0.5).pas.g = 0.0001
    if t == 'd':
        sec.insert('hh2')
        hh2_mech = sec(0.5).hh2
    elif t == 's':
        sec.insert('hh2_cn')
        hh2_mech = sec(0.5).hh2_cn
    sec.ek = -100
    sec.ena = 50
    hh2_mech.vtraub = -55
    hh2_mech.gnabar = 0.05
    hh2_mech.gkbar = 0.005
    if t == 'd':
        sec.insert('im')
        h.taumax_im = 1000
        im_mech = sec(0.5).im
    elif t == 's':
        sec.insert('im_cn')
        h.taumax_im_cn = 1000
        im_mech = sec(0.5).im_cn
    im_mech.gkbar = 7e-5
    return sec

def makeFSIN(L=20, diam=20, type='deterministic'):
    type = type.lower()
    sec = h.Section()
    if type == 'deterministic' or type == 'det' or type == 'd':
        sec.insert('Inafs')
        sec.insert('Ikdrfs')
        sec.insert('Id')
    else:
        raise Exception('Unknown type: %s.' % type)
    sec.insert('pas')
    sec.L = L
    sec.diam = diam
    sec.ena = 50
    sec.ek = -90
    sec(0.5).pas.g = 0.25e-3
    sec(0.5).pas.e = -70
    return sec

def makeL5PC(L=20, diam=20, type='deterministic'):
    type = type.lower()
    mechs = ['Ina','Ika','Inap','Ikdr','Ika','Ikslow']
    if type == 'deterministic' or type == 'det' or type == 'd':
        pass
    elif type == 'stochastic' or type == 'stoch' or type == 's':
        for k in range(len(mechs)):
            mechs[k] = 'cn' + mechs[k]
    else:
        raise Exception('Unknown type: %s.' % type)
    mechs.append('pas')
    sec = makeSectionWithMechanisms(mechs, L, diam)
    sec.ena = 55
    sec.ek = -90
    sec(0.5).pas.g = 0.02e-3
    sec(0.5).pas.e = -70
    return sec

def makePC(L=20, diam=20, type='deterministic'):
    sec = h.Section()
    if type == 'deterministic':
        sec.insert('naRsg')
        sec.insert('kpkj')
    else:
        raise Exception('PC: no such mechanism.')        
    sec.insert('kpkj2')
    sec.insert('kpkjslow')
    sec.insert('bkpkj')
    sec.insert('cadiff')
    sec.insert('cap')
    sec.insert('lkpkj')
    sec.insert('hpkj')
    sec.L = L
    sec.diam = diam
    sec.ena = 60
    sec.ek = -88
    return sec

def makeWB(L=30, diam=30, type='deterministic'):
    sec = h.Section()
    sec.insert('pas')
    sec.L = L
    sec.diam = diam
    sec(0.5).pas.g = 1e-4
    sec(0.5).pas.e = -67
    sec.cm = 1.0
    sec.Ra = 35.4
    if type == 'deterministic':
        sec.insert('WB')
    elif type == 'stochastic':
        sec.insert('WBcn')
        sec(0.5).WBcn.seed = np.random.poisson(1000)
    else:
        raise Exception('WB: no such mechanism.')
    sec.ena = 55
    sec.ek = -90
    return sec

def makeHH(L=30, diam=30, type='deterministic'):
    sec = h.Section()
    if type == 'deterministic':
        sec.insert('hh')
    elif type == 'stochastic':
        sec.insert('HHcn')
        sec(0.5).HHcn.seed = np.random.poisson(1000)
    else:
        raise Exception('HH: no such mechanism.')
    return sec

def OU(mu, sigma, tau, dur, dt, seed):
    np.random.seed(seed)
    nsteps = int(np.ceil((dur)/dt)) + 1
    coeff = np.exp(-dt/tau)
    x = (1-np.exp(-dt/tau))*mu + sigma * np.sqrt(2*dt/tau) * np.random.normal(size=nsteps)
    x[0] = mu
    for i in range(1,nsteps):
        x[i] = x[i] + coeff*x[i-1]
    return x

def makeIclamp(segment, dur, amp, delay=0):
    stim = h.IClamp(segment)
    stim.delay = delay
    stim.dur = dur
    stim.amp = amp
    return stim

def makeNoisyIclamp(segment, dur, dt, mu, sigma, tau, delay=0, seed=5061983):
    I = OU(mu,sigma,tau,dur+delay,dt,seed)
    vec = h.Vector(I)
    stim = h.IClamp(segment)
    stim.dur = dur
    stim.delay = delay
    vec.play(stim._ref_amp,dt)
    return stim,vec

def makeGclamp(segment, dur, amp, e, delay=0):
    stim = h.GClamp(segment)
    stim.delay = delay
    stim.dur = dur
    stim.g = amp
    return stim

def makeNoisyGclamp(segment, dur, dt, mu, sigma, tau, e, delay=0, seed=5061983):
    G = OU(mu,sigma,tau,dur+delay,dt,seed)
    G[G<0] = 0
    vec = h.Vector(G)
    stim = h.GClamp(segment)
    stim.dur = dur
    stim.delay = delay
    stim.e = e
    vec.play(stim._ref_g,dt)
    return stim,vec

def makePeriodicPulse(segment, duration=3, amplitude=0, period=20, number=15, delay=0):
    stim = h.Ipulse2(segment)
    stim.delay = delay
    stim.dur = duration
    stim.amp = amplitude
    stim.per = period
    stim.num = number
    return stim

def makeAPcount(segment, thresh=-20):
    apc = h.APCount(segment)
    apc.thresh = thresh
    return apc

def makeRecorders(segment, labels, rec=None):
    if rec is None:
        rec = {'t': h.Vector()}
        rec['t'].record(h._ref_t)
    for k,v in labels.items():
        rec[k] = h.Vector()
        rec[k].record(getattr(segment, v))
    return rec

def run(tstop=1000, dt=0, V=-65):
    h.load_file('stdrun.hoc')
    #h.finitialize(V)
    if dt > 0:
        h.dt = dt
    h.tstop = tstop
    h.run()

def continuerun(tstop, dt=0):
    if dt > 0:
        h.dt = dt
    h.continuerun(tstop)

def computefIcurve(segment, Irange, dur, delay, ntrials=1, dt=0.005):
    stim = makeIclamp(segment, dur, 0, delay)
    ap = makeAPcount(segment)
    I = np.arange(Irange[0],Irange[1]+Irange[2]/2,Irange[2])
    f = np.zeros((I.size,ntrials))
    for i,amp in enumerate(I):
        stim.amp = amp
        for j in range(ntrials):
            ap.n = 0
            run(2*delay+dur, dt)
            f[i,j] = ap.n/(dur*1e-3)
    del stim
    del ap
    return I,f

def computeInputResistance(segment, Irange, dur, delay, dt=0.005, plot=False):
    if plot:
        import pylab as p
    stim = makeIclamp(segment, dur, 0, delay)
    rec = makeRecorders(segment, {'v': '_ref_v'})
    ap = makeAPcount(segment)
    I = []
    V = []
    if plot:
        p.figure()
        p.subplot(1,2,1)
    for k,i in enumerate(np.arange(Irange[0],Irange[1],Irange[2])):
        ap.n = 0
        stim.amp = i
        run(2*delay+dur, dt)
        t = np.array(rec['t'])
        v = np.array(rec['v'])
        if ap.n == 0:
            idx = np.intersect1d(np.nonzero(t > delay+0.75*dur)[0], np.nonzero(t < delay+dur)[0])
            I.append(i)
            V.append(np.mean(v[idx]))
        else:
            print('The neuron emitted spikes at I = %g pA' % (stim.amp*1e3))
        if plot:
            p.plot(1e-3*t,v)
    V = np.array(V)*1e-3
    I = np.array(I)*1e-9
    poly = np.polyfit(I,V,1)
    if plot:
        ymin,ymax = p.ylim()
        p.plot([1e-3*(delay+0.75*dur),1e-3*(delay+0.75*dur)],[ymin,ymax],'r--')
        p.plot([1e-3*(delay+dur),1e-3*(delay+dur)],[ymin,ymax],'r--')
        p.xlabel('t (s)')
        p.ylabel('V (mV)')
        p.box(True)
        p.grid(False)
        p.subplot(1,2,2)
        x = np.linspace(I[0],I[-1],100)
        y = np.polyval(poly,x)
        p.plot(1e12*x,1e3*y,'k--')
        p.plot(1e12*I,1e3*V,'bo')
        p.xlabel('I (pA)')
        p.ylabel('V (mV)')
        p.show()
    return poly[0]
    
def computeCapacitance(segment, transient, stimdur, stimamp, trials, R, dt=0.005):
    stim = makeIclamp(segment, stimdur, stimamp, transient)
    rec = makeRecorders(segment, {'v': '_ref_v'})
    tstop = stimdur+2*transient
    n = tstop/dt + 1
    V = np.zeros((trials,n))
    for k in range(trials):
        run(tstop, dt)
        V[k,:] = np.array(rec['v'])
    t = np.array(rec['t'])
    idx = np.nonzero(t > stimdur+transient)
    t = t[idx] - (stimdur+transient)
    Vm = np.mean(V,0)[idx]
    idx = np.nonzero(t > 0.5*transient)
    Voff = np.mean(Vm[idx])
    x = t / 1000
    y = -(Vm-Voff) / 1000
    idx = np.nonzero(y < 0)[0]
    return - np.polyfit(x[0:idx[0]], np.log(y[0:idx[0]]), 1)[0]/R

def optimizeF(segment, F, ftol=0.1, dur=5000, dt=0.025, amp=[0,0.2], delay=200, maxiter=50):
    from sys import stdout
    f = F + 2*ftol
    iter = 0
    stim = makeIclamp(segment, dur, amp[1], delay)
    spks = h.Vector()
    apc = makeAPcount(segment)
    apc.record(spks)
    rec = makeRecorders(segment, {'v': '_ref_v'})
    print('\nStarting frequency optimization: target is F = %.2f.' % F)

    run(dur+2*delay, dt)

    f = float(apc.n)/(dur*1e-3)
    if f < F:
        print('[00] !! Increase maximal current !!')
        raise Exception('Required frequency out of current bounds')
    else:
        print('[00] I = %.4f -> F = %.4f Hz.' % (stim.amp, f))

    while abs(F - f) > ftol and iter < maxiter:
        iter = iter+1
        stim.amp = (amp[0]+amp[1])/2
        stdout.write('[%02d] I = %.4f ' % (iter, stim.amp))
        spks = h.Vector()
        apc.n = 0
        apc.record(spks)
        run(dur+2*delay, dt)
        if len(spks) == 0:
            amp[0] = stim.amp
            stdout.write('no spikes.\n')
            stdout.flush()
            continue
        f = float(apc.n) / (dur*1e-3)
        stdout.write('-> F = %.4f Hz.\n' % f)
        stdout.flush()
        if f > F:
            amp[1] = stim.amp
        else:
            amp[0] = stim.amp
    I = stim.amp
    del apc
    del stim
    del spks
    return f,I

if __name__ == '__main__':
    soma = makeWB(30,30,'stochastic')
    R = computeInputResistance(soma(0.5), [-0.25,0.001,0.05], 2000, 1000, plot=False)
    C = computeCapacitance(soma(0.5), 300, 300, -0.005, 30, R)
    print('A = %.0f um^2 -> R = %.2f MOhm, C = %.4f uF, tau = %g ms' % 
          (np.pi*soma.L*soma.diam, R*1e-6, C*1e6, 1000/(R*C)))
    soma = makeWB(80,80,'stochastic')
    R = computeInputResistance(soma(0.5), [-0.25,0.051,0.05], 2000, 1000, plot=False)
    C = computeCapacitance(soma(0.5), 300, 300, -0.005, 30, R)
    print('A = %.0f um^2 -> R = %.2f MOhm, C = %.4f uF, tau = %g ms' % 
          (np.pi*soma.L*soma.diam, R*1e-6, C*1e6, 1000/(R*C)))
