
import matplotlib.pyplot as p
import numpy as np

__all__ = ['set_rc_defaults','remove_border', 'make_axes']

def set_rc_defaults():
    p.rc('font', family='helvetica', size=11)
    p.rc('lines', linewidth=1, color='k')
    p.rc('axes', linewidth=1, titlesize='medium', labelsize='medium')
    p.rc('xtick', direction='out')
    p.rc('ytick', direction='out')
    p.rc('figure', dpi=300)

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or p.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

def make_axes(r, c, i, offset=[0.1,0.1], spacing=[0.1,0.1], border=[0.05,0.05]):
    """
    A better subplot.
    """
    if np.isscalar(offset):
        offset = [offset,offset]
    if np.isscalar(spacing):
        spacing = [spacing,spacing]
    if np.isscalar(border):
        border = [border,border]
    w = (1 - offset[0] - spacing[0]*(c-1) - border[0])/c
    h = (1 - offset[1] - spacing[1]*(r-1) - border[1])/r
    i = i-1
    x = i%c
    y = r - 1 - int(i/c)
    ax = p.axes([offset[0] + (w+spacing[0])*x, 
                 offset[1] + (h+spacing[1])*y,
                 w, h])
    return ax

