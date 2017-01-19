# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 18:19:09 2016

@author: Pierre H. Richemond
"""
from __future__ import division, absolute_import #print_function
from scipy import signal
import numpy as np
from numpy import log
import pandas as pd
import matplotlib.pyplot as plt
import tables
from scipy.stats import norm, scoreatpercentile
from collections import OrderedDict

# Adapted from
# https://github.com/buckie/wtmm-python/blob/master/notebooks/old_fractal_demos/multifractal%20decomposition%20with%20uniform%20noise.ipynb


np.random.seed(234897229)

def _sort_tuples(x, y):
    if x[0] < y[0]:
        return -1
    elif x[0] == y[0]:
        if x[1] > y[1]:
            return -1
        elif x[1] < y[1]:
            return 1
        else:
            return 0
    else:
        return 1

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K
    
def possible_direction(hood, proximity, center_row=3, center_col=3):
    nzs = np.where(hood[:center_row] > 0)
    #tuple of (abs-dist, value, (row, col))
    matches = [(abs(row - center_row) + abs(col - center_col), hood[row,col], (row, col)) for row, col in zip(*nzs)]
    matches.sort(key=cmp_to_key(_sort_tuples))
    assert hood[center_row, center_col] != 0, matches
    return matches

def walk_bifurcation(mtx, start_row, start_col, proximity=5, step=-1):
    slope = 0
    center_row, center_col = start_row, start_col
    max_row, max_col = [i - 1 for i in mtx.shape]
    trace_rt = []
    
    while center_row > 0:
            
        #get the prox bounds
        right_bound = center_col + proximity + 1
        left_bound = center_col - proximity
        hood_center_col = proximity
        if right_bound > max_col:
            right_bound = max_col
        elif left_bound < 0:
            # edge case when the hood extends beyond the bounds of the matrix
            # center in the hood is usually proximity, but if the left_bound is in conflict
            # then we have to adjust the center. As left_bound is negative at this point, 
            # it is also the ammount of shift the center needs...
            #  eg:
            #     proximity = 3
            #     center_col = 2
            #     left_bound = -1
            #     hood[-1] = [0, 0, center, 0, 0 ,0] <-- hood[-1] is always hood_center_row
            # thus hood_center_col need to have -1 applied (or just the invalid left_bound)
            hood_center_col = proximity + left_bound
            left_bound = 0
        
        
        lower_bound = center_row - proximity
        if lower_bound < 0:
            # same arguement as above applies
            hood_center_row = proximity + lower_bound
            lower_bound = 0
        else:
            hood_center_row = proximity 
            
        
        # get the neighborhood...
        hood = mtx[lower_bound:center_row+1, left_bound:right_bound]
#         if center_row
#         print_hood(hood)
        
        # find the best choice for the ridge
        try:
            possibles = possible_direction(hood, proximity, center_row=hood_center_row, center_col=hood_center_col)
        except AssertionError as e:
            print(e)
            print("Center (row, col)",  center_row, center_col)
            print("bounds (lower, left, right)",  lower_bound, left_bound, right_bound)
            print("hood (row, col)", hood_center_row, hood_center_col)
            print_hood(hood)
            print(trace_rt)
            raise ValueError("the bifucation walked has lost its tracking")
            

        if not possibles:
            return False, trace_rt
#             print(center_row, center_col, mtx[center_row, center_col])
#             print(trace_rt)
#             print_hood(hood)
#             expand_hood = 5
#             lower_bound = lower_bound - expand_hood if lower_bound - expand_hood > 0 else 0
#             left_bound = left_bound - expand_hood if left_bound - expand_hood > 0 else 0
#             right_bound = right_bound + expand_hood if right_bound + expand_hood < max_col else max_col
#             print_hood(mtx[lower_bound:center_row, left_bound:right_bound])
#             return None
        
        # get the winner
        match = possibles.pop(0)
        
        #recompute the center and continue
        match_hood_row, match_hood_col = match[2]
        
        # TODO: we need to keep track of the movement of the curves
        
        # match_hood_row < proximity always (this moves us up the matrix rows) but is always off by 1
        center_row += match_hood_row - hood_center_row
        # this can be +/- depending on the direction
        center_col += match_hood_col - hood_center_col
#         print(center_row, center_col, mtx[center_row, center_col])
        if center_row >= 0:
            trace_rt.append((center_row, center_col))
        else:
            trace_rt.append((0, center_col))
        
        if center_col == max_col or center_col == 0:
            #If we end up on and edge, this is not a valid bifurcation
            return False, trace_rt
        
    return True, trace_rt

        
    
def print_hood(hood):
    print(hood)
    plt.figure(figsize=(14,10))
    plt.pcolormesh(hood, cmap='Greys')
    plt.colorbar()
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    plt.show()

def skeletor(mtx, proximity=9, smallest_scale=0):
    '''
    Skeleton Construction
    
    proximity: defines region around the matrix
    '''
    # NB: scale <-> row
    # NB: shift <-> col
    max_row, max_col = mtx.shape
    max_row -= 1
    max_col -= 1
    
    # holder for the ridges
    bifurcations = OrderedDict()
    invalids = OrderedDict()
    bi_cnt = 0
    
    for n, row_num in enumerate(range(max_row, smallest_scale, -1)):
        # loop from the max-scale up
        maxs = signal.argrelmax(mtx[row_num])[0]
        #print("loop", n, "row_num", row_num, "maxs", maxs)
        
        if not maxs.any():
            # Nothing here...
            #print "exit", maxs.any()
            continue
        
        for start_pt in maxs:
            continuous, bifurc_path = walk_bifurcation(mtx, row_num, start_pt, proximity=proximity)
#             print row_num, start_pt, "cont", continuous, "bifurc_len", len(bifurc_path)
            if continuous:
                # add the bifurcation to the collector; key == row[0] intercept's column number
                bifurcations[(bi_cnt, bifurc_path[-1][1])] = bifurc_path
                bi_cnt += 1
            elif bifurc_path:
                invalids[bifurc_path[-1]] = bifurc_path
            
            if len(bifurc_path):
                #now zero out all of the entries that were used to walk the bifucation
                rows_b, cols_b = zip(*bifurc_path)
                rows_b = np.array(rows_b)
                cols_b = np.array(cols_b)
#                 for d in range(-del_spread, del_spread):
                mtx[rows_b, cols_b] = 0
    
    return bifurcations

def _create_w_coef_mask(w_coefs, epsilon=0.1, order=1):
    mask = np.zeros_like(w_coefs, dtype=int)
    epsilon = 0.1
    for n, row in enumerate(w_coefs):
        maxs = signal.argrelmax(row, order=order)[0]
        mask[n, maxs] = row[maxs]/epsilon
    
    return mask

def perform_cwt(sig, width=0.5, max_scale=None, wavelet=signal.ricker, epsilon=0.1, order=1, plot=False):
    #Literature suggests that len/4 is the best bet
    if not max_scale:
        max_scale = len(sig)/4
    widths = np.arange(1, max_scale, width)
    
    #normalize the signal to fit in the wavelet
    sig_max = sig.max()
    sig_min = sig.min()
    sig = (sig - (sig_min - 0.01)) / (sig_max - sig_min + 0.02)


    #Run the transform
    #    theres a bug in here somewhere -- somehow I'm getting entries where -1 * log(abs(row[maxs])) < 0
    w_coefs =  abs(-1 * log(abs(signal.cwt(sig, signal.ricker, widths))))
    
    #Create the mask, keeping only the maxima
    mask = _create_w_coef_mask(w_coefs, epsilon=epsilon, order=order)
    
    if plot:
        plt.figure(figsize=(14,10))
        plt.pcolormesh(w_coefs)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.xaxis.tick_top()
        plt.show()

    return mask
    
    
def forgery(Iterations=0, Multifractal=1):
    if Multifractal:
        turns=((0.25,0.5),(0.75, 0.25))
    else:
        turns=((0.4,0.6),(0.6, 0.4))
    first_turn, second_turn = turns
    ys = [0,1]
    ts = [0,1]
    for i in range(0, Iterations + 1):
        
        j=0
        while ts[j] < 1:
            dt = ts[j+1] - ts[j] 
            dy = ys[j+1] - ys[j]
            
            ts.insert(j+1, ts[j] + first_turn[0]*dt)
            ts.insert(j+2, ts[j] + second_turn[0]*dt)
            ys.insert(j+1, ys[j] + first_turn[1]*dy)
            ys.insert(j+2, ys[j] + second_turn[1]*dy)
            
            j += 3
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'axisbg':'#EEEEEE', 'axisbelow':True})
    ax.grid(color='w', linewidth=2, linestyle='solid')
    ax.plot(ts, ys, color='b', alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

def forgery_prices(Iterations=10, Multifractal=1, noise_type=None, noise_level=1.0):
    if Multifractal:
        turns=((0.25,0.5),(0.75, 0.25))
    else:
        turns=((0.4,0.6),(0.6, 0.4))
    first_turn, second_turn = turns
    ys = [0,1]
    ts = [0,1]
    
    if not noise_type:
        for i in range(0, Iterations + 1):

            j=0
            while ts[j] < 1:
                dt = ts[j+1] - ts[j] 
                dy = ys[j+1] - ys[j]

                ts.insert(j+1, ts[j] + first_turn[0]*dt)
                ts.insert(j+2, ts[j] + second_turn[0]*dt)
                ys.insert(j+1, ys[j] + first_turn[1]*dy)
                ys.insert(j+2, ys[j] + second_turn[1]*dy)

                j += 3
    else:
        if noise_type == 'uniform':
            noise = np.random.rand
        elif noise_type == 'normal':
            noise = np.random.randn
            
        for i in range(0, Iterations + 1):

            j=0
            while ts[j] < 1:
                dt = ts[j+1] - ts[j] 
                dy = ys[j+1] - ys[j]
                
                #normalize the noise versus the current dt
                n_a, n_b = (noise(2) * noise_level) * float(dy)

                ts.insert(j+1, ts[j] + first_turn[0]*dt)
                ts.insert(j+2, ts[j] + second_turn[0]*dt)
                ys.insert(j+1, ys[j] + n_a + first_turn[1]*dy)
                ys.insert(j+2, ys[j] + n_b + second_turn[1]*dy)

                j += 3
    
    return np.array(ts), np.array(ys)
    
cartoon_t, cartoon_s = forgery_prices(5, Multifractal=True, noise_type='uniform', noise_level=0.15)
print("cartoon's signal is {} in length".format(len(cartoon_s)))
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'axisbg':'#EEEEEE', 'axisbelow':True})
ax.grid(color='w', linewidth=2, linestyle='solid')
ax.plot(cartoon_t, cartoon_s, color='b', alpha=0.4)
plt.show()

#Run the Direct CWT
sig = cartoon_s
sig_t = cartoon_t

sig2 = sig.copy()
sig_max = sig2.max()
sig_min = sig2.min()
sig2 = (sig2 - (sig_min - 0.01)) / (sig_max - sig_min + 0.02)
plot(sig2)

w_coefs = perform_cwt(sig, width=0.25, max_scale=100, plot=True)

# Some plots of the w_coef matrix
plt.figure(figsize=(14,10))
plt.pcolormesh(w_coefs, cmap='Greys')
plt.colorbar()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
ax.xaxis.tick_top()
plt.show()

plt.figure(figsize=(14,10))
plt.pcolormesh(w_coefs[:100,:100], cmap='Greys')
plt.colorbar()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
ax.xaxis.tick_top()
plt.show()


#Create the bifucration skeleton
bifucations = skeletor(w_coefs, smallest_scale=1)

plt.figure(figsize=(14,10))

for n, (k, v) in enumerate(bifucations.items()):
    rows, cols = zip(*v)
    plt.plot(cols, rows)
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
ax.xaxis.tick_top()
plt.show()
#Number of turning points found at this resolution
print(len(bifucations))


# Plot the results
max_res = 100000
plt.figure(figsize=(24,50))
end_pt = len(sig_t) - 1

lst_color = 'r'
# sig = bwn_m # <-- this was already defined above
ts = sig_t
offset_plot = (max(sig) - min(sig))

def build_extremas(bifurc, max_res):
    exts = set()
    for num, t in bifurc.keys():
        if len(exts) > max_res:
            break
        exts.add(t)

    return sorted(exts)
    
for pl_num, max_res in enumerate([4, 10, 28, 82, 244]):
    extremas = [0] + build_extremas(bifucations, max_res - 2) + [end_pt]
    num_extremas = len(extremas)
    for n in range(num_extremas-1):
        if lst_color == 'r':
            lst_color = 'b'
        elif lst_color == 'b':
            lst_color = 'm'
        elif lst_color == 'm':
            lst_color = 'g'
        else:
            lst_color = 'r'
        
        true_t, true_s = forgery_prices(pl_num, Multifractal=1)
        plt.plot(true_t, true_s - (pl_num*offset_plot), 'y', alpha=0.5)
        plt.plot(ts[extremas[n]:extremas[n+1]+1],
                 sig[extremas[n]:extremas[n+1]+1] - (pl_num*offset_plot),
                 c=lst_color)

plt.plot(ts[[0] + extremas + [end_pt]], sig[[0] + extremas + [end_pt]] + offset_plot, 'y')
plt.plot(ts, sig + (2*offset_plot), 'k')

# extremas = sorted([t for num, t in bifucations.keys() if num < 10])
# plt.plot([0] + extremas + [999], sig[[0] + extremas + [999]] + (2*offset_plot), 'y')
# extremas = sorted([t for num, t in bifucations.keys() if num < 25])
# plt.plot([0] + extremas + [999], sig[[0] + extremas + [999]] + (2*offset_plot), 'g')
# plt.savefig('./example_clustering.png', dpi=5000)
plt.show()
# Top (blue) -> original signal
# Middle (yellow) -> piecewise vector representation
# Bottom (red/blue/green alternating) -> groupings based on the piecewise vector

"""
foo = np.arange(100).reshape((10,10))
print(foo[2])
foo[2, ~np.array([1,2,3])]
~np.array([1,2,3])

foo = walk_bifurcation(small_samp, 49, 83)
rows, cols = zip(*foo)

plt.figure(figsize=(14,10))
plt.pcolormesh(small_samp, cmap='Greys')
plt.colorbar()
plt.plot(cols, rows)
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
ax.xaxis.tick_top()
plt.show()

foo = walk_bifurcation(small_samp_2, 149, 883)
rows, cols = zip(*foo)

plt.figure(figsize=(14,10))
plt.pcolormesh(small_samp_2, cmap='Greys')
plt.colorbar()
plt.plot(np.array(cols) - 10, rows)
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
ax.xaxis.tick_top()
plt.show()
"""
