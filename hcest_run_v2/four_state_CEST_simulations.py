#!/usr/bin/env python
# coding: utf-8

# In[1]:


from uncertainties import ufloat
import uncertainties.unumpy as unumpy
import pandas as pd
import numpy as np
import sys
from nmrglue import proc_base
from nmrglue import analysis
import nmrglue as ng
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as st
from scipy.optimize import curve_fit
import matplotlib
from sklearn.metrics import r2_score
import pickle
from tqdm.auto import tqdm, trange
from uncertainties.umath import *
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['legend.title_fontsize'] = 30
sns.set_context('poster')
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def plot_CEST_cor(path):
#     df = pd.read_csv('/Users/beiliu/SELOPE/Exp/LSRC600/multi_hpA6DNA/fit/result.csv')
    colors_plot = ['k', 'r', 'b', 'g', 'cyan', 'magenta', 'brown', 'yellow', 'teal', 'lightgreen']
    df = pd.read_csv(path)
    slp =  df['slp(hz)'].unique()
    fig = plt.figure(figsize = (12,12))
    for i,slp1 in enumerate(slp):
        sl200_neg = df[(df['slp(hz)']==slp1) & (df['offset(hz)']<0)]['norm_intensity']
        sl200_pos = df[(df['slp(hz)']==slp1) & (df['offset(hz)']>0)]['norm_intensity']

#         sl500_neg = df[(df['slp(hz)']==slp2) & (df['offset(hz)']<0)]['norm_intensity']
#         sl500_pos = df[(df['slp(hz)']==slp2) & (df['offset(hz)']>0)]['norm_intensity']
        plt.scatter(sl200_neg[::-1], sl200_pos, s = 20, color = colors_plot[i], label='{:.0f}'.format(slp1))
#         plt.scatter(sl500_neg[::-1], sl500_pos, s = 8)
    plt.xlim(0,np.max(df['norm_intensity'])*1.1)
    plt.ylim(0,np.max(df['norm_intensity'])*1.1)
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), 'k')
    plt.xlabel('Norm. Intensity ($\Omega$<0)')
    plt.ylabel('Norm. Intensity ($\Omega$>0)')
    
    l=plt.legend(loc = 4, fontsize = 30, frameon=False,
    handletextpad=-1.8, handlelength=0, markerscale=0,
    title = '$\omega$' + ' $2\pi$' + '$^{-1}$'+ ' (Hz)',
    ncol=3, columnspacing=2)
    l._legend_box.align = "right"

    for item in l.legendHandles:
        item.set_visible(False)
    for handle, text in zip(l.legendHandles, l.get_texts()):
        text.set_color(handle.get_facecolor()[0])
#     return df


# In[3]:


def MatrixBM_4state(k12, k21, k23, k32, k34, k43, k45, k54, 
                    delta1, delta2, delta3, delta4, 
                    w1, 
                    R1a, R2a, R1b, R2b, R1c, R2c, R1d, R2d, 
                    pa, pb, pc, pd):
    global params

    # Find range of SLPs to simulate inhomogeniety
    if params['inhomo'] != 0.0:
        sigma_slp = w1 * params['inhomo']
        slps_net = np.linspace(w1 - (2*sigma_slp), w1 + (2*sigma_slp), params['number_inhomo_slps'])
    else:
        slps_net = [w1]
    
    # Initialize net BM matrix variable
    BM_Mat = np.zeros((13, 13, params['number_inhomo_slps']))
    
    counter = 0
    for dummy_slp in slps_net: 
        # Reaction rates matrix
        K = np.array([[0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , -k12-k54, 0.0     , 0.0     , k21     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , k45     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , -k12-k54, 0.0     , 0.0     , k21     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , k45     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , -k12-k54, 0.0     , 0.0     , k21     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , k45     ],
                      [0.0     , k12     , 0.0     , 0.0     , -k23-k21, 0.0     , 0.0     , k32     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , k12     , 0.0     , 0.0     , -k23-k21, 0.0     , 0.0     , k32     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , k12     , 0.0     , 0.0     , -k23-k21, 0.0     , 0.0     , k32     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , 0.0     , k23     , 0.0     , 0.0     , -k32-k34, 0.0     , 0.0     , k43     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , k23     , 0.0     , 0.0     , -k32-k34, 0.0     , 0.0     , k43     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , k23     , 0.0     , 0.0     , -k32-k34, 0.0     , 0.0     , k43     ],
                      [0.0     , k54     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , k34     , 0.0     , 0.0     , -k43-k45, 0.0     , 0.0     ],
                      [0.0     , 0.0     , k54     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , k34     , 0.0     , 0.0     , -k43-k45, 0.0     ],
                      [0.0     , 0.0     , 0.0     , k54     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , k34     , 0.0     , 0.0     , -k43-k45]], dtype=float) 


        # Relaxation rates matrix 
        R = np.array([[0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , -R2a    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , -R2a    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [R1a*pa  , 0.0     , 0.0     , -R1a    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , 0.0     , -R2b    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R2b    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [R1b*pb  , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R1b    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R2c    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R2c    , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [R1c*pc  , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R1c    , 0.0     , 0.0     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R2d    , 0.0     , 0.0     ],
                      [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R2d    , 0.0     ],
                      [R1d*pd  , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R1d    ]], dtype=float)

        # Offsets matrix
        OMEGA = np.array([[0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , 0.0     , -delta1 , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , delta1  , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -delta2 , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , delta2  , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -delta3 , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , delta3  , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -delta4 , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , delta4  , 0.0     , 0.0     ],
                          [0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ]], dtype=float)

        # Spin-lock matrix  
        SL = np.array([[0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , dummy_slp, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , -dummy_slp, 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , dummy_slp, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , 0.0      , -dummy_slp, 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , dummy_slp, 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , -dummy_slp, 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , dummy_slp],
                       [0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0     , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , -dummy_slp, 0.0     , 0.0      ]], dtype=float)
        

        
        # Store matrix slice in master variable
        BM_Mat[:, :, counter] = (K + R + OMEGA + SL) 

        counter = counter + 1

    
    return BM_Mat


# matrix exponent
def matrix_exponential(bm_matrix, time_point, status_complexvals):
    ''' Matrix exponent using bm_matrix at given time point '''
    # Sanity check
    if np.isnan(bm_matrix).any() == True or np.isinf(bm_matrix).any() == True:
        print ("There are nan elements in BM matrix!!!")
        sys.exit(0)
                
    # Get the eigen decomposition of the B-M matrix
    bm_matrix_evals, bm_matrix_evecs = np.linalg.eig(bm_matrix*time_point)

    if status_complexvals == 'off':
        if np.iscomplexobj(bm_matrix_evals):
            bm_matrix_evals = bm_matrix_evals.real

    # Compute matrix exponent using eigen decomposition
    mat_exp_value = np.dot(np.dot(bm_matrix_evecs, np.diag(np.exp(bm_matrix_evals))), np.linalg.inv(bm_matrix_evecs))
   
    # Return the real part of this matrix exponential matrix
    return mat_exp_value.real

# simulate CEST profile
def simulate_cest(params_dict, status_complex):
    ''' Simulate an acutal CEST profile given exchange parameters '''
    # Generate the weights for each SLP here
    # Inhomogeniety is same across all spins
    
    kexab = params_dict['kexAB']
    kexcd = params_dict['kexCD']
    kexbc = params_dict['kexBC']
    kexad = params_dict['kexAD']
    pa, pb, pc, pd = params_dict['pa'], params_dict['pb'], params_dict['pc'], params_dict['pd']
    
    pa = 1 - pb - pc - pd
    k12 = kexab * pb / (pb + pa)
    k21 = kexab * pa / (pa + pb)
    # k23 = kexbc * pc / (pb + pc)
    # k32 = kexbc * pb / (pb + pc)
    # k34 = kexcd * pd / (pc + pd)
    # k43 = kexcd * pc / (pc + pd)
    k45 = kexad * pa / (pa + pd)
    k54 = kexad * pd / (pa + pd)    
    
    if (pb + pc) <= 0.0:
        k23 = 0.0
        k32 = 0.0
    else:
        k23 = kexbc * pc / (pb + pc)
        k32 = kexbc * pb / (pb + pc)

    if (pd + pc) <= 0.0:
        k34 = 0.0
        k43 = 0.0
    else:
        k34 = kexcd * pd / (pc + pd)
        k43 = kexcd * pc / (pc + pd)
        
    params_dict['k12'] =k12
    params_dict['k21'] =k21
    params_dict['k23'] =k23
    params_dict['k32'] =k32  
    params_dict['k34'] =k34  
    params_dict['k43'] =k43 
    params_dict['k45'] =k45  
    params_dict['k54'] =k54 
    
    if params_dict['inhomo'] == 0.0:
        weights_slps = [1.0]
    else:
        w1_dummy = 10.0
        sigma_slp = w1_dummy * params_dict['inhomo']
        slps_net = np.linspace(w1_dummy - (2*sigma_slp), w1_dummy + (2 * sigma_slp), params['number_inhomo_slps'])
        weights_slps = (np.exp((-1*np.square(slps_net-w1_dummy))/(2*sigma_slp*sigma_slp))/math.sqrt(2*math.pi*sigma_slp*sigma_slp))

    BM_mat = MatrixBM_4state(params_dict['k12'], params_dict['k21'], params_dict['k23'], params_dict['k32'], params_dict['k34'], params_dict['k43'], params_dict['k45'], params_dict['k54'],
                             params_dict['delta1'][0], params_dict['delta2'][0], params_dict['delta3'][0], params_dict['delta4'][0] ,params_dict['slps'][0], 
                             params_dict['R1a'], params_dict['R2a'], params_dict['R1b'], params_dict['R2b'], params_dict['R1c'], params_dict['R2c'], params_dict['R1d'], params_dict['R2d'],
                             params_dict['pa'], params_dict['pb'], params_dict['pc'], params_dict['pd'])

    # Define starting magnetization
    # (unit vector, GS_x, GS_y, GS_z, ES1_x, ES1_y, ES1_z, ES2_x, ES2_y, ES3_z, ES3_x, ES3_y, ES3_z)
    if params_dict['equil'] == 'Y':
#         print ("*** COMPLETE EQUILIBRATIONi ***")
        M0_plusx = np.array([1.0, 0.0, 0.0, params_dict['pa'], 0.0, 0.0, params_dict['pb'], 0.0, 0.0, params_dict['pc'], 0.0, 0.0, params_dict['pd']])
        M0_minusx = np.array([1.0, 0.0, 0.0, -1.0*params_dict['pa'], 0.0, 0.0, -1.0*params_dict['pb'], 0.0, 0.0, -1.0*params_dict['pc'], 0.0, 0.0, -1.0*params_dict['pd']])
    elif params_dict['equil'] == 'N':
#         print ("*** NO EQUILIBRATIONi ***")
        M0_plusx = np.array([1.0, 0.0, 0.0, params_dict['pa'], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        M0_minusx = np.array([1.0, 0.0, 0.0, -1.0*params_dict['pa'], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Now, simulate!
    # Initial time point
    # Take into account phase cycling
    time_point = 0.0
    intensities_initial = np.zeros(13)
    for dummy_inhomo in range(params['number_inhomo_slps']):
        intensities_plusx_initial = np.dot(matrix_exponential(BM_mat[:,:,dummy_inhomo], time_point, status_complex), M0_plusx) 
        intensities_minusx_initial = np.dot(matrix_exponential(BM_mat[:,:,dummy_inhomo], time_point, status_complex), M0_minusx) 
        intensities_initial = intensities_initial + ((intensities_plusx_initial-intensities_minusx_initial)*weights_slps[dummy_inhomo])

    # Get errors on intensity matrix
#     print ('base inten', params['error_baseinten'])
    intensities_initial_pointerror = intensities_initial * params['error_point']
    intensities_initial_error = intensities_initial * params['error_baseinten']
    intensities_initial = intensities_initial 

    # Define variable for the magnetization of each state at each slp, offset value
    #intensities_net = np.zeros((10,params_dict['slps'].shape[0]))
    intensities_net = np.zeros((params_dict['slps'].shape[0]))
    intensities_net = unumpy.uarray(intensities_net, intensities_net)
 
    for dummy in range(len(params_dict['slps'])):
        #print 'SLP = ', params_dict['slps'][dummy]/(2*math.pi), ' Hz'
        #print 'Offset = ', params_dict['offsets'][dummy]/(2*math.pi), ' Hz'
        #print 'Offset GS = ', params_dict['delta1'][dummy]/(2*math.pi), ' Hz'
        #print 'Offset ES1 = ', params_dict['delta2'][dummy]/(2*math.pi), ' Hz'
        #print 'Offset ES2 = ', params_dict['delta3'][dummy]/(2*math.pi), ' Hz'
        ##print params_dict['wa'], params_dict['wb'], params_dict['wc'], params_dict['obs_peakpos']
        #print
    
        # Get the B-M matrix for this particular SLP and Offset combination
        BM_mat = MatrixBM_4state(params_dict['k12'], params_dict['k21'], params_dict['k23'], params_dict['k32'], params_dict['k34'], params_dict['k43'], params_dict['k45'], params_dict['k54'],
                                 params_dict['delta1'][dummy], params_dict['delta2'][dummy], params_dict['delta3'][dummy], params_dict['delta4'][dummy] ,params_dict['slps'][dummy], 
                                 params_dict['R1a'], params_dict['R2a'], params_dict['R1b'], params_dict['R2b'], params_dict['R1c'], params_dict['R2c'], params_dict['R1d'], params_dict['R2d'],
                                 params_dict['pa'], params_dict['pb'], params_dict['pc'], params_dict['pd'])

        # Now, simulate at time point of interest
        time_point = params_dict['T']
        #print params_dict['T']
        intensities = np.zeros(13)
        for dummy_inhomo in range(params['number_inhomo_slps']):
            intensities_plusx = np.dot(matrix_exponential(BM_mat[:,:,dummy_inhomo], time_point, status_complex), M0_plusx) 
            intensities_minusx = np.dot(matrix_exponential(BM_mat[:,:,dummy_inhomo], time_point, status_complex), M0_minusx) 
            intensities = intensities + ((intensities_plusx-intensities_minusx)*weights_slps[dummy_inhomo])

        # Now, normalize this intensity
        # Key point is that normalization will be different if exchange regime is different
        # Put alternatively, super slow exchange ==> GS_mag(t)/GS_mag(0)
        # At super fast exchange ==> (GS_mag(t) + ES_mag(t)) / (GS_mag(0) + ES_mag(0))
        # Things could be more complicated wherein GS <-> ES1 is fast while GS <-> ES2 is slow
        # One can accordingly define more complicated normalization schemes
        # Here it would be  ==> (GS_mag(t) + ES1_mag(t)) / (GS_mag(0) + ES1_mag(0))
        #intensities_normalized = np.divide(intensities, intensities_initial)
        #print params_dict['mode'], len(params_dict['mode']) 
        if params_dict['mode'] == 'GS':
            #print "GS Alignment"
            intensities_net[dummy] = ufloat(intensities[3]+np.random.normal(0.0, intensities_initial_pointerror[3],1), intensities_initial_error[3]) / intensities_initial[3]
        else:
            #print "AVG Alignment"
            denominator = (intensities_initial[3] + intensities_initial[6] + intensities_initial[9] + intensities_initial[12])
            intensities_net[dummy] = ufloat((intensities[3]+intensities[6]+intensities[9]+intensities[12]+np.random.normal(0.0, intensities_initial_pointerror[3],1)+np.random.normal(0.0, intensities_initial_pointerror[6],1)+np.random.normal(0.0,intensities_initial_pointerror[9],1)+np.random.normal(0.0,intensities_initial_pointerror[12],1)), intensities_initial_error[3]) / denominator 
 
    return intensities_net  

# get parameters from BMNS.txt file
def input_parser(input_filename, slp_exp=[], offset_exp=[]):
    ''' parses input parameters file '''
    params_dict = {}

    f = open(input_filename, 'r')

    # Parse until you get to a +
    while 1:
        line = f.readline()
        if line[0] == '+':
            break

    # Now, keep reading until you get to a blank line
    # Parse the lines to get all relevant exchange parameters
    while 1:
        line = f.readline()
        line = line.strip('\n')
        if len(line) <= 1:
            break
        line = line.split(' ')
        if str(line[0]) != 'mode' and str(line[0]) != 'equil' and str(line[0]) != 'ls':
            params_dict[str(line[0])] = float(line[1])
        else:
            params_dict[str(line[0])] = str(line[1])
            
    if params_dict['mode'] == "AUTO":
        # Compute exchange regime & decide alignment based on the same
        if (params_dict['dwb'] != 0.0):
            exch_regime = params_dict['kexAB']/(params_dict['dwb']*2*math.pi*params_dict['lf'])
        else:
            exch_regime = 0.1

        # Now, the exchange regime has been computed
        # decide alignment
        if exch_regime <= 1:
            params_dict['mode'] = 'GS'
#             print ("Auto decision, mode=GS")
        else:
            params_dict['mode'] = 'AVG'
#             print ("Auto decision, mode=AVG")
            
    # Sanity checking
    if params_dict['mode'] not in ['GS', 'AVG']:
        print ("!!! Enter GS for GS alignment and AVG for avg alignment !!!" )
        sys.exit(0)
    if params_dict['equil'] not in ['Y', 'N']:
        print ("!!! Please enter Y to allow equilibration and N for no equilibration !!!")
        sys.exit(0)
    if params_dict['ls'] not in ['Y', 'N']:
        print ("!!! Wrong lineshape input. Enter Y for performing a lineshape simulation to get peak position, ")
        print ("and N to assume observed peak is GS peak !!!")
        sys.exit(0)

    if params_dict['inhomo'] == 0:
        params_dict['number_inhomo_slps'] = 1
    else:
        params_dict['number_inhomo_slps'] = 20

    # Convert into appropriate units
    # and define missing parameters
    params_dict['pa'] = 1 - params_dict['pb'] - params_dict['pc'] - params_dict['pd']
    params_dict['wa'] = 0.0
    params_dict['wb'] = (params_dict['dwb'] - params_dict['wa']) * params_dict['lf'] * 2 * math.pi
    params_dict['wc'] = (params_dict['dwc'] - params_dict['wa']) * params_dict['lf'] * 2 * math.pi
    params_dict['wd'] = (params_dict['dwd'] - params_dict['wa']) * params_dict['lf'] * 2 * math.pi 
    

    # Now, key point - do the lineshape simulation
    # to get observed peak position
#     if params_dict['ls'] == "Y":
#         print "***LINESHAPE SIMULATION***"    
#         [observed_peakpos, observed_peakheight] = ls_3state(params_dict['pa'], params_dict['pb'], params_dict['pc'], params_dict['k12'], params_dict['k21'], params_dict['k13'], params_dict['k31'], params_dict['k23'], params_dict['k32'], params_dict['wa'], params_dict['wb'], params_dict['wc'], params_dict['R2a'], params_dict['R2b'], params_dict['R2c'], params_dict['lf'], params_dict['resn'])
#         print "Observed peakpos = ", '%4.3f'%(observed_peakpos/(2*math.pi)), " Hz, ", '%4.3f'%((observed_peakpos/(2*math.pi))/params_dict['lf']), " ppm"
#     else:
#         observed_peakpos = 0.0
    params_dict['obs_peakpos'] = 0.0

    #for ele in params_dict.keys():
    #    print ele, params_dict[ele]
 
    # Keep reading all the comment lines
    while 1:
        line = f.readline()
        if line[0] == '+':
            break

    # Read until you get to blank line
    # Parse the spin-lock power offset combinations
    slps = np.array([])
    offset = np.array([])
    if len(slp_exp) >= 1:
        for slp in slp_exp:        
            # use experimental offsets for the simulation
            slps = np.append(slps, np.tile(np.array([slp]), int(len(offset_exp))))

        f.close()

        # Add the slps and offsets to params_dict
        params_dict['offsets'] = np.tile(offset_exp, int(len(slps)/len(offset_exp)))  # experimental offsets
        params_dict['slps'] = slps

        # Key point - convert offset and slp into Hz
        params_dict['offsets'] = params_dict['offsets'] * 2 * math.pi
        params_dict['slps'] = params_dict['slps'] * 2 * math.pi

        # Now, define the offsets, i.e., delta values for the 3 states 
        # Offset_peak = Offset + (obs_peakpos - w_peak)
        params_dict['delta1'] = -params_dict['offsets'] - (params_dict['obs_peakpos'] - params_dict['wa'])
        params_dict['delta2'] = -params_dict['offsets'] - (params_dict['obs_peakpos'] - params_dict['wb'])
        params_dict['delta3'] = -params_dict['offsets'] - (params_dict['obs_peakpos'] - params_dict['wc'])
        params_dict['delta4'] = -params_dict['offsets'] - (params_dict['obs_peakpos'] - params_dict['wd'])
    
    else:
        while 1:
            line = f.readline()
            if len(line) <= 1:
                break
            line = line.split(' ')
            lower_limit = float(line[2])
            upper_limit = float(line[3])
            no_points = int(line[4])
            spinlock_power = float(line[1])
            offset = np.append(offset, np.linspace(lower_limit, upper_limit, no_points))
            slps = np.append(slps, np.tile(np.array([spinlock_power]), no_points))

            # use experimental offsets for the simulation
#             slps = np.append(slps, np.tile(np.array([spinlock_power]), int(len(offset_exp)/2)))      


        f.close()

        # Add the slps and offsets to params_dict
        params_dict['offsets'] = offset  # experimental offsets
        params_dict['slps'] = slps

        # Key point - convert offset and slp into Hz
        params_dict['offsets'] = params_dict['offsets'] * 2 * math.pi
        params_dict['slps'] = params_dict['slps'] * 2 * math.pi

        # Now, define the offsets, i.e., delta values for the 3 states 
        # Offset_peak = Offset + (obs_peakpos - w_peak)
        params_dict['delta1'] = params_dict['offsets'] + (params_dict['obs_peakpos'] - params_dict['wa'])
        params_dict['delta2'] = params_dict['offsets'] + (params_dict['obs_peakpos'] - params_dict['wb'])
        params_dict['delta3'] = params_dict['offsets'] + (params_dict['obs_peakpos'] - params_dict['wc'])
        params_dict['delta4'] = -params_dict['offsets'] - (params_dict['obs_peakpos'] - params_dict['wd'])
    return params_dict

# plot cest profile
def plot_profiles(params, intensities, save = False, output_name=None):
    ''' Given the parameters, intensity from simulation, plot the CEST profile '''
    # Correct CEST output for fast exchange = run lineshape with pB at the end of the CEST period
    # Determine intensity at observed peak position
    # Plot intensity
    # For now, plot GS peak intensity only
    number_slps = np.unique(params['slps'])
    colors_plot = ['k', 'r', 'b', 'g', 'cyan', 'magenta', 'brown', 'yellow', 'teal', 'lightgreen']
    # Plot profile
    sns.set_context('poster')
    plt.figure(1, figsize=(10, 8))
    counter = 0
    for dummy_slp in number_slps:
        plt.errorbar((params['offsets'][np.where(params['slps'] == dummy_slp)]/(2*math.pi*params['lf'])), 
                     unumpy.nominal_values(intensities[np.where(params['slps'] == dummy_slp)]), 
                     yerr=unumpy.std_devs(intensities[np.where(params['slps'] == dummy_slp)]), 
                     color = colors_plot[counter], label='%4.0f'%float(dummy_slp/(2*math.pi)), 
                     linewidth=5, fmt='o', markersize=10)
        
#         plt.errorbar((params['offsets'][np.where(params['slps'] == dummy_slp)]/(2*math.pi*params['lf'])), 
#              ref_df[np.abs(ref_df['slp(hz)']-dummy_slp)<1]['norm_intensity'], 
#              color = colors_plot[counter], 
#              linewidth=5, fmt='o', markersize=10, alpha=0.5)

        counter = counter + 1
    plt.xlim([((np.amin(params['offsets'])-(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf'])), 
              ((np.amax(params['offsets'])+(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf']))])
#     plt.ylim([-0.1, 1])
#     plt.vlines(2.4, -0.1, 0.6, linestyle='dashed')
#     plt.vlines(1.6, -0.1, 0.6, linestyle='dashed')
#     plt.legend(loc=0)

    plt.ylabel('$I/I_{0}$')
    plt.xlabel('$\Omega(ppm)$')
    
    l=plt.legend(loc = 4, fontsize = 30, frameon=False,
       handletextpad=-2.0, handlelength=0, markerscale=0,
       title = '$\omega$' + ' $2\pi$' + '$^{-1}$'+ ' (Hz)',
       ncol=3, columnspacing=2.2)
    l._legend_box.align = "right"

    for item in l.legendHandles:
        item.set_visible(False)
    for handle, text in zip(l.legendHandles, l.get_texts()):
        text.set_color(handle.get_color()[0])
    plt.tight_layout()
    
    if save:
        plt.savefig(output_name + '.pdf', dpi= 300)
        
# function to get the output (CEST profile + csv file containing simulated data) for each pathway
# file name is the BMNS.txt file name
def get_output(input_filename, slp_exp=[], offset_exp=[], save=False, output_name=None, plot = True):
    
    global params
    
    # get the parameters from BMNS.txt file
    params = input_parser(input_filename, slp_exp = slp_exp, offset_exp = offset_exp)
    
    # simulate cest profile
    intensity_mag_complex = simulate_cest(params, 'on')

    # plot the file
    if plot:
        plot_profiles(params, intensity_mag_complex, save=save, output_name=output_name)
    
    # get parameters for the output csv file
    output_dict = {}
    output_dict['slp(hz)'] = params['slps']/(2*math.pi)
    output_dict['offset(hz)'] = params['offsets']/(2*math.pi)
    output_dict['norm_intensity'] = unumpy.nominal_values(intensity_mag_complex)
    output_dict['norm_intensity_error'] = unumpy.std_devs(intensity_mag_complex)
#     output_dict['norm_intensity_nocomplex'] = unumpy.nominal_values(intensity_mag_nocomplex)
#     output_dict['norm_intensity_nocomplex_error'] = unumpy.std_devs(intensity_mag_nocomplex)
    output_dict['trelax(s)'] = np.ones(intensity_mag_complex.shape)*params['T']

    bf=pd.DataFrame(data=output_dict)

    # create output csv
    output_filename = output_name
    
    if save:

        print (output_filename)
        bf.to_csv(output_filename)
    
    return bf, params

def get_offset_exp(path):
    '''
    function to extract offsets and spin lock power from experimental data
    path is the path to data csv
    '''
    df_exp_result = pd.read_csv(path)
    
    slp_exp = (np.round(df_exp_result['slp(hz)'].unique(),0)).astype(int)
    
    offset_exp = np.array(df_exp_result[np.round(df_exp_result['slp(hz)']) == slp_exp[0]]['offset(hz)'])
    
    return slp_exp, offset_exp, df_exp_result


# In[ ]:




