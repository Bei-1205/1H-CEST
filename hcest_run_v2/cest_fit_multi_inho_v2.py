##!/usr/bin/python

from numpy import linspace, zeros, array, amax, abs, dot, diag, exp, arange, sum, argmax, argmin, isnan, isinf, square, divide, unique, where, amin, ones, mean, std, sqrt
from numpy.linalg import eig, inv
from numpy.random import normal
import pandas as pd
import uncertainties.unumpy as unumpy
import uncertainties.umath as umath
from uncertainties import ufloat
import sys
import os
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nmrglue import proc_base
from nmrglue import analysis
import time
from multiprocessing import Manager, Process, Pipe, cpu_count
import seaborn as sns
from matplotlib import rcParams
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
# rcParams['legend.title_fontsize'] = 15
sns.set_context('poster', font_scale=1.2)

### update in this fitting script
### specify inhomogeneity for each slp
### in the input BMNS txt file
### specify different inhomogeneity values for different slp (ascending order)
# e.g. slp 50hz, 100hz, 250hz, 1000hz
# inhomo 0.05 0.03 0.02 0.01

# Generic global fitting implemented
# with spin-lock inhomogeneity
# with J coupling handling - with unit conversion
# global arg passing 
# AVG alignment vs. GS alignment treated in simulate_cest_offsets
# MC iterations have been parallelized 
# AUTO alignment choice implemented
# Input files copied at start

# some global initializations
# Variables for storing intensity and their errors for all data sets
intensity_master = []
intensity_master_error = []

# entire data for one data set
params = []

# global data variable
data_sets = []

# net list of distinct (non shared) variables for each data set
net_vars = []

# bound variable for fixed params
eps = 1.0*math.pow(10, -12)

# list of all variables for a given data set
params_fit = ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R2b', 'R2c', 'R1b', 'R1c']

# print variable for type of fit
print_counter = 0

# variable to store fit type based on nature of relaxation constants that are supplied
params_fit_dict = {}
params_fit_dict['0000'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2']
params_fit_dict['1000'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R2b']
params_fit_dict['0100'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R2c']
params_fit_dict['0010'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1b']
params_fit_dict['0001'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1c']
params_fit_dict['1100'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R2b', 'R2c']
params_fit_dict['1010'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1b', 'R2b']
params_fit_dict['1001'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1c', 'R2b']
params_fit_dict['0110'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1b', 'R2c']
params_fit_dict['0101'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1c', 'R2c']
params_fit_dict['0011'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1b', 'R1c']
params_fit_dict['1110'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1b', 'R2b', 'R2c']
params_fit_dict['1101'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R2b', 'R1c', 'R2c']
params_fit_dict['1011'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1b', 'R2b', 'R1c']
params_fit_dict['0111'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1b', 'R1c', 'R2c']
params_fit_dict['1111'] =  ['pB', 'pC', 'dwB', 'dwC', 'kexAB', 'kexAC', 'kexBC', 'R1', 'R2', 'R1b', 'R2b', 'R1c', 'R2c']


def fun(f,q_in,q_out):
    ''' Function for parallelization '''
    while True:
       i,x = q_in.get()
       if i is None:
           break
       q_out.put((i,f(x)))

def parmap(f, X, nprocs = cpu_count()):
    ''' Function for parallelization '''
    m = Manager()
    q_in   = m.Queue(1)
    q_out  = m.Queue()
    print "***", nprocs

    proc = [Process(target=fun,args=(f,q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
      p.daemon = True
      p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]

    return [x for i,x in sorted(res)]

# Define BM Matrix for CEST
def MatrixBM_3state(k12, k21, k13, k31, k23, k32, delta1, delta2, delta3, w1, R1a, R2a, R1b, R2b, R1c, R2c, pa, pb, pc, inh=0.0):
    ''' B-M matrix for 3-state CEST '''
    # This is the best place to handle the inhomogeniety
    # this function is passed a single w1 value
    global params
 
    # Find range of SLPs to simulate inhomogeniety
    # if inhomogeniety is to be simulated in the 1st place
    if inh != 0.0:
        sigma_slp = w1 * inh
        slps_net = linspace(w1 - (2 * sigma_slp), w1 + (2 * sigma_slp), params['number_inhomo_slps'])
    else:
        slps_net = [w1]

    # Initialize the net BM matrix variable
    BM_Mat = zeros((10, 10, params['number_inhomo_slps']))
   
    # Generate BM matrix indibidually for each SLP
    counter = 0
    for dummy_slp in slps_net:
        # Reaction rates matrix
        K = array([[0.0, 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0, -k12-k13, 0.0     , 0.0     , k21     , 0.0     , 0.0     , k31     , 0.0     , 0.0     ],
                      [0.0, 0.0     , -k12-k13, 0.0     , 0.0     , k21     , 0.0     , 0.0     , k31     , 0.0     ],
                      [0.0, 0.0     , 0.0     , -k12-k13, 0.0     , 0.0     , k21     , 0.0     , 0.0     , k31     ],
                      [0.0, k12     , 0.0     , 0.0     , -k21-k23, 0.0     , 0.0     , k32     , 0.0     , 0.0     ],
                      [0.0, 0.0     , k12     , 0.0     , 0.0     , -k21-k23, 0.0     , 0.0     , k32     , 0.0     ],
                      [0.0, 0.0     , 0.0     , k12     , 0.0     , 0.0     , -k21-k23, 0.0     , 0.0     , k32     ],
                      [0.0, k13     , 0.0     , 0.0     , k23     , 0.0     , 0.0     , -k31-k32, 0.0     , 0.0     ],
                      [0.0, 0.0     , k13     , 0.0     , 0.0     , k23     , 0.0     , 0.0     , -k31-k32, 0.0     ],
                      [0.0, 0.0     , 0.0     , k13     , 0.0     , 0.0     , k23     , 0.0     , 0.0     , -k31-k32]], dtype=float)
  
        # Relaxation rates matrix 
        R = array([[0.0   , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0   , -R2a    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0   , 0.0     , -R2a    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [R1a*pa, 0.0     , 0.0     , -R1a    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0   , 0.0     , 0.0     , 0.0     , -R2b    , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [0.0   , 0.0     , 0.0     , 0.0     , 0.0     , -R2b    , 0.0     , 0.0     , 0.0     , 0.0     ],
                      [R1b*pb, 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R1b    , 0.0     , 0.0     , 0.0     ],
                      [0.0   , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R2c    , 0.0     , 0.0     ],
                      [0.0   , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R2c    , 0.0     ],
                      [R1c*pc, 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -R1c    ]], dtype=float)

        # Spin-lock matrix
        SL = array([[0.0, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0, 0.0       , 0.0     , dummy_slp, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0, -dummy_slp, 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , dummy_slp, 0.0       , 0.0     , 0.0      ],
                       [0.0, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0, 0.0       , 0.0     , 0.0      , -dummy_slp, 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , dummy_slp],
                       [0.0, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      ],
                       [0.0, 0.0       , 0.0     , 0.0      , 0.0       , 0.0     , 0.0      , -dummy_slp, 0.0     , 0.0      ]], dtype=float)

        # Offsets matrix
        OMEGA = array([[0.0, 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0, 0.0     , -delta1 , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0, delta1  , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0, 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0, 0.0     , 0.0     , 0.0     , 0.0     , -delta2 , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0, 0.0     , 0.0     , 0.0     , delta2  , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0, 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ],
                          [0.0, 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , -delta3 , 0.0     ],
                          [0.0, 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , delta3  , 0.0     , 0.0     ],
                          [0.0, 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     , 0.0     ]], dtype=float)
 
        # Store B-M matrix in master variable
        BM_Mat[:, :, counter] = (K + R + SL + OMEGA)
    
        # Incremenet matrix counter
        counter = counter + 1

    # Return master BM matrix
    return BM_Mat

def ls_3state(pa, pb, pc, k12, k21, k13, k31, k23, k32, wa, wb, wc, R2a, R2b, R2c, frq0, resn):
    ''' Do a 3-state lineshape simulation to get observed peak position '''
    # Pass the function the populations and individual rate constants 
    # populations = fractions or % of 1
    # rates = /s
    # w = rad/s
    # R2 = /s
    # frq0 = MHz
    # Resolution (resn) = Hz
 
    # Based on resolution, get acquisition time
    aq = 1/resn
  
    # Set sweep width, as 1.5x the greater of the two dWs. 
    # This sw is in Hz
    sw = amax(1.5*abs(array([wb, wc]))/(2*math.pi))
    
    # Now, determine number of points
    N = int(aq * 2 * sw)
    
    # Get dwell time 
    dt = aq / N
    
    # Set initial magnetization
    M0 = array([pa, pb, pc])
    
    # Title of plot
    # As the gyromagnetic ratio for Carbon is positive
    # w = -gamma * B
    # The - in the omega and in the matrix equation cancel out to 
    # give a +ve number
    R = zeros((3, 3), dtype=complex)
    R[0, 0] = -R2a + 1j*wa - k12 - k13
    R[0, 1] = k21
    R[0, 2] = k31
    R[1, 0] = k12
    R[1, 1] = -R2b + 1j*wb - k21 - k23
    R[1, 2] = k32
    R[2, 0] = k13
    R[2, 1] = k23
    R[2, 2] = -R2c + 1j*wc - k32 - k31
    
    # v is eigenvalue
    # G is eigenvector
    # G_1 is inverse of eigenvector matrix
    v,G = eig(R)
    G_1 = inv(G)
    
    # T is time domain data
    # dM/dT = A . M
    # Solution is M = M0 e^(At)
    # Then expand A according to eigen value equation
    T = linspace(0., aq, N)
    fid = zeros(T.shape, dtype=complex)
    for i,t in enumerate(T):
        A = dot(dot(G,diag(exp(v*t))), G_1)
        fid[i] = sum(dot(A, M0))
    
    data = proc_base.fft(fid).real
    
    # 1/dwell_time = 2 * sweep_width
    # we can go sweep width on either side
    # This interval is divided into the same # of points as time
    # domain data
    # Units of xf = Hz
    xf = arange(-1./(2*dt), 1./(2*dt), 1./dt/N)
    yf = data
    #print xf.shape, yf.shape 
    #plt.figure(1)
    #plt.plot(xf, yf, 'r-', lw=2, marker='o', markersize=5)
   

    # Find the peak width
    max_height = amax(yf)
    #print "SEE", max_height

    # Pick peaks
    #data = data[1:-1]
    #peaks = analysis.peakpick.pick(data, pthres=max_height/50.0, est_params=False, cluster=False, algorithm='downward')
    #print "peak fitting = ", peaks

    # Find peak maximum and convert to rad/s
    max_position = xf[argmax(yf)] * 2 * math.pi
    #print "COMP", argmax(yf)
    #print "peak position =", max_position, " Hz"
    #print "peak height =", max_height, " Hz"
    #plt.plot([max_position/(2*math.pi), max_position/(2*math.pi)], [0, max_height], color='k')

    #yf_side1 = absolute(yf-(max_height*0.5))
    #xf_side1 = xf[argmin(yf_side1)]
    #peak_width = (xf_side1 - xf[argmax(yf)])*frq0*2
    #print "Peak width = ", xf_side1, peak_width
    
    #plt.show()
 
    # return peak position in rad/s
    return [max_position, max_height]



def matrix_exponential(bm_matrix, time_point):
    ''' Matrix exponent using bm_matrix at given time point '''
    # Sanity check
    if isnan(bm_matrix).any() == True or isinf(bm_matrix).any() == True:
        print "There are nan elements in BM matrix!!!"
        sys.exit(0)
    
    # Get the eigen decomposition of the B-M matrix
    bm_matrix_evals, bm_matrix_evecs = eig(bm_matrix*time_point)

    # Compute matrix exponent using eigen decomposition
    mat_exp_value = dot(dot(bm_matrix_evecs, diag(exp(bm_matrix_evals))), inv(bm_matrix_evecs))
   
    # Return the real part of this matrix exponential matrix
    return mat_exp_value.real

def input_parser(input_filename):
    ''' parses input parameters file '''
    global eps
    global params_fit
    global data_sets

    # open input file
    f = open(input_filename, 'r')

    # Parse until you get to a +
    while 1:
        line = f.readline()
        if line[0] == '+':
            break

    # Keep reading until all data sets are exhausted
    # Assume there are multiple data sets to be fit
    eof_file = 0
    while 1:
        params_dict = {}

        # Extract the csv filename
        # Add error checking argument
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = str(line[1])

        # Extract the larmor frequency
        # Add error checking argument
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = float(line[1])

        # Get the alignment
        line = f.readline() 
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = str(line[1])
        if params_dict[str(line[0])] not in ['GS', 'AVG', 'AUTO']:
            print "!!! Please enter GS for GS alignment and AVG for avg alignment !!!"
            print "!!! In this fitting, alignment mode controls intensities used for computing experimental observable !!!"
            sys.exit(0)

        # Get the resn for the peak pos finder
        # Add error checking argument
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = float(line[1])

        # Get the equilibration variable
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = str(line[1])
        if params_dict[str(line[0])] not in ['Y', 'N']:
            print "!!! Please enter Y to  allow equilibration and N for no equilibration !!!"
            sys.exit(0)

        # Get the Intensity/Volume decision variable
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = str(line[1])
        
        # Get the obs_peakpos variable
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = str(line[1])
        if params_dict[str(line[0])] == 'Y':
            print "WARNING: FITTING MAY TAKE A LONG TIME \n"
        elif params_dict[str(line[0])] == 'N':
            pass
        else:
            print "!!! Wrong lineshape input. Please enter Y to perform a lineshape simulation to get "
            print " peak position and N to assume observed peak is GS peak "
            sys.exit(0)

        # Get the error estimation mode
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = str(line[1])
        if params_dict[str(line[0])] not in ['MC', 'STD']:
            print "!!! Please specify MC for monte-carlo and STD for standard error estimation !!!"
            sys.exit(0)

        # Get the number of mc iterations
        # Add error checking argument
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = int(line[1])

        # Get the SLP inhomogeniety
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = [float(i) for i in line[1:]]
        # Add the number of inhomogeneous SLPs here

        # get the inhomogeneity sampling size
        line = f.readline()
        line = line.strip('\n').split(' ')
        params_dict[str(line[0])] = int(line[1])

        #if float(line[1]) == 0.0:
        #    params_dict['number_inhomo_slps'] = 1
        #else:
        #    params_dict['number_inhomo_slps'] = 120

        print "inhomogeneity sampling size is {}".format(params_dict['number_inhomo_slps'])
    
        # Get the J coupling
        line = f.readline()
        line = line.strip('\n').split(' ')
        if line[1].isalpha() == True:
            print "!!! Input a number for the J coupling !!!"
            sys.exit(0)
        # This is very important 
        # J coupling is in Hz. Should be in units of rad/s i.e., multiply by 2pi
        params_dict[str(line[0])] = float(line[1]) * 2 * math.pi


        # Initialize variables to store input parameters that are considered in the fit
        # have separate lists for free and fixed params
        free_params_list = []
        fixed_params_list = []

        # Process all other variables
        # populations, kinetics, dW, R1, R2 
        number_fixed = 0
        number_floating = 0
        shared_vars = []
        while 1:
            # Keep reading lines until end of file is reached
            line = f.readline()
            if len(line) <= 2:
                break
      
            # Add error checking i.e., multiple spaces between arguments
            line = line.strip('\n').split(' ')
            # Each fit variable that has limits -> [guess, lower_limit, upper_limit]
            # Handle floating variables that are not shared
            # Add a 'y' or 'n' status variable for checking the status whether the variable is shared or not
            # each shared variable is a floating variable for the data set in which it is present
            # account for the fact that it is shared and represents one less fit parameter at the end
            if line[0][-1] != "!" and line[0][-1] != "*":
                number_floating = number_floating + 1
                params_dict[str(line[0])] = array([float(line[1]), float(line[2]), float(line[3])])
                #print line[0], params_dict[line[0]]
                free_params_list.append(str(line[0]))
            # Handle floating variables that are shared
            elif line[0][-1] == "*":
                number_floating = number_floating + 1
                params_dict[str(line[0][:-1])] = array([float(line[1]), float(line[2]), float(line[3])])
                #print line[0], params_dict[line[0]]
                free_params_list.append(str(line[0][:-1]))
                shared_vars.append(str(line[0][:-1]))
            # Handle fixed variables
            # by keeping small bounds for the variable
            else:
                number_fixed = number_fixed + 1
                params_dict[str(line[0][:-1])] = array([float(line[1]), float(line[1])-eps, float(line[1])+eps])
                #print line[0][:-1], params_dict[line[0][:-1]]
                fixed_params_list.append(str(line[0][:-1]))

        params_dict['free_vars'] = free_params_list
        params_dict['fixed_vars'] = fixed_params_list
        params_dict['shared_vars'] = shared_vars

        # Now in params_fit, look again in params_dict at all relaxation constants
        # If they are explicitly specified, note it down in the uneq variable
        # otherwise, initialize variable with a dummy bound
        uneq_r2b = 0
        uneq_r2c = 0
        uneq_r1b = 0
        uneq_r1c = 0
        if 'R2b' in params_dict.keys():
            uneq_r2b = 1
        else:
            number_fixed = number_fixed + 1
            params_dict['R2b'] = array([0.0, -1*eps, 1*eps])
            #print "Dummy R2b"
        
        if 'R2c' in params_dict.keys():
            uneq_r2c = 1
        else:
            number_fixed = number_fixed + 1
            params_dict['R2c'] = array([0.0, -1*eps, 1*eps])
            #print "Dummy R2c"

        if 'R1b' in params_dict.keys():
            uneq_r1b = 1
        else:
            number_fixed = number_fixed + 1
            params_dict['R1b'] = array([0.0, -1*eps, 1*eps])
            #print "Dummy R1b"

        if 'R1c' in params_dict.keys():
            uneq_r1c = 1
        else:
            number_fixed = number_fixed + 1
            params_dict['R1c'] = array([0.0, -1*eps, 1*eps])
            #print "Dummy R1c"
        print
 
        # Store the status variables for all relaxation constants in one master variable
        params_dict['relax_params'] = str(uneq_r2b) + str(uneq_r2c) + str(uneq_r1b) + str(uneq_r1c) 
  
        # print out fit vars
        #for ele in params_dict.keys():
        #    if ele in params_fit:
        #        print ele, params_dict[ele]
        #print number_fixed, params_dict['relax_params']
        print "Number of floating variables = ", number_floating
        print "Free variables", free_params_list
        print "Fixed variables", fixed_params_list

        # Sanity check to see whether all variables are accounted for
        if number_fixed + number_floating != len(params_fit):
            print "ERROR IN NUMBER OF FIXED AND FLOATING VARIABLES"
            sys.exit(0)
  
        # update overall fitting variables 
        params_dict['number_floating'] = number_floating
        params_dict['number_fixed'] = number_fixed
        params_dict['number_vars'] = number_fixed + number_floating

        # Convert the dW to rad / s
        params_dict['dwB'] = params_dict['dwB'] * 2 * math.pi * params_dict['lf']
        params_dict['dwC'] = params_dict['dwC'] * 2 * math.pi * params_dict['lf']

        # read the actual data in
        # make sure to convert everything into rad/s
        bf = pd.read_csv(params_dict['Name'])
        # sort the bf to ensure slp match the input inhomogeneity values
        bf = bf.sort_values(['slp(hz)', 'offset(hz)'], ascending=[True, True])
        bf.to_csv(params_dict['Name'])
        if params_dict['fitvar'] == 'I':
            print "Intensities used for fitting"
            params_dict['norm_intensity'] = array(bf['norm_intensity'])
            params_dict['norm_intensity_error'] = array(bf['norm_intensity_error'])
        elif params_dict['fitvar'] == 'V':
            print "Volumes used for fitting"
            params_dict['norm_intensity'] = array(bf['norm_volume'])
            params_dict['norm_intensity_error'] = array(bf['norm_volume_error'])
        else:
            print "!!! Please specify I for fitting intensity and V for fitting volume !!!"
            sys.exit(0)

        params_dict['offset'] = array(bf['offset(hz)']) * 2 * math.pi
        params_dict['slp'] = array(bf['slp(hz)']) * 2 * math.pi
        params_dict['trelax'] = array(bf['trelax(s)'])
        
        slp_list = bf['slp(hz)'].unique()
        print "inhomogeneity"
        if len(slp_list) == len(params_dict['inhomo']):
            for i,n in enumerate(params_dict['inhomo']):
                print "slp {:.0f} hz, inhomo {}".format(slp_list[i], n)
        else:
            print '!!!please specify inhomogeneity for each slp!!!'
            sys.exit(0)

        # Read any number of blank lines that are present until you fall on a "+"
        # or reach end of file
        while 1:
            line = f.readline()
            # If you fall on end of file, note it down
            if len(line) == 0:
                eof_file = 1
                break
            if line[0] == "+":
                break

        # append read info to global data sets variable
        data_sets.append(params_dict)

        # if end of file was reached earlier, quit master loop
        if eof_file == 1:
            break
        print 
          
    f.close()
    
    # That's it. Now we need to do the fitting
#     return params_dict

def simulate_cest_offset(expt_var, offset_value, *fit_args):
    ''' Simulate an acutal CEST profile given exchange parameters '''
    global params
    global print_counter

    # THis is required because if we explicitly pass arguments as a list to simualte_cest as an argument, it is converted to a tuple
    # however, when curve_fit does it, it internall passes individual arguments
    if len(fit_args) == 1:
        fit_args = fit_args[0]

    # Unpack the parameters and expt arrays
    # If a global CEST data set is being handled
    if len(expt_var) != 3:
        offsets = params['offset']
        slps = params['slp']
        trelax = params['trelax']
    # If only a trendline for a global data set is required
    else:
        [offsets, slps, trelax] = expt_var

    alignmag = params['AlignMag']
    lf = params['lf']
    resn = params['resn']
    inhomo = params['inhomo']

 

    # Unpact the fit params arrays
    # key point is that all arguments need not be used in the fitting at all
    # especially depending on whether the relaxation params are equal or not
    # then it is likely that that parameter initial and final values are the same - check this!

    if params['relax_params'] == '0000':
        # All constants are equal
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2] = fit_args
        r1b = r1
        r1c = r1
        r2b = r2
        r2c = r2
    elif params['relax_params'] == '1000':
        # unequal r2b
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r2b] = fit_args
        r1b = r1
        r1c = r1
        r2c = r2
    elif params['relax_params'] == '0100':
        # unequal r2c
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r2c] = fit_args
        r1b = r1
        r1c = r1
        r2b = r2
    elif params['relax_params'] == '0010':
        # unequal r1b
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1b] = fit_args
        r1c = r1
        r2b = r2
        r2c = r2
    elif params['relax_params'] == '0001':
        # unequal r1c
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1c] = fit_args
        r1b = r1
        r2b = r2
        r2c = r2
    elif params['relax_params'] == '1100':
        # unequal r2b + unequal r2c
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r2b, r2c] = fit_args
        r1b = r1
        r1c = r1
    elif params['relax_params'] == '1010':
        # unequal r2b + unequal r1b
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1b, r2b] = fit_args
        r1c = r1
        r2c = r2
    elif params['relax_params'] == '1001':
        # unequal r2b + unequal r1c
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1c, r2b] = fit_args
        r1b = r1
        r2c = r2
    elif params['relax_params'] == '0110':
        # unequal r2c + unequal r1b
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1b, r2c] = fit_args
        r1c = r1
        r2b = r2
    elif params['relax_params'] == '0101':
        # unequal r2c + unequal r1c
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1c, r2c] = fit_args
        r1b = r1
        r2b = r2
    elif params['relax_params'] == '0011':
        # unequal r1b + unequal r1c
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1b, r1c] = fit_args
        r2b = r2
        r2c = r2
    elif params['relax_params'] == '0111':
        # unequal r1b + unequal r1c + unequal r2c
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1b, r1c, r2c] = fit_args
        r2b = r2
    elif params['relax_params'] == '1011':
        # unequal r1b + unequal r1c + unequal r2b
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1b, r2b, r1c] = fit_args
        r2c = r2
    elif params['relax_params'] == '1101':
        # unequal r1c + unequal r2b + unequal r2c
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r2b, r1c, r2c] = fit_args
        r1b = r1
    elif params['relax_params'] == '1110':
        # unequal r2b + unequal r2c + unequal r1b
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1b, r2b, r2c] = fit_args
        r1c = r1
    elif params['relax_params'] == '1111':
        # unequal r2b + unequal r2c + unequal r1b + unequal r1c
        [pb, pc, dwb, dwc, kexab, kexac, kexbc, r1, r2, r1b, r2b, r1c, r2c] = fit_args

    # Now define auto alignment
    if params['AlignMag'] == 'AUTO':
        # Compute exchange ratio based on kex/dw for state B
        if dwb != 0.0:
            exch_ratio = kexab / dwb
        else:
            exch_ratio = 0.1

        if exch_ratio <= 1:
            alignmag = 'GS'
        else:
            alignmag = 'AVG' 
 
    # Define derived quantities
    pa = 1 - pb - pc
    k12 = kexab * pb / (pb + pa)
    k21 = kexab * pa / (pa + pb)
    k13 = kexac * pc / (pc + pa)
    k31 = kexac * pa / (pa + pc)
    if (pb + pc) <= 0.0:
        k23 = 0.0
        k32 = 0.0
    else:
        k23 = kexbc * pc / (pb + pc)
        k32 = kexbc * pb / (pb + pc)

    # compute the observed peak positions
    # as kinetic params change during fitting, this has to be done afresh for each fitting
    if params['ls'] == 'Y': 
        #print "LS"
        [obs_peakpos, obs_peakheight] = ls_3state(pa, pb, pc, k12, k21, k13, k31, k23, k32, 0.0, dwb, dwc, r2, r2b, r2c, lf, resn) 
    else:
        #print "NO LS SIMULATION"
        # If observed peak is GS peak
        # offset_value represents any couplings that may be present
        if alignmag == "GS":
            obs_peakpos = 0.0 + offset_value
        # If observed peak is AVG peak
        else:
            obs_peakpos = (0.0*pa) + (dwb*pb) + (dwc*pc) + offset_value
 
    # using observed peak position, compute the offsets for each spin
    delta_a = (-1.*offsets) + (0.0-obs_peakpos) 
    delta_b = (-1.*offsets) + (dwb-obs_peakpos)
    delta_c = (-1.*offsets) + (dwc-obs_peakpos)

    # get the ordered list of slp in order to input different inhomo for different slp
    slp_set = list(set(params['slp']))
    slp_set.sort()
    
    # Define starting magnetization
    # depending on equilibration flag
    if params['eqbn'] == 'N':
       # (unit_vector, GS_x, GS_y, GS_z, ES1_x, ES1_y, ES1_z, ES2_x, ES2_y, ES2_z
       #print "*** NO EQUILIBRATIONi ***"
        M0_plusx = array([1.0, 0.0, 0.0, pa, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        M0_minusx = array([1.0, 0.0, 0.0, -1.0*pa, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif params['eqbn'] == 'Y': 
       # (unit_vector, GS_x, GS_y, GS_z, ES1_x, ES1_y, ES1_z, ES2_x, ES2_y, ES2_z
       #print "*** COMPLETE EQUILIBRATIONi ***"
        M0_plusx = array([1.0, 0.0, 0.0, pa, 0.0, 0.0, pb, 0.0, 0.0, pc])
        M0_minusx = array([1.0, 0.0, 0.0, -1.0*pa, 0.0, 0.0, -1.0*pb, 0.0, 0.0, -1.0*pc])

    # variable to save weights_slp and initial intensities for different slp
    weight_slp_list = []
    intensities_initial_list = []

    # for each slp calculate weight_slp and initial intensities
    for slp_ind in slp_set:
        indx_inhom = slp_set.index(slp_ind)
        inhom = inhomo[indx_inhom]

        if inhom == 0.0:
            weights_slps = [1.0]
            params['number_inhomo_slps'] = 1
        else:
            w1_dummy = slp_ind
            #params['number_inhomo_slps'] = 120
            #print "slp is {}, inhomo is {}".format(params['slp'][dummy]/(2*math.pi), inhom)
            # incorporate inhomogeneity
            sigma_slp = w1_dummy * inhom
            slps_net = linspace(w1_dummy - (2 * sigma_slp), w1_dummy + (2 * sigma_slp), params['number_inhomo_slps'])
            weights_slps = (exp((-1*square(slps_net - w1_dummy))/(2*sigma_slp*sigma_slp))/math.sqrt(2*math.pi*sigma_slp*sigma_slp)) 
        weight_slp_list.append(weights_slps)

        # This is where the fun happens
        # depending on what combination of relaxation constants are kept constant
        # Handle all conventional cases we would encounter
        # throw an error if a weird case happens
        # For this case, the SLP, offsets do not matter as time point is set to 0
        BM_mat = MatrixBM_3state(k12, k21, k13, k31, k23, k32, delta_a[0], delta_b[0], delta_c[0], slps[0], r1, r2, r1b, r2b, r1c, r2c, pa, pb, pc, inh=inhom)

        # Now, simulate!
        # Initial time point
        # Take into account phase cycling
        # This is where spin-lock inhomogeniety needs to be taken into account
        time_point = 0.0
        intensities_initial = zeros(10) 

        # Define variable for the magnetization of each state at each slp, offset value
        intensities_net = zeros((slps.shape[0]))

        # Now, simulate!
        # Initial time point
        # Take into account phase cycling
        # This is where spin-lock inhomogeniety needs to be taken into account
        for dummy_inhomo in range(params['number_inhomo_slps']):
            matrix_expo_result = matrix_exponential(BM_mat[:,:,dummy_inhomo], time_point)
            intensities_initial = intensities_initial + ((dot(matrix_expo_result, M0_plusx)-dot(matrix_expo_result, M0_minusx))*weights_slps[dummy_inhomo])
        intensities_initial_list.append(intensities_initial)


    # For each slp, offset combination, compute the normalized intensity
    # and compare to expt value 
    for dummy in range(slps.shape[0]):
    # generate the weights for each slp here
    # Key point is that each spin set experiences only one value of the inhomogeniety
    # which is same across all spin-locks
    # so any spin-lock power can be used for generating gaussian weights
    
        # get the inhomogeneity, weight_slp and initial intensity for each slp
        indx_inhom = slp_set.index(slps[dummy])
        inhom = inhomo[indx_inhom]
        weights_slps = weight_slp_list[indx_inhom]
        intensities_initial = intensities_initial_list[indx_inhom]

        #if inhom == 0.0:
        #    params['number_inhomo_slps'] = 1
        #else:
        #    params['number_inhomo_slps'] = 120

        # print weights_slps
        # Get the B-M matrix for this particular SLP and Offset combination
        # Again, take into account relax params
        BM_mat = MatrixBM_3state(k12, k21, k13, k31, k23, k32, delta_a[dummy], delta_b[dummy], delta_c[dummy], slps[dummy], r1, r2, r1b, r2b, r1c, r2c, pa, pb, pc, inh=inhom)

        # Now, simulate at time point of interest
        # again handle inhomogeniety here
        time_point = trelax[dummy]
        intensities = zeros(10)
        for dummy_inhomo in range(params['number_inhomo_slps']):
            matrix_expo_result = matrix_exponential(BM_mat[:, :, dummy_inhomo], time_point)
            intensities = intensities + ((dot(matrix_expo_result, M0_plusx)-dot(matrix_expo_result, M0_minusx))*weights_slps[dummy_inhomo]) 

        # Now, normalize this intensity
        # Key point is that normalization will be different if exchange regime is different
        # Put alternatively, super slow exchange ==> GS_mag(t)/GS_mag(0)
        # At super fast exchange ==> (GS_mag(t) + ES_mag(t)) / (GS_mag(0) + ES_mag(0))
        # Things could be more complicated wherein GS <-> ES1 is fast while GS <-> ES2 is slow
        # One can accordingly define more complicated normalization schemes
        # Here it would be  ==> (GS_mag(t) + ES1_mag(t)) / (GS_mag(0) + ES1_mag(0))
        #intensities_normalized = divide(intensities, intensities_initial)
        if alignmag == 'GS':
            #print "GS Alignment"
            intensities_net[dummy] = intensities[3] / intensities_initial[3]
        elif alignmag == 'AVG':
            #print "AVG Alignment"
            denominator = (intensities_initial[3] + intensities_initial[6] + intensities_initial[9])
            intensities_net[dummy] = (intensities[3]+intensities[6]+intensities[9]) / (denominator)
 
    return intensities_net    

def simulate_cest(expt_var, *fit_args):
    ''' Treat splittings for spins '''
    global params
    # Generate a pair of profiles for spins assuming wobs-(pi*J) & wobs+(pi*J)
    if params['J'] != 0.0:
        #print params['J'] * -0.5, params['J']*0.5
        intensity1 = simulate_cest_offset(expt_var, params['J']*-0.5, *fit_args)
        intensity2 = simulate_cest_offset(expt_var, params['J']*0.5, *fit_args)
        return (intensity1+intensity2)*0.5
    else:
        return simulate_cest_offset(expt_var, 0.0, *fit_args)

def plot_profiles(fit_answer, rchi2):
    ''' Given the parameters, fit answer, plot the CEST profile '''
    global params
    global output_foldername

    number_slps = unique(params['slp'])
    colors_plot = ['k', 'r', 'b', 'g', 'cyan', 'magenta', 'brown', 'yellow', 'teal', 'lightgreen']

    # Plot profile
    fig, ax = plt.subplots(figsize = (12,10))
    plt.title("$r\chi^{2}=$" + '%4.3f'%rchi2)
    counter = 0
    for dummy_slp in number_slps:
        # Get SLP and offset corresponding to offset of interest
        slp_positions = where(params['slp'] == dummy_slp)
        offset_vals = params['offset'][slp_positions]

        # Generate the corresponding trendline
        offset_vals_trendline = linspace(amin(offset_vals), amax(offset_vals), 400)
        slp_vals_trendline = ones(offset_vals_trendline.shape) * dummy_slp
        trelax_vals_trendline = ones(offset_vals_trendline.shape) * params['trelax'][slp_positions][0]
        intensity_trendline = simulate_cest([offset_vals_trendline, slp_vals_trendline, trelax_vals_trendline], *unumpy.nominal_values(fit_answer)) 

        # Plot experimental data
        plt.errorbar((offset_vals/(2*math.pi*params['lf'])), params['norm_intensity'][slp_positions], yerr=params['norm_intensity_error'][slp_positions], color = colors_plot[counter], label='%4.0f'%float(dummy_slp/(2*math.pi)), linewidth=0, marker='o', markersize=10)
        # Plot fit vals
        #plt.plot((params['offset'][where(params['slp'] == dummy_slp)]/(2*math.pi*params['lf'])), intensity_answer[where(params['slp'] == dummy_slp)], color = colors_plot[counter], linewidth=3)
        # plot trend lines
        plt.errorbar((offset_vals_trendline/(2*math.pi*params['lf'])), intensity_trendline, color = colors_plot[counter], linewidth=5)
 
        counter = counter + 1
    plt.xlim([((amin(params['offset'])-(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf'])), ((amax(params['offset'])+(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf']))])
    #plt.ylim([-0.1, 1])
    plt.ylabel('$I/I_{0}$')
    plt.xlabel('$\Omega(ppm)$')
    # plt.legend(loc=4)
    l=plt.legend(loc = 4, fontsize = 30, frameon=False,
           handletextpad=-2.0, handlelength=0, markerscale=0,
           title = '$\omega$' + ' $2\pi$' + '$^{-1}$'+ ' (Hz)',
           ncol=3, columnspacing=1.8)
    l._legend_box.align = "right"

    for item in l.legendHandles:
        item.set_visible(False)
    for handle, text in zip(l.legendHandles, l.get_texts()):
        text.set_color(handle.get_color()[0])
    plt.tight_layout()
    #plt.show()
    #print str(output_foldername) + "/fit-" + str(params['Name'][:params['Name'].index('.csv')]) + ".pdf"
    fig.savefig(str(output_foldername) + "/fit-" + str(params['Name'][:params['Name'].index('.csv')]) + ".pdf", dpi = 300)
    plt.close(fig)

def mc_error_sets(intensity_answer, initial_vals, lbounds, ubounds):
    # number of mc iterations
    mc_iter = data_sets[0]['mc_iter']
    print "MC iterations = ", mc_iter

    # variables for storing noise corrupted intensities
    intensity_corrupted = zeros((mc_iter, intensity_answer.shape[0]))

    # populate matrix to store variables in each mc iteration
    #popt_matrix = zeros((mc_iter, number_paropt))

    # populate the intensities to be corrupted
    # Loop over each data point
    for dummy in range(intensity_answer.shape[0]):
        intensity_corrupted[:, dummy] = normal(loc=intensity_master[dummy], scale=intensity_master_error[dummy], size=mc_iter)

    # fitting mc iterations
    #for dummy in range(mc_iter):
    #    print "iteration ", dummy+1
    # Get the optimum set of parameters with minimum least squares
    # Parallelize this part
    def fit_iter(iter_number):    
        [popt, pcov] = curve_fit(simulate_cest_sets, 'test', intensity_corrupted[iter_number,:], p0=initial_vals, sigma=intensity_master_error, bounds=(lbounds, ubounds), method='trf')
        return popt
    popt_matrix = parmap(fit_iter, range(mc_iter)) 
    popt_matrix = array(popt_matrix)
    #print popt_matrix
    #print type(popt_matrix)

    # print mean and standard deviation of parameters
    #print "**", mean(popt_matrix, axis=0), std(popt_matrix, axis=0)
    #mean_vars = mean(popt_matrix, axis=0)
    std_vars = std(popt_matrix, axis=0)

    #for dummy in range(number_paropt):
    #    if str(params_fit_dict[params['relax_params']][dummy]) in params['free_vars']:
    #        print 'mc', params_fit_dict[params['relax_params']][dummy], (mean_vars[dummy]), '+/-', (std_vars[dummy]) 
    #    elif str(params_fit_dict[params['relax_params']][dummy]) in params['fixed_vars']:
    #        print 'mc', params_fit_dict[params['relax_params']][dummy], (mean_vars[dummy]), '+/-', '0.0' 
    #    else:
    #        pass 
    return std_vars 

def mc_error(intensity_answer, par_opt):
    global params

    # number of mc iterations
    mc_iter = params['mc_iter']

    # variables for storing noise corrupted intensities
    intensity_corrupted = zeros((mc_iter, intensity_answer.shape[0]))

    # populate matrix to store variables in each mc iteration
    #popt_matrix = zeros((mc_iter, number_paropt))

    # populate the intensities to be corrupted
    # Loop over each data point
    for dummy in range(intensity_answer.shape[0]):
        intensity_corrupted[:, dummy] = normal(loc=params['norm_intensity'][dummy], scale=params['norm_intensity_error'][dummy], size=mc_iter)

    # now, for each mc iteration, fit back the noise corrupted intensities
    # intialize variable bounds
    initial_vals = [params[ele][0] for ele in params_fit_dict[params['relax_params']]]
    lower_bounds = [params[ele][1] for ele in params_fit_dict[params['relax_params']]]
    upper_bounds = [params[ele][2] for ele in params_fit_dict[params['relax_params']]]
 
    # fitting mc iterations
    #for dummy in range(mc_iter):
    #    print "iteration ", dummy+1
    #    # Get the optimum set of parameters with minimum least squares
    #    [popt, pcov] = curve_fit(simulate_cest, 'test', intensity_corrupted[dummy,:], p0=par_opt, sigma=params['norm_intensity_error'], bounds=(lower_bounds, upper_bounds), method='trf') 
    #    popt_matrix[dummy, :] = popt

    def fit_iter(iter_number):    
        [popt, pcov] = curve_fit(simulate_cest, 'test', intensity_corrupted[iter_number,:], p0=par_opt, sigma=params['norm_intensity_error'], bounds=(lower_bounds, upper_bounds), method='trf')
        return popt
    popt_matrix = parmap(fit_iter, range(mc_iter)) 
    popt_matrix = array(popt_matrix)
  
    # print mean and standard deviation of parameters
    #print "**", mean(popt_matrix, axis=0), std(popt_matrix, axis=0)
    #mean_vars = mean(popt_matrix, axis=0)
    std_vars = std(popt_matrix, axis=0)

    #for dummy in range(number_paropt):
    #    if str(params_fit_dict[params['relax_params']][dummy]) in params['free_vars']:
    #        print 'mc', params_fit_dict[params['relax_params']][dummy], (mean_vars[dummy]), '+/-', (std_vars[dummy]) 
    #    elif str(params_fit_dict[params['relax_params']][dummy]) in params['fixed_vars']:
    #        print 'mc', params_fit_dict[params['relax_params']][dummy], (mean_vars[dummy]), '+/-', '0.0' 
    #    else:
    #        pass 
    return std_vars 


def fit_cest():
    global params
    initial_vals = [params[ele][0] for ele in params_fit_dict[params['relax_params']]]
    lower_bounds = [params[ele][1] for ele in params_fit_dict[params['relax_params']]]
    upper_bounds = [params[ele][2] for ele in params_fit_dict[params['relax_params']]]

    # Get the optimum set of parameters with minimum least squares
    [popt, pcov] = curve_fit(simulate_cest, 'test', params['norm_intensity'], p0=initial_vals, sigma=params['norm_intensity_error'], bounds=(lower_bounds, upper_bounds), method='trf') 

    # Call simulate cest once more to get simulated optimum profile
    intensity_answer = simulate_cest('test', *popt) 

    # Estimate the error
    if params['err_mode'] == "STD":
        # Estimate the standard error
        #print "STANDARD ERROR"
        perr = sqrt(diag(pcov))
    elif params['err_mode'] == "MC":
        # Get the mc error
        #print "MC ERROR"
        perr = mc_error(intensity_answer, popt)
 
    # Compute the chi^2
    chi2 = sum(square((params['norm_intensity']-intensity_answer)/params['norm_intensity_error']))
    #print "**", params['norm_intensity'].shape[0], params['number_floating']
    rchi2 = chi2 / (params['norm_intensity'].shape[0] - params['number_floating'])
    print "CHI2", chi2, "rchi2", rchi2, len(initial_vals), params['number_floating']

    # Now, we have fit values and errors
    # combine into an uncertainties matrix
    fit_answer = unumpy.uarray(popt, perr)

    return [fit_answer, rchi2, intensity_answer]

def print_answer(fit_answer, intensity_answer, data_file, params_file, rchi2):
    ''' Write out optimal answer to a directory '''
    global output_foldername
    global params_fit
    global params

    # Change the units of dW fit params to be in ppm units
    fit_answer[2] = fit_answer[2] / (params['lf'] * 2 * math.pi)
    fit_answer[3] = fit_answer[3] / (params['lf'] * 2 * math.pi)
 
    # Extract back values, error
    fit_answer_nominal_values = unumpy.nominal_values(fit_answer)
    fit_answer_std_devs = unumpy.std_devs(fit_answer)

    print "Optimal parameters are "
    for dummy in range(fit_answer.shape[0]):
        if str(params_fit_dict[params['relax_params']][dummy]) in params['free_vars']:
            #print params_fit_dict[params['relax_params']][dummy], '%4.3f'%(fit_answer_nominal_values[dummy]), '+/-', '%4.3f'%(fit_answer_std_devs[dummy]) 
            print params_fit_dict[params['relax_params']][dummy], '%4.3f'%(fit_answer_nominal_values[dummy]), '+/-', '%4.3f'%(fit_answer_std_devs[dummy]) 
        elif str(params_fit_dict[params['relax_params']][dummy]) in params['fixed_vars']:
            print params_fit_dict[params['relax_params']][dummy], '%4.3f'%(fit_answer_nominal_values[dummy]), '+/-', '0.0' 
        else:
            pass 

    # Copy over input files into new directory
    os.system('cp ' + str(data_file) + ' ' + str(output_foldername) + "/copy-" + str(data_file))
    #os.system('cp ' + str(params_file) + ' ' + str(output_foldername) + "/copy-" + str(params_file))
 
    # Read the input file and append the fitted intensity to it
    bf_input = pd.read_csv(data_file)
    bf_input.insert(len(list(bf_input)), 'fit_norm_intensity', intensity_answer)
    bf_input.to_csv(str(output_foldername) + "/fit-" + str(data_file))

    # Write out output parameters to directory
    f_answer = open(str(output_foldername) + "/fitparms-" + str(params['Name'][:params['Name'].index('.csv')]) + '.csv', "w")
    f_answer.write('param,fitval,fiterror\n')
    for dummy in range(fit_answer.shape[0]):
        if str(params_fit_dict[params['relax_params']][dummy]) in params['free_vars']:
            #f_answer.write(str(params_fit_dict[params['relax_params']][dummy]) + "," + '%6.5f'%(fit_answer_nominal_values[dummy]) + "," + '%6.5f'%(fit_answer_std_devs[dummy]) + "\n")
            f_answer.write(str(params_fit_dict[params['relax_params']][dummy]) + "," + str('%6.5f'%fit_answer_nominal_values[dummy]) + "," + str('%6.5f'%fit_answer_std_devs[dummy]) + "\n")
        elif str(params_fit_dict[params['relax_params']][dummy]) in params['fixed_vars']:
            #f_answer.write(str(params_fit_dict[params['relax_params']][dummy]) + "," + '%6.5f'%(fit_answer_nominal_values[dummy]) + "," + '%6.5f'%(0.0) + "\n")
            f_answer.write(str(params_fit_dict[params['relax_params']][dummy]) + "," + str('%6.5f'%fit_answer_nominal_values[dummy]) + "," + '%6.5f'%(0.0) + "\n")
        else:
            pass
    f_answer.write("rchi2," + '%4.3f'%rchi2 + ",0.0")
    f_answer.close()

def generate_vals_sets(fit_args):
    ''' given a flattened list of initial values, generates list of lists
    of initial values for each data set '''
    global net_vars
    global data_sets

    shared_vars_lims = []
    initial_vals_sets = []
    initial_vals_sets.append([fit_args[dummy] for dummy in range(len(net_vars[0]))])
    shared_vars_lims.extend([(net_vars[0][dummy], fit_args[dummy]) for dummy in range(len(net_vars[0])) if net_vars[0][dummy] in data_sets[0]['shared_vars']])
    #print shared_vars_lims 
    #print initial_vals_sets
    shared_vars_lims = dict(shared_vars_lims)

    # Unpack other data sets
    for dummy_new in range(1, len(data_sets)):
        initial_vals_set = []

        # Loop over each data sets variables
        for ele in params_fit_dict[data_sets[dummy_new]['relax_params']]:
            # If variable is shared, and has appeared before
            if (ele in shared_vars_lims.keys()) and (ele in data_sets[dummy_new]['shared_vars']):
                #print ele, shared_vars_lims[ele][0], shared_vars_lims[ele][1], shared_vars_lims[ele][2]
                initial_vals_set.append(shared_vars_lims[ele])

            # New variable in given data set
            else:
                dummy = dummy + 1
                #print ele, initial_vals[dummy], lower_bounds[dummy], upper_bounds[dummy]
                initial_vals_set.append(fit_args[dummy])
                # If variable is shared, add its limits to variable monitoring the same
                if (ele in data_sets[dummy_new]['shared_vars']) and (ele not in shared_vars_lims.keys()):
                    #shared_vars_lims[ele] = [initial_vals[dummy], lower_bounds[dummy], upper_bounds[dummy]]
                    shared_vars_lims[ele] = fit_args[dummy]
        #print "share", shared_vars_lims
        #print  
        
        # add list of initial values to master variable storing values for all data sets 
        initial_vals_sets.append(initial_vals_set)
    return initial_vals_sets

def simulate_cest_sets(expt_var, *fit_args):
    ''' simulate CEST data for many data sets '''
    global net_vars
    global params

    # Unpack the fit_args variable to get variables for each data set, 
    # that can be used to simulate its corresponding CEST profile
    # Store limits of variables that have been shared
    #for dummy in range(len(net_vars[0])):
    #    print net_vars[0][dummy], initial_vals[dummy], lower_bounds[dummy], upper_bounds[dummy]
    #    if net_vars[0][dummy] in data_sets[0]['shared_vars']:
    #        shared_vars_lims[net_vars[0][dummy]] = [initial_vals[dummy], lower_bounds[dummy], upper_bounds[dummy]]
    initial_vals_sets = generate_vals_sets(fit_args)
 
    #print "****", initial_vals_sets

    # Now, we can pass initial_vals_sets to simulate_cest to get back corresponding intensity profile
    intensity_computed = []
    for dummy in range(len(data_sets)):
        params = data_sets[dummy]
        #intensity_computed.extend(simulate_cest([params['offset'], params['slp'], params['trelax']], tuple(initial_vals_sets[dummy])))
        intensity_computed.extend(simulate_cest('test', tuple(initial_vals_sets[dummy])))
    return array(intensity_computed)
 
def shared_fit():
    ''' Carry out shared fitting '''
    global data_sets
    global net_vars
    global intensity_master
    global intensity_master_error
    global input_filename
    global params

    # Strategy here is to pack and unpack
    # given data_sets, create a concatenated variable set based on which variables are shared
    # key point is that a given variable, say pB/pC can only be shared once
    
    # Include all variables in 1st data set in master variable set
    initial_vals = [data_sets[0][ele][0] for ele in params_fit_dict[data_sets[0]['relax_params']]]
    lower_bounds = [data_sets[0][ele][1] for ele in params_fit_dict[data_sets[0]['relax_params']]]
    upper_bounds = [data_sets[0][ele][2] for ele in params_fit_dict[data_sets[0]['relax_params']]]
    #print params_fit_dict[data_sets[0]['relax_params']]

    # Store variables used in each data set
    net_vars.append(params_fit_dict[data_sets[0]['relax_params']])

    # Store shared variables used
    net_shared_params = []
    net_shared_params.extend(data_sets[0]['shared_vars'])

    # Now, loop over other data sets
    # exclude everything that have been shared till now
    for dummy in range(1, len(data_sets)):
        variables_included = []
        for ele in params_fit_dict[data_sets[dummy]['relax_params']]: 
            # A variable that has been shared would appear in shared_vars set of the particular data set
            # and also in net list of variables that are shared
            if (ele in data_sets[dummy]['shared_vars']) and (ele in net_shared_params):
                pass
            # If a variable has not been included in a set
            # add it to list of variables in fit. 
            else:
                variables_included.append(ele)
                # Also add to net list of shared variables if it is shared
                if ele in data_sets[dummy]['shared_vars']:
                    net_shared_params.append(ele)

        # update master variable list based on variables that are to be included
        initial_vals.extend([data_sets[dummy][ele][0] for ele in variables_included])
        lower_bounds.extend([data_sets[dummy][ele][1] for ele in variables_included])
        upper_bounds.extend([data_sets[dummy][ele][2] for ele in variables_included])
        #print "*", variables_included       
        net_vars.append(variables_included)

    print "Shared parameters ", net_shared_params
    #print len(initial_vals), len(lower_bounds), len(upper_bounds)
    #print net_vars

    # Now, we have a flattened variable list 
    # We need to create a flattened intensity list, which should be the ideal output of the fitting function
    intensity_master = []
    intensity_master_error = []
    for dummy in range(len(data_sets)):
        intensity_master.extend(data_sets[dummy]['norm_intensity'])
        intensity_master_error.extend(data_sets[dummy]['norm_intensity_error'])
    intensity_master = array(intensity_master)
    intensity_master_error = array(intensity_master_error)
   
    # We have everything for calling the shared fitting function
    [popt, pcov] = curve_fit(simulate_cest_sets, 'test', intensity_master, p0=initial_vals, sigma=intensity_master_error, bounds=(lower_bounds, upper_bounds), method='trf')

    # Get the intensities corresponding to the fit answer
    intensity_answer = simulate_cest_sets('test', *popt) 

    # Estimate the error
    if data_sets[0]['err_mode'] == "STD":
        # Estimate the standard error
        #print "STANDARD ERROR"
        perr = sqrt(diag(pcov))
    elif data_sets[0]['err_mode'] == "MC":
        # Get the mc error
        #print "MC ERROR"
        perr = mc_error_sets(intensity_answer, popt, lower_bounds, upper_bounds)
 
    # Compute the chi^2
    chi2 = sum(square((intensity_master-intensity_answer)/intensity_master_error))
    number_floating_net = sum([data_sets[dummy]['number_floating'] for dummy in range(len(data_sets))]) - len(net_shared_params)
    rchi2 = chi2 / (intensity_master.shape[0] - number_floating_net)
    print "CHI2", chi2, "rchi2", rchi2, number_floating_net, len(net_shared_params)

    # get list of lists of each variable 
    unpacked_popt = generate_vals_sets(popt)
    unpacked_perr = generate_vals_sets(perr)
    unpacked_popt = [array(ele) for ele in unpacked_popt]
    unpacked_perr = [array(ele) for ele in unpacked_perr]
    unpacked_popt_perr = [unumpy.uarray(unpacked_popt[dummy], unpacked_perr[dummy]) for dummy in range(len(unpacked_popt))]
     
    #print "look" 
    #print unpacked_popt_perr

    # Change the units of fitted dW in each data set
    #for dummy in range(len(data_sets)):

    # Now, we have fit values and errors
    # combine into an uncertainties matrix
    #fit_answer = unumpy.uarray(popt, perr)

    # Copy over input files into new directory
    #os.system('cp ' + str(input_filename) + ' ' + str(output_foldername) + "/copy-" + str(input_filename))

    # Print out the fit parameters and write them to a file
    for dummy in range(len(data_sets)):
        # Compute fitted curve for each data set
        # append it to a new column into input data file
        params = data_sets[dummy]
        bf_input = pd.read_csv(data_sets[dummy]['Name'])
        bf_input.insert(len(list(bf_input)), 'fit_norm_intensity', simulate_cest('test', tuple(unpacked_popt[dummy])))
        bf_input.to_csv(str(output_foldername) + "/fit-" + str(data_sets[dummy]['Name']))

        # Also, perform the plotting
        plot_profiles(unpacked_popt_perr[dummy], rchi2)

        # Change units of dW now
        unpacked_popt_perr[dummy][2] = unpacked_popt_perr[dummy][2] / (2 * math.pi * data_sets[dummy]['lf'])
        unpacked_popt_perr[dummy][3] = unpacked_popt_perr[dummy][3] / (2 * math.pi * data_sets[dummy]['lf'])

        # Create copy of data csvs
        os.system('cp ' + str(data_sets[dummy]['Name']) + ' ' + str(output_foldername) + "/copy-" + str(data_sets[dummy]['Name']))

        # Write out and print fitted parameters for each data set
        fout = open(str(output_foldername) + "/" + "fitparams-" + str(data_sets[dummy]['Name'][:data_sets[dummy]['Name'].index('.csv')]) + '.csv', 'w')
        fout.write('param,fitval,fiterror\n')
        for dummy_int in range(len(params_fit_dict[data_sets[dummy]['relax_params']])):
            if params_fit_dict[data_sets[dummy]['relax_params']][dummy_int] in data_sets[dummy]['free_vars']:
                print params_fit_dict[data_sets[dummy]['relax_params']][dummy_int], '%6.5f'%(unumpy.nominal_values(unpacked_popt_perr[dummy][dummy_int])), "+-", '%6.5f'%(unumpy.std_devs(unpacked_popt_perr[dummy][dummy_int]))
                fout.write(str(params_fit_dict[data_sets[dummy]['relax_params']][dummy_int]) + ',' + str('%6.5f'%(unumpy.nominal_values(unpacked_popt_perr[dummy][dummy_int]))) + "," + str('%6.5f'%(unumpy.std_devs(unpacked_popt_perr[dummy][dummy_int]))) + '\n')
            elif params_fit_dict[data_sets[dummy]['relax_params']][dummy_int] in data_sets[dummy]['fixed_vars']:
                print params_fit_dict[data_sets[dummy]['relax_params']][dummy_int], '%6.5f'%(unumpy.nominal_values(unpacked_popt_perr[dummy][dummy_int])), "+-", '%6.5f'%(0.0)
                fout.write(str(params_fit_dict[data_sets[dummy]['relax_params']][dummy_int]) + ',' + str('%6.5f'%(unumpy.nominal_values(unpacked_popt_perr[dummy][dummy_int]))) + "," + str('%6.5f'%(0.0)) + '\n')
            else:
                pass
        fout.write("rchi2," + '%6.5f'%rchi2 + ",0.0")
        fout.close()
        print
   
    return [unpacked_popt_perr, rchi2, intensity_answer]
 

# parse the input parameters 
input_filename = str(sys.argv[1])
output_foldername = str(sys.argv[2])
start_time = time.time()
input_parser(input_filename)

# Create new output directory, delete old one
os.system("rm -r " + str(output_foldername) + "/")
os.system('mkdir ' + str(output_foldername))

# Copy over input files into new directory
os.system('cp ' + str(input_filename) + ' ' + str(output_foldername) + "/copy-" + str(input_filename))

# individual fit
if len(data_sets) == 1:
    params = data_sets[0]
    [fit_answer, rchi2, intensity_answer] = fit_cest()
    plot_profiles(fit_answer, rchi2)
    print_answer(fit_answer, intensity_answer, params['Name'], input_filename, rchi2)
# Some form of global fitting
else:
    shared_fit() 

end_time = time.time()
net_time = end_time - start_time
print "END TIME = ", net_time
