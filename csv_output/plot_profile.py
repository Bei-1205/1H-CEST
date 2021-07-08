#!/usr/bin/python
import pandas as pd
import numpy as np
import sys
from nmrglue import proc_base
from nmrglue import analysis
import math
import matplotlib.pyplot as plt
import sys
lf = 599.659943

def plot_profiles(filename, op_filename):
    ''' Given the parameters, intensity from simulation, plot the CEST profile '''
    bf = pd.read_csv(filename)
    number_slps = np.unique(bf['slp(hz)'])
    colors_plot = ['k', 'r', 'b', 'g', 'cyan', 'magenta', 'brown', 'yellow', 'teal', 'lightgreen']
    # Plot profile

    plt.figure(1)
    counter = 0
    for dummy_slp in number_slps:
        bf_new = bf.loc[bf['slp(hz)'] == dummy_slp]
        #plt.errorbar(bf_new['offset(hz)']/lf, bf_new['norm_intensity'], yerr=bf_new['norm_intensity_error'], color = colors_plot[counter], label='%4.2f'%float(dummy_slp), linewidth=3, fmt='o')
        plt.errorbar(bf_new['offset(hz)']/lf, bf_new['norm_intensity'], yerr=bf_new['norm_intensity_error'], color = colors_plot[counter], label='%4.2f'%float(dummy_slp), linewidth=3, fmt='o')
        counter = counter + 1
    #plt.xlim([((np.amin(params['offsets'])-(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf']))+138.4, ((np.amax(params['offsets'])+(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf']))+138.4])
    #plt.xlim([((np.amin(params['offsets'])-(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf'])), ((np.amax(params['offsets'])+(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf']))])
    plt.ylim([-0.05, 1])
    # plt.xlim(5, 20)
    plt.xlabel('$\Omega (ppm)$') 
    plt.ylabel('$I/I_{0}$')
    #plt.xticks([-12, -8, -4, 0, 4, 8, 12])
    plt.legend()
    #plt.ylim([0.0, 0.35])
    plt.savefig(op_filename + "-intensity.pdf")
    plt.show()

    #plt.savefig(op_filename + "-intensity.pdf")

    plt.figure(2)
    counter = 0
    for dummy_slp in number_slps:
        bf_new = bf.loc[bf['slp(hz)'] == dummy_slp]
        #plt.errorbar(bf_new['offset(hz)']/lf, bf_new['norm_volume'], yerr=bf_new['norm_volume_error'], color = colors_plot[counter], label='%4.2f'%float(dummy_slp), linewidth=3, fmt='o')
        plt.errorbar(bf_new['offset(hz)']/lf, bf_new['norm_volume'], yerr=bf_new['norm_volume_error'], color = colors_plot[counter], label='%4.2f'%float(dummy_slp), linewidth=3, fmt='o')
        counter = counter + 1
    #plt.xlim([((np.amin(params['offsets'])-(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf']))+138.4, ((np.amax(params['offsets'])+(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf']))+138.4])
    #plt.xlim([((np.amin(params['offsets'])-(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf'])), ((np.amax(params['offsets'])+(2*math.pi*0.8*params['lf']))/(2*math.pi*params['lf']))])
    #plt.ylim([-0.05, 0.6])
    # plt.xlim(5, 20)
    plt.xlabel('$\Omega (ppm)$') 
    plt.ylabel('$V/V_{0}$')
    #plt.xticks([-12, -8, -4, 0, 4, 8, 12])
    plt.legend()
    #plt.ylim([0.0, 0.35])
    #plt.savefig(op_filename + "-volume.pdf")
    plt.show()
 
filename = str(sys.argv[1])
op_filename = filename[:filename.index('.')]
plot_profiles(filename, op_filename) 
