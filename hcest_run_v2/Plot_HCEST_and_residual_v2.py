#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import import_ipynb
import numpy as np
import pandas as pd
import four_state_CEST_simulations as CEST_sim
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os
import sys
from matplotlib.offsetbox import AnchoredText
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['legend.title_fontsize'] = 15
sns.set_context('poster', font_scale=1.5)
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# plot 1H CEST profiles
# read data csv, dot plot for HCEST data
# read fit results csv, run simulations to plot the trendline

# BMNS input needed for simulation
BMNS = '''
+
lf {7}
pb {0}
pc 0.0
pd 0.0
dwb {1}
dwc 0.0
dwd 0.0
kexAB {2}
kexAD 0
kexBC 0
kexCD 0
R1a {3}
R2a {4}
R1b {3}
R2b {4}
R1c {3}
R2c {4}
R1d {3}
R2d {4}
resn 1.0
T {6}
mode AUTO
error_point 0.00000
error_baseinten 0.00
inhomo {5}
equil Y
ls N

+
off 50 -3600 3600 100
off 100 -3600 3600 100
off 250 -3600 3600 100
off 500 -3600 3600 100
off 2000 -3600 3600 100
'''

# larmor frequency
if str(sys.argv[4]) == '700':      
    lf = 699.9348710
elif str(sys.argv[4]) == '600':
    lf = 599.659943

# colors
colors_plot = ['k', 'r', 'b', 'g', 'cyan', 'magenta', 'brown', 'yellow', 'teal', 'lightgreen']

def ax_plot_CEST_fitting(path, num, ax, tdelay = 0.1):
    '''
    plot CEST data and run simulation to re-create the fitting profile
    path is the path to the master fit
    num is the number of fitting folders
    '''
    pb, dw, kex, R1, R2, rchi2 = read_param(path, 'fit_{}/test'.format(num), num)
    slp_exp, offset_exp, exp_df = CEST_sim.get_offset_exp(path + '/fit_{}/test'.format(num) + '/fit-result_{}.csv'.format(num))
    slp_exp.sort()
    number_slps = np.unique(slp_exp)

    for i, n in enumerate(number_slps):
        # get inhomogeneity for each slp
        slp_e = n
        inhome = get_inhome(slp_e, lf)
        
        # simulation
        with open('dummy_BMNS.txt', 'w+') as f:
            f.write(BMNS.format(pb, dw, kex, R1, R2, inhome, tdelay, lf))
        
        # a is the df output of simulation
        a, b = CEST_sim.get_output('dummy_BMNS.txt', slp_exp= [slp_e], 
                                   offset_exp=np.linspace(np.min(offset_exp), np.max(offset_exp), 100), plot = False)

        sim_inten = a['norm_intensity']
        sim_offset = a['offset(hz)']
        offset = exp_df[np.round(exp_df['slp(hz)']) == slp_e ]['offset(hz)'].values
        exp_inten = exp_df[np.round(exp_df['slp(hz)']) == slp_e ]['norm_intensity'].values
        exp_inten_error = exp_df[np.round(exp_df['slp(hz)']) == slp_e]['norm_intensity_error'].values
        
        # plot data
        ax.errorbar(offset/lf, exp_inten, yerr = exp_inten_error,
                    color = colors_plot[i], label='{}'.format(n), 
                    linewidth=3, fmt='o', markersize=7)
        
        # plot similation
        ax.plot(sim_offset/lf, sim_inten, color = colors_plot[i], linewidth=3)

        l = ax.legend(loc = 4, fontsize = 20, frameon=False,
                     handletextpad=-1.0, handlelength=0, markerscale=0,
                     title = '$\omega$' + '/$2\pi$' + ' (Hz)',
                     ncol=3, columnspacing=2)
        l._legend_box.align = "right"
        

        line = 'r$\chi$$^2$' + '= {:.2f}'.format(rchi2)
        anchored_text = AnchoredText(line, loc = 3, prop=dict(size=30), frameon= False)
        ax.add_artist(anchored_text)

        for item in l.legendHandles:
            item.set_visible(False)
        for handle, text in zip(l.legendHandles, l.get_texts()):
            text.set_color(handle.get_color()[0])
            
    return dw

def read_param(path_home, pathtest, num):
    '''
    read exchange parameters from the fitparms=result_{num}.csv
    pathtest is the folder of the fitting output e.g. fit_{num}/test
    '''
    df = pd.read_csv(path_home + '/' + pathtest + '/fitparms-result_{}.csv'.format(num))
    pB = df[df['param']=='pB']['fitval'].values[0]
    dwB = df[df['param']=='dwB']['fitval'].values[0]
    kexAB = df[df['param']=='kexAB']['fitval'].values[0]
    rchi2 = df[df['param']=='rchi2']['fitval'].values[0]
    R1 = df[df['param']=='R1']['fitval'].values[0]
    R2 = df[df['param']=='R2']['fitval'].values[0]
    
    return [pB, dwB, kexAB, R1, R2, rchi2]     
        
    


# In[6]:


def get_inhome(slp, lf):
    '''
    Decide inhomogeneity used for simulation
    '''
    if lf == 599.659943:
        if slp >=1000:
            return 0.01
        elif slp >= 500:
            return 0.014
        elif slp >= 200:
            return 0.02
        elif slp >= 100:
            return 0.03
        elif slp >= 50:
            return 0.05
        else:
            return 0.19
    elif lf == 699.9348710:
        if slp >= 250:
            return 0.0497
        elif slp >= 100:
            return 0.06485
        else:
            return 0.1016
    else:
        print('invalid lf, please choose 700 or 600')
        sys.exit(0)

# Ref: http://www.originlab.com/doc/Origin-Help/PostFit-CompareFitFunc
# Ref:  Psychonomic Bulletin & Review 2004, 11 (1), 192-196
def AIC(N, exp, fit, K):
    '''
    calculate AIC given N(numbe of data), K(variables), exp:experimental intensity, fit:fitting intensity
    '''
    # residual sum of squares 
    RSS = np.sum((exp-fit)**2)
    
    if N/K >= 40:
        AIC_v = N*np.log(RSS/N) + 2*K
    else:
        AIC_v = N*np.log(RSS/N) + 2*K + 2*K*(K+1)/(N-K-1)
        
    return AIC_v

def BIC(N, exp, fit, K):
    '''
    calculate AIC given N(numbe of data), K(variables), exp:experimental intensity, fit:fitting intensity
    '''
    RSS = np.sum((exp-fit)**2)
    
    BIC_v = N*np.log(RSS/N) + K*np.log(N)
    
    return BIC_v

# calculate AIC/BIC
def AIC_BIC(df_kex, df_nokex):
    
    number_slps = df_kex['slp(hz)'].unique()
    colors_plot = ['k', 'r', 'b', 'g', 'cyan', 'magenta', 'brown', 'yellow', 'teal', 'lightgreen']
    # calculate AIC/BIC
    kex_exp = df_kex['norm_intensity']
    kex_fit = df_kex['fit_norm_intensity']

    nokex_exp = df_nokex['norm_intensity']
    nokex_fit = df_nokex['fit_norm_intensity'] 
    
    
    AIC_kex = AIC(len(kex_exp), kex_exp.values, kex_fit.values, 5)
    BIC_kex = BIC(len(kex_exp), kex_exp.values, kex_fit.values, 5)

    AIC_nokex = AIC(len(nokex_exp), nokex_exp.values, nokex_fit.values, 2)
    BIC_nokex = BIC(len(nokex_exp), nokex_exp.values, nokex_fit.values, 2)

    AIC_total = np.array([AIC_kex, AIC_nokex])
    delta_AIC = AIC_total - np.min(AIC_total)
    wAIC_kex, wAIC_nokex = (np.exp(-0.5*delta_AIC)/np.sum(np.exp(-0.5*(delta_AIC))))

    BIC_total = np.array([BIC_kex, BIC_nokex])
    delta_BIC = BIC_total - np.min(BIC_total)
    wBIC_kex, wBIC_nokex = (np.exp(-0.5*delta_BIC)/np.sum(np.exp(-0.5*(delta_BIC)))) 
    
    return wAIC_kex, wBIC_kex, wAIC_nokex, wBIC_nokex


def plot_residual(path, num, ax, AIC_BIC = None):
    '''
    residula plot containing the AIC/BIC values
    '''
    df_kex = pd.read_csv(path + '/fit_{0}/test/fit-result_{0}.csv'.format(num))
    
    number_slps = df_kex['slp(hz)'].unique()
    number_slps.sort()
    
    for i, dummy_slp in enumerate(number_slps):
        df_kex2 = df_kex.loc[df_kex['slp(hz)'] == dummy_slp]
        ax.scatter(df_kex2['offset(hz)']/lf, df_kex2['norm_intensity'] - df_kex2['fit_norm_intensity'], 
                    color = colors_plot[i], label='%4.0f'%float(dummy_slp), s = 30)


#         l = ax.legend(fontsize = 20, frameon=False,
#                      handletextpad=-2.0, handlelength=0, markerscale=0,
#                      title = '$\omega$' + '/$2\pi$' + ' (Hz)',
#                      ncol=3, columnspacing=2.0)
#         l._legend_box.align = "right"
#         for item in l.legendHandles:
#             item.set_visible(False)
#         for handle, text in zip(l.legendHandles, l.get_texts()):
#             text.set_color(handle.get_facecolor()[0])
            
        if AIC_BIC:
            line = '$wAIC$$_{+ex}$' + '= {:.3f}'.format(AIC_BIC[0]) + '\n$wBIC$$_{+ex}$'+' ={:.3f}'.format(AIC_BIC[1])
            anchored_text = AnchoredText(line, loc=2, prop=dict(size=30), frameon= False)
            ax.add_artist(anchored_text)


# In[8]:


path_ex = os.getcwd() + '/' + str(sys.argv[1])
path_no_ex = os.getcwd() + '/' + str(sys.argv[2])
assignment = ['T5', 'T6', 'T7', 'T8/T4', 'T9', 'T22', 'G11', 'G10', 'G2']

# number of row in subplots
num_row = int(np.ceil(len(assignment)/3))

# plots 6 columns * num_row rows
fig, ax = plt.subplots(2*num_row, 6, figsize = (16*3,6*2*num_row), sharex='col', sharey='row')
    
# plot A6RNA    
count = 0
for i, n in enumerate(assignment):
    ax_plot_CEST_fitting(path_no_ex, i+1, ax[2*(count//3),(count%3)*2])
    ax_plot_CEST_fitting(path_ex, i+1, ax[2*(count//3),(count%3)*2+1])

    df1 = pd.read_csv(path_ex + '/fit_{0}/test/fit-result_{0}.csv'.format(i+1))
    df2 = pd.read_csv(path_no_ex + '/fit_{0}/test/fit-result_{0}.csv'.format(i+1))
    AIC_v, BIC_v, _, _ = AIC_BIC(df1, df2)
    plot_residual(path_no_ex, i+1, ax[2*(count//3)+1,(count%3)*2])
    plot_residual(path_ex, i+1, ax[2*(count//3)+1,(count%3)*2+1], AIC_BIC=[AIC_v, BIC_v])

    if 2*(count//3) == 0:
        ax[2*(count//3),(count%3)*2].set_title('- exchange\n' + n, fontsize = 40)
        ax[2*(count//3),(count%3)*2+1].set_title('+ exchange\n' + n, fontsize = 40)
    else:
        ax[2*(count//3),(count%3)*2].set_title(n, fontsize = 40)
        ax[2*(count//3),(count%3)*2+1].set_title(n, fontsize = 40)

    count += 1
for i in range(6):
    ax[2*num_row-1,i].set_xlabel('$\Omega(ppm)$')

# adjust ylim for each row here
for i in range(num_row*2)[::2]:
    ax[i,0].set_ylabel('Norm. Intensity')
    ax[i,0].set_ylim(0,1.0)
    ax[i+1,0].set_ylabel('Residual')
    ax[i+1,0].set_ylim(-0.1, 0.1)
    
# remove empty plots
if len(assignment) != num_row*3:
    for i in [2*num_row-2, 2*num_row-1]:
        for j in range(len(assignment)%3*2, 6):
            ax[i,j].remove()
            ax[2*num_row-3,j].set_xlabel('$\Omega(ppm)$')
        
# title 
plt.figtext(0.51,1.01,"{}".format(str(sys.argv[3])), va="center", ha="center", size=50)
# plt.suptitle("A$_6$-DNA 25$^{\circ}$C small offsets", fontsize = 40)
fig.align_labels()
plt.tight_layout()

# save
plt.savefig('{}.pdf'.format(str(sys.argv[3])), dpi = 300, transparent = True, bbox_inches = "tight")

