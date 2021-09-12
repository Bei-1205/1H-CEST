#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
import os
import sys
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
sns.set_context('poster', font_scale=1.0)
# matplotlib.rcParams['legend.title_fontsize'] = 'xx-small'


# In[6]:


def read_param(path_home, pathtest, num):
    df = pd.read_csv(path_home + '/' + pathtest + '/fitparms-result_{}.csv'.format(num))
#     pB = ufloat(*df[df['param']=='pB'][['fitval', 'fiterror']].values[0])
#     dwB = ufloat(*df[df['param']=='dwB'][['fitval', 'fiterror']].values[0])
#     kexAB = ufloat(*df[df['param']=='kexAB'][['fitval', 'fiterror']].values[0])
#     rchi2 = ufloat(*df[df['param']=='rchi2'][['fitval', 'fiterror']].values[0])
    pB = df[df['param']=='pB']['fitval'].values[0]*100
    dwB = df[df['param']=='dwB']['fitval'].values[0]
    kexAB = df[df['param']=='kexAB']['fitval'].values[0]
    rchi2 = df[df['param']=='rchi2']['fitval'].values[0]
    
    return [pB, dwB, kexAB, rchi2]

path = os.getcwd()
num = num = int(path.split('_')[-1])
pb_value = []
dw_value = []
kex_value = []

foldername = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
kex_foldername = [name for name in foldername if len(name.split('_')) == 3 and name.split('_')[1] == 'kex']
pb_foldername = [name for name in foldername if len(name.split('_')) == 3 and name.split('_')[1] == 'pb']
dw_foldername = [name for name in foldername if len(name.split('_')) == 3 and name.split('_')[1] == 'dw']


# In[3]:


test_opt_value = read_param(path, 'test', num)
for i in pb_foldername:
    pb_value.append(read_param(path, i, num))
pb_value.append(test_opt_value)
pb_value = np.array(pb_value)
    
for i in kex_foldername:
    kex_value.append(read_param(path, i, num))
kex_value.append(test_opt_value)
kex_value = np.array(kex_value)

for i in dw_foldername:
    dw_value.append(read_param(path, i, num))
dw_value.append(test_opt_value)
dw_value = np.array(dw_value)


# In[4]:


fig, ax = plt.subplots(3, 1, figsize = (8 , 12))

ax[0].scatter(pb_value[:,0], pb_value[:,-1])
ax[1].scatter(dw_value[:,1], dw_value[:,-1])
ax[2].scatter(kex_value[:,2], kex_value[:,-1])

ax[0].scatter(test_opt_value[0], test_opt_value[-1], color = 'r', label = 'best fit')
ax[1].scatter(test_opt_value[1], test_opt_value[-1], color = 'r', label = 'best fit')
ax[2].scatter(test_opt_value[2], test_opt_value[-1], color = 'r', label = 'best fit')
for i in range(3):
    ax[i].set_xlabel(['$p$$_{ES}$ (%)', '$\Delta$$\omega$ (ppm)', '$k$$_{ex}$ (s$^{-1}$)'][i])
    ax[i].set_ylabel('r$\chi$$^2$')



name = sys.argv[1]
ax[0].set_title(name)
plt.tight_layout()
plt.savefig('{}.pdf'.format(name), dpi = 300, transparent = True)


# In[ ]:




