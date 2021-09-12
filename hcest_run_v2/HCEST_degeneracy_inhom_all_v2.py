import os 
import pandas as pd
import sys
# get parameters from the best fitting in the "test" folder
num = int(os.getcwd().split('_')[-1])

df = pd.read_csv('test/fitparms-result_{}.csv'.format(num))

best_fit_pb = df[df['param']=='pB']['fitval'].values[0]
#dwB = df[df['param']=='dwB']['fitval'].values[0]
best_fit_kex = df[df['param']=='kexAB']['fitval'].values[0]
#rchi2 = df[df['param']=='rchi2']['fitval'].values[0]

# larmor frequency
if str(sys.argv[1]) == '700':
    lf = 699.9348710
elif str(sys.argv[1]) == '600':
    lf = 599.659943

# pb
BM_file1 = '''
+
Name result_{}.csv
lf {}
AlignMag AUTO
resn 1.0 
eqbn Y
fitvar I
ls N
err_mode STD
mc_iter 10
inhomo 0.03 0.02 0.014 0.01
number_inhomo_slps 40
J 0.0
pB! {} 0.000001 0.1
pC! 0.0 1e-6 0.5
dwB -2.0 -6.0 0.0
dwC! 0.0 -80 80
kexAB 2000 1.0 5000000.0
kexAC! 0.0 1.0 500000.0
kexBC! 0.0 1.0 500000.0
R1 3.0 1 50.
R2 11.0 1 50.
'''

sh_script1 = '''#!/usr/bin/bash
#SBATCH --mem=4G

python cest_fit_multi_inho_v2.py BMNS_params_pb_{0}.txt test_pb_{0}
'''
# pB range
for i in [0.1, 0.33, 1, 3, 10]:
	with open ("BMNS_params_pb_{}.txt".format(i), "w") as f:
		f.write(BM_file1.format(num, lf, i*best_fit_pb))
	with open ('exec_inho_pb_{}.sh'.format(i), 'w') as f:
		f.write(sh_script1.format(i))
	# os.system('python cest_fit_final_v10.py BMNS_params_v10_{0}.txt test_fold_{0}'.format(i))
	os.system('sbatch -p common ' + 'exec_inho_pb_{}.sh'.format(i))

# dw
BM_file2 = '''
+
Name result_{}.csv
lf {}
AlignMag AUTO
resn 1.0 
eqbn Y
fitvar I
ls N
err_mode STD
mc_iter 10
inhomo 0.03 0.02 0.014 0.01
number_inhomo_slps 40
J 0.0
pB 0.001 0.000001 0.1
pC! 0.0 1e-6 0.5
dwB! {} -6.0 0.0
dwC! 0.0 -80 80
kexAB 2000 1.0 5000000.0
kexAC! 0.0 1.0 500000.0
kexBC! 0.0 1.0 500000.0
R1 3.0 1 50.
R2 11.0 1 50.
'''

sh_script2 = '''#!/usr/bin/bash
#SBATCH --mem=4G

python cest_fit_multi_inho_v2.py BMNS_params_dw_{0}.txt test_dw_{0}
'''
# dw range
for i in [-4, -3, -2, -1]:
	with open ("BMNS_params_dw_{}.txt".format(i), "w") as f:
		f.write(BM_file2.format(num, lf, i))
	with open ('exec_inho_dw_{}.sh'.format(i), 'w') as f:
		f.write(sh_script2.format(i))
	# os.system('python cest_fit_final_v10.py BMNS_params_v10_{0}.txt test_fold_{0}'.format(i))
	os.system('sbatch -p common ' + 'exec_inho_dw_{}.sh'.format(i))


# kex
BM_file3 = '''
+
Name result_{}.csv
lf {}
AlignMag AUTO
resn 1.0 
eqbn Y
fitvar I
ls N
err_mode STD
mc_iter 10
inhomo 0.03 0.02 0.014 0.01
number_inhomo_slps 40
J 0.0
pB 0.001 0.000001 0.1
pC! 0.0 1e-6 0.5
dwB -2.0 -6.0 0.0
dwC! 0.0 -80 80
kexAB {} {} {}
kexAC! 0.0 1.0 500000.0
kexBC! 0.0 1.0 500000.0
R1 3.0 1 50.
R2 11.0 1 50.
'''

sh_script3 = '''#!/usr/bin/bash
#SBATCH --mem=4G

python cest_fit_multi_inho_v2.py BMNS_params_kex_{0}.txt test_kex_{0}
'''
# kex range
for i in [0.1, 0.33, 1, 3, 10]:
	with open ("BMNS_params_kex_{}.txt".format(i), "w") as f:
		f.write(BM_file3.format(num, lf, i*best_fit_kex, i*best_fit_kex-1, i*best_fit_kex+1))
	with open ('exec_inho_kex_{}.sh'.format(i), 'w') as f:
		f.write(sh_script3.format(i))
	os.system('sbatch -p common ' + 'exec_inho_kex_{}.sh'.format(i))
