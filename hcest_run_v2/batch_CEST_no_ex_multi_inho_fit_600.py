import os
import pandas as pd

BM_file = '''
+
Name result_{}.csv
lf 599.659944
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
pB! 0.0 0.0001 0.1
pC! 0.0 1e-6 0.5
dwB! 0.0 -3 0.0
dwC! 0.0 -80 80
kexAB! 0.0 1.0 50000.0
kexAC! 0.0 1.0 500000.0
kexBC! 0.0 1.0 500000.0
R1 3.0 1 30.
R2 11.0 1 50.
'''

file_names = os.listdir('.')
files_to_loop = filter(lambda x: x.endswith('.csv'), file_names)
files_to_loop = filter(lambda x: x.startswith('result_'), files_to_loop)

for i,n in enumerate(files_to_loop):
	print 'start csv file # {}'.format(i+1)
	os.system('mkdir fit_{}'.format(i+1))
	os.chdir('fit_{}'.format(i+1))
        os.system('cp ../result_{}.csv .'.format(i+1))
        df_t = pd.read_csv('result_{}.csv'.format(i+1))
        # change offset and slp range here
        df_t = df_t[df_t['offset(hz)']>-600*6]
        #df_t = df_t[df_t['slp(hz)']>10]
        df_t = df_t[df_t['offset(hz)']<600*6]
        df_t.to_csv('result_{}.csv'.format(i+1))
	os.system('cp ~/SELOPE/cest_fit_multi_inho_v2.py .')
	os.system('cp ~/SELOPE/exec_inho_v2.sh .')
        with open ("BMNS_params_multi_inho.txt", "w") as f:
		f.write(BM_file.format(i+1))
        os.system('sbatch -p common exec_inho_v2.sh')
	#os.system('python cest_fit_multi_inho.py BMNS_params_multi_inho.txt test')
	print 'finish csv file # {}'.format(i+1)
	os.chdir('../')
