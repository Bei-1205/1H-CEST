import os 

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
inhomo 0.02
J 0.0
pB 0.0020000 0.0001 0.1
pC! 0.0 1e-6 0.5
dwB -2.0 -5.0 0.0
dwC! 0.0 -80 80
kexAB 3000.0 1.0 50000.0
kexAC! 0.0 1.0 500000.0
kexBC! 0.0 1.0 500000.0
R1 3.0 1 30.
R2 11.0 1 50.
'''

file_names = os.listdir('.')
files_to_loop = filter(lambda x: x.endswith('.csv'), file_names)

for i,n in enumerate(files_to_loop):
	print 'start csv file # {}'.format(i)
	os.system('mkdir fit_{}'.format(i+1))
	os.chdir('fit_{}'.format(i+1))
        os.system('cp ../result_{}.csv .'.format(i+1))
	os.system('cp ~/SELOPE/cest_fit_final_v10.py .')
	with open ("BMNS_params_v10.txt", "w") as f:
		f.write(BM_file.format(i+1))
	os.system('python cest_fit_final_v10.py BMNS_params_v10.txt test')
	print 'finish csv file # {}'.format(i+1)
	os.chdir('../')
