# 1H-CEST
### Fitting scripts for 1H CEST data

### Ref: Liu et al, (2021) "Rapid assessment of Watson-Crick to Hoogsteen exchange in unlabeled DNA duplexes using high-power SELOPE imino 1H CEST" Magnetic Resonance discussion

#### csv_output: the folder contains fitting results for 1HCEST data assuming no exchange
#### csv_output_no_kex: the folder contains fitting results for 1HCEST data assuming there is exchange

#### inside csv_output or csv_output_no_kex, result1-9.csv are the 1HCEST raw data for all 9 peaks in the 1D imino spectra;
#### Fit_1-9 are the folders contains all the fitting results;
#### batch_CEST_fit.py or batch_CEST_fit_offset_filter.py are the fitting scripts to fit 9 1HCEST profiles, with or without constraining offsets
#### inside each fit folder (e.g. Fit_1), cest_fit_final_v10.py is the 1HCEST fitting script and BMNS_params_v10.txt is the input parameters for fitting, test is the folder including fitting results
#### The plot_profile.py script is used to plot the 1HCEST data without fitting. To plot the result_1.csv data, use the command  "plot_profile.py result_1.csv"
#### python 2.7 is required for the scripts described above



#### residual_AIC_BIC_HCEST_v2.py (python 3.6 needed for this script), this scirpt is used to combine all fitting results (profiles in pdf and exchange parameters in csv)
#### the csv_output and csv_output_no_kex folders are the input, the assignment should be changed in line 246 in residual_AIC_BIC_HCEST_v2.py
#### the output files are 
#### 1. *_fit.pdf, which is the pdf containing all 1HCEST profiles assuming there is exchange;
#### 2. *_nokex.pdf, the pdf containing all 1HCEST profiles assuming no exchange
#### 3. *_combined.pdf, the combination of 1 and 2
#### 4. *_fitparm.csv inside csv_output, the fitting parameters for all 9 1HCEST fitting
#### 5. residual_*_residual.pdf, the residual plot and AIC/BIC values for each 1HCEST profile


