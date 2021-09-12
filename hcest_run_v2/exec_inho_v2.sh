#!/usr/bin/bash
#SBATCH --mem=4G

python2 cest_fit_multi_inho_v2.py BMNS_params_multi_inho.txt test
