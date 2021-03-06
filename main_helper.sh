#!/bin/bash
#SBATCH -o val.%j.txt
#SBATCH -e val.%j.err
#SBATCH -t 5-05:01:00
#SBATCH --mem 999
#SBATCH -n 36
#SBATCH -J adaptive
#SBATCH -p shared

python SpeechDenoise_full5.py \
--fraction_to_drop=${1} \
--k=${2} \
--r=${3} \
--audio=${4} \
--num_samples=${5} \
--speed_over_accuracy=${6}