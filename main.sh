#!/bin/bash
#SBATCH -o master.%j.txt
#SBATCH -e master.%j.err
#SBATCH -t 05:01:00
#SBATCH --mem 999
#SBATCH -J adaptive
#SBATCH -p shared

ks=(60 80 100 150 200 250 300 400 500)
rs=(4 6 8 10 12)
fraction_to_drops=(0.05 0.10 0.15 0.18 0.20 0.30 0.40 0.50)
audios=('alexa' 'piano' 'marshmello')

for k in ${ks[@]}
do
for fraction_to_drop in ${fraction_to_drops[@]}
do
for r in ${rs[@]}
do
for audio in ${audios[@]}
do
./main_helper.sh \
$fraction_to_drop \
$k \
$r \
$audio \

done
done
done
done