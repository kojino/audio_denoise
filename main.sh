#!/bin/bash
#SBATCH -o master.%j.txt
#SBATCH -e master.%j.err
#SBATCH -t 05:01:00
#SBATCH --mem 999
#SBATCH -J adaptive
#SBATCH -p shared

ks=(60)
rs=(30)
fraction_to_drops=(0.50)
audios=('call' 'billy')
num_sampless=(216)
speed_over_accuracys=(1)

for k in ${ks[@]}
do
for r in ${rs[@]}
do
for audio in ${audios[@]}
do
for speed_over_accuracy in ${speed_over_accuracys[@]}
do
for fraction_to_drop in ${fraction_to_drops[@]}
do
for num_samples in ${num_sampless[@]}
do

./main_helper.sh \
$fraction_to_drop \
$k \
$r \
$audio \
$num_samples \
$speed_over_accuracy

done
done
done
done
done
done
