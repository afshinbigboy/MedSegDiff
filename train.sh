#!/bin/bash

word_dir=/home/staff/azad/deeplearning/afshin/MICCAI_2023/diffusion/MedSegDiff
command=/work/scratch/azad/anaconda3/envs/pytorch_cuda11/bin/python
python_file=/home/staff/azad/deeplearning/afshin/MICCAI_2023/diffusion/MedSegDiff/scripts/segmentation_train.py

cd $word_dir

chmod +x train.sh

$command $python_file "$@"