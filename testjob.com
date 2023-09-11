#$ -S /bin/bash
#$ -m e
#$ -N Test_job_kmeans
#$ -q serial
#$ -l h_vmem=60G

source /etc/profile

module add cuda/10.2
module add anaconda3
conda init bash
python kmeans_cloud.py
