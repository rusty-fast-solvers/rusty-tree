#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=e681
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --time=00:30:00
#SBATCH --nodes=1

# Load required modules
module load PrgEnv-aocc
module load cray-hdf5

# Export PKG_CONFIG path, missing on Archer2
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/cray/pe/mpich/8.1.4/ucx/cray/9.1/lib/pkgconfig/

# Export work and home paths
export WORK=/work/e681/e681/skailasa

# Define a scratch directory for the job
export TEST=${WORK}/rusty-tree/weak
export SCRATCH=${TEST}/${SLURM_JOBID}

# Known issue with ofi
module swap craype-network-ofi craype-network-ucx 
module swap cray-mpich cray-mpich-ucx 
export UCX_IB_REG_METHODS=direct

# Create a scratch directory for this run
mkdir -p ${SCRATCH}
cd ${SCRATCH}

# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/weak_scaling_${SLURM_JOBID}.csv
touch ${OUTPUT}
echo "n_processes, n_leaves, runtime, encoding_time, sorting_time" >> ${OUTPUT}

n_tasks=(1)

# Run jobs
for i in ${!n_tasks[@]}; do
    srun --ntasks=${n_tasks[$i]} --ntasks-per-core=1 --nodes=16 ${TEST}/weak >> ${OUTPUT}
done