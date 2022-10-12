#!/bin/bash

slurm_conf="/opt/slurm/slurm-19.05.8/etc/slurm.conf"
uid=$(date | sed "s/ /-/g")
cmd="/tmp/cmd-${uid}.out"
stdout="/tmp/stdout-${uid}.out"

configure_compute_node_memory() {
    # CONFIGURE COMPUTE NODE MEMORY:
    # POST to fix memory issue: - Memory issue... https://www.gitmemory.com/issue/aws/aws-parallelcluster/1517/561775124
    #     NOTE that NodeName=DEFAULT RealMemory=[RealMemory] need to go in the same line of the config!
    echo; echo
    echo "CONFIGURING COMPUTE NODE MEMORY"
    # Get real memory (QUESTION: What unit is this in?)
    RealMemory=$(slurmd -C | tr ' ' '\n' | grep RealMemory | cut -d'=' -f2)
    # Use less than 100% of reported memory
    RealMemory=$((RealMemory*9/10))
    echo "Real Memory ${RealMemory}"
    echo "Adding RealMemory to slurm configuration file:"
    echo "${slurm_conf}"
    sudo sed -i "/^NodeName=/ s/$/ RealMemory=${RealMemory}/" ${slurm_conf}
    echo; cat ${slurm_conf}; echo
    # AWS: Edit slurm.conf and add NodeName=DEFAULT RealMemory=[RealMemory for compute nodes] BEFORE include!
    # AWS:    /opt/slurm/etc/slurm.conf
    echo "Re-starting Slurmctld"
    sudo /opt/slurm/current/sbin/slurmctld
    # Ass root:
    #slurmctld # Google: /opt/slurm/current/sbin/slurmctld
    # Verify the change:
    echo "Verifying configuration change:"
    scontrol show nodes | grep RealMemory
}

# Parse arguments
memory="--memory 1GB"
cores="--cores 1"
index=1
for arg in $@; do
    prefix=$(echo "${arg}" | cut -c1-2)
    if [[ ${prefix} == '--' ]]; then
	    if [[ ${arg} == '--remote_dir' ]]; then
	        # Maybe need to change directory in the future?
	        remote_dir=$(echo $@ | cut -d ' ' -f$((index + 1)))
        elif [[ ${arg} == '--conda_env' ]]; then
	        conda_env=$(echo $@ | cut -d ' ' -f$((index + 1)))
	        echo "Conda environment: ${conda_env}"
	    elif [[ ${arg} == '--conda_sh' ]]; then
	        conda_sh=$(echo $@ | cut -d ' ' -f$((index + 1)))
            echo "Conda source: ${conda_sh}"
	        source ${conda_sh}
        elif [[ ${arg} == '--backend' ]]; then
	        backend=$(echo $@ | cut -d ' ' -f$((index + 1)))
	        if [[ ${backend} == 'dask' ]]; then
                log='/tmp/configure_compute_node_memory.out'
		        # Check if jupyterlab was already started:
		        if ! [ -f "${log}" ]; then
		            configure_compute_node_memory > ${log}
                fi
		        cat ${log}
                RealMemory=$(slurmd -C | tr ' ' '\n' | grep RealMemory | cut -d'=' -f2)
                RealMemory=$((RealMemory*9/10))
                memory="--memory $((RealMemory*9/10000))GB"
                cores="--cores $(($(slurmd -C | tr ' ' '\n' | grep CPUs | cut -d'=' -f2)/2))"
            fi
	    fi
    fi
    index=$((index+1))
done

#source /home/avidalto/miniconda3/etc/profile.d/conda.sh
#source /contrib/Alvaro.Vidal/miniconda3/etc/profile.d/conda.sh
# FIXME: Generalize conda env!
conda activate ${conda_env}
echo "$@ ${memory} ${cores}" >> ${cmd}
$@ ${memory} ${cores}  2>&1 | tee ${stdout}
