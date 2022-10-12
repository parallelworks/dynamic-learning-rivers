

    WORKDIR=/tmp/pworks
    #set -x

    (
        remote_rundir=$1
        workerid=$2
        clienthost=$3
        clientuser=$4
        sleeptime=$5
        
        # the tunnelhost is the ssh config location for the user container
        tunnelhost="$clienthost-container"

        echo "Parsl CoG job starting, $# args: $*"
        echo "Args: remote_rundir: $remote_rundir, workerid: $workerid, clienthost: $clienthost, clientuser: $clientuser, sleeptime: $sleeptime, tunnelhost: $tunnelhost"
        echo "Running as user: $(id) in dir: $PWD, workdir: $WORKDIR"

        # HANDLE THE COMPUTE NODE CONDA ENVIRONMENT
        
        condaenv="parsl-pw"

        CONDA_EXISTS_LOCAL="false"

        # first check if conda the desired conda env already exists on the node
        # this is the default selection that leaves responsibility of conda setup up to the user
        if command -v conda &> /dev/null
        then
            echo "conda env exists... checking if $condaenv already exists:"

            conda activate $condaenv
            RESULT=$?
            if [ $RESULT -eq 0 ]; then
                CONDA_EXISTS_LOCAL="true"
                echo "success... conda $condaenv env already exists - using this."
            else
                echo "conda $condaenv env does not exist - going to bootstrap a new conda."
            fi

        fi

        # if conda doesn't exist, lets try to stage it in
        if [[ "$CONDA_EXISTS_LOCAL" == "false" ]];then

            # we are going to put the default conda env in the user specifed working directory
            
            # this needs to allow only one worker to pull the conda env if it doesn't exist, and others will wait for it to finish
            
            # lockf -> revise files to use this
            
            if [ ! -f "$WORKDIR/CONDA_STAGED" ];then
            
                # check if another process is already loading it
                if [ ! -f "$WORKDIR/CONDA_LOADING" ];then
                
                    touch "$WORKDIR/CONDA_LOADING"
                
                    echo "staged conda env does not yet exist... staging the conda env over now"
                    
                    # check present or stage the data using rsync
                    # File on GCE /tmp are not persistent, so use /var/lib, stateful partition
                    # according to: https://cloud.google.com/container-optimized-os/docs/concepts/disks-and-filesystem
                    if [ -d "/var/lib/pworks/.miniconda3" ]; then
                        # Already present so just move it to where the staging would happen.
                        # Need to change ownership of Conda directory since
                        # different users can spin up this image and ownership is
                        # required for installing the parslmods.tgz, below.
                        current_user=$(id -u -n)
                        current_group=$(id -g -n)
                        sudo chown $current_user --recursive /var/lib/pworks/.miniconda3
                        sudo chgrp $current_group --recursive /var/lib/pworks/.miniconda3
                        mv /var/lib/pworks/.miniconda3 /tmp/
                    else
                        # stage the data using rsync
                        cd /tmp
                        echo rsync -vvv $tunnelhost:/pw/.packs/miniconda3.tgz ./miniconda3.tgz
                        rsync -vvv $tunnelhost:/pw/.packs/miniconda3.tgz ./miniconda3.tgz
                        tar xzf miniconda3.tgz
                        mv miniconda3 .miniconda3 > /dev/null 2>&1
                    fi
    
                    # replace the conda path with the new download location path
                    find /tmp/.miniconda3/bin/conda -type f -exec sed -i "s|/pw/|/tmp/|g" {} +
                    find /tmp/.miniconda3/envs/parsl-pw/bin/process_worker_pool.py -type f -exec sed -i "s|/pw/|/tmp/|g" {} +
                    find /tmp/.miniconda3/etc -type f -exec sed -i "s|/pw/|/tmp/|g" {} +
                    find /tmp/.miniconda3/envs/parsl-pw/etc -type f -exec sed -i "s|/pw/|/tmp/|g" {} +
                    find /tmp/.miniconda3/envs/parsl-pw/conda-meta -type f -exec sed -i "s|/pw/|/tmp/|g" {} +
    
                    # link it back to the run directory for easy access
                    ln -s /tmp/.miniconda3 $WORKDIR/.miniconda3
                    
                    touch "$WORKDIR/CONDA_STAGED"
                    rm "$WORKDIR/CONDA_LOADING"
                
                elif [ -f "$WORKDIR/CONDA_LOADING" ];then
                
                    echo "staged conda env does not yet exist but another worker is loading it... waiting for it to finish"
                    
                    while [ ! -f "$WORKDIR/CONDA_STAGED" ]
                    do
                      sleep 1
                    done
                    
                    echo "conda env finished loading from another worker..."
                    
                fi
            
            else
            
                echo "staged conda already exists - moving on..."
                
            fi

            # load the staged conda environment
            . "$WORKDIR/.miniconda3/etc/profile.d/conda.sh"

            conda activate $condaenv

        fi

        # go the working directory
        cd $WORKDIR

        # List info about current environment
        conda info

        # Install user's remote Python packages if present
        if [ -f $WORKDIR/remotepacks.tgz ]; then
            ( cd $WORKDIR/.miniconda3/envs/parsl-pw/bin
            tar zxvf $WORKDIR/remotepacks.tgz
            )
        fi

        # Install parslmods if present
        # parslmods provides a way to make changes to just the parsl build without having to reinstall conda
        if [ -f $WORKDIR/parslmods.tgz ]; then
            echo "loading parslmods"
            cd $WORKDIR/.miniconda3/envs/parsl-pw/lib/python3.7/site-packages/
            tar --overwrite -xzf $WORKDIR/parslmods.tgz
            cp parsl/executors/high_throughput/process_worker_pool.py $WORKDIR/.miniconda3/envs/parsl-pw/bin/process_worker_pool.py
            chmod +x $WORKDIR/.miniconda3/envs/parsl-pw/bin/process_worker_pool.py
            cd $WORKDIR
        fi

        python -c "import parsl; print('\nRunning parsl version {}\n'.format(parsl.__version__))"
        export CONDA_ENV=$condaenv PARSL_CLIENT_USER=$clientuser PW_USER_HOST=$clienthost PARSL_CLIENT_HOST=$tunnelhost PARSL_CLIENT_SSH_PORT=64004  # To communicate host and user to launchcmds py scripts

        # run the parsl launch commands
        # this launchcmds will establish tunnels back to either direct user container or login node
        source $WORKDIR/launchcmds.txt &

        ssh -p $PARSL_CLIENT_SSH_PORT $clientuser@$tunnelhost "echo status:up hostname:$(hostname) > $remote_rundir/$workerid.started"

        wait  # Waiting for multiple workers and manager to exit here. # FIXME: Get multiple status from wait???
        rc=$?

        ssh $clientuser@$tunnelhost "echo status:done hostname:$(hostname) rc:$rc > $remote_rundir/$workerid.ended"

    ) 2>&1 | tee parsl-htex-worker.out

    exit 0
    