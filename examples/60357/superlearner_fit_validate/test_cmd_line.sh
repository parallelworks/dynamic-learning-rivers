#!/bin/bash

# Run command in local from local:
fit_with_hold_out_MIMOSuperLearner_local_to_local() {
    python main.py \
       --backend threading \
	   --remote_dir /tmp \
	   --conda_sh  /home/avidalto/miniconda3/etc/profile.d/conda.sh \
       --conda_env base \
	   --superlearner_conf ./superlearner_fit_validate/sample_io/converge/superlearner_conf.py \
	   --data ./superlearner_fit_validate/sample_io/converge/data250.csv \
	   --num_inputs 9 \
	   --model_dir ./superlearner_fit_validate/sample_io/converge/ConvergeSL \
	   --n_jobs 4 \
       --hpo True \
       --cross_val_score True \
       --sanity_test None
}

# Run command in remote from local:
fit_with_hold_out_MIMOSuperLearner_local_to_remote() {
    python main.py \
	   --app_host Alvaro.Vidal@34.71.208.176 \
	   --backend dask \
	   --remote_dir /tmp \
	   --conda_sh /contrib/Alvaro.Vidal/miniconda3/etc/profile.d/conda.sh \
       --conda_env base \
	   --superlearner_conf ./sample_io/converge/superlearner_conf.py \
	   --data ./sample_io/converge/data250.csv \
	   --num_inputs 9 \
	   --model_dir ./sample_io/converge/ConvergeSL \
	   --n_jobs 4 \
	   --hpo True \
       --cross_val_score True \
	   --sanity_test None
}

fit_with_hold_out_MIMOSuperLearner_local_to_local
#fit_with_hold_out_MIMOSuperLearner_local_to_remote
