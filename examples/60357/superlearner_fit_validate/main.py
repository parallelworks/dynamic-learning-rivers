import sklearn
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import joblib; print(joblib.__version__)

import pandas as pd
import numpy as np

import importlib

import sys
import argparse
import os, shutil, pickle, json
from copy import deepcopy

def clean_data_df(data_df):
    # FIXME: Clean outliers, decide filling NAN methods, etc
    data_df.fillna(data_df.mean(), inplace = True)
    return data_df

def load_data_csv_io(data_csv, num_inputs):
    data_df = pd.read_csv(data_csv).astype(np.float32)
    data_df = clean_data_df(data_df)
    pnames = list(data_df)
    X = data_df.values[:, :num_inputs]
    Y = data_df.values[:, num_inputs:]
    return X, Y, pnames[:num_inputs], pnames[num_inputs:]

def save_data_csv_io(x_np, y_np, inames, onames, out_file_name):
    # Concatenate X and Y into single csv output
    x_df = pd.DataFrame(x_np,columns=inames)
    y_df = pd.DataFrame(y_np,columns=onames)
    df_out = pd.concat([x_df,y_df],axis=1)
    df_out.to_csv(out_file_name,index=False,na_rep='NaN')

def format_estimators(estimators_dict):
    # Define StackingRegressor
    estimators = []
    for est_id, est_conf in estimators_dict.items():
        estimators.append((est_id, est_conf['model']))
    return estimators


# LOAD DATA
# If dataset is too big fitting may fail due to memory issues that are not catched in the logs!
# FIT MODELS IN PARALLEL
if __name__ == '__main__':
    # FIXME! Where is this running from!
    parser = argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)

    args = parser.parse_args()

    if args.backend == 'dask':
        n_jobs = int(args.n_jobs)
        # FIXME: Make this code common
        from dask_jobqueue import SLURMCluster
        import dask
        from dask.distributed import Client

        # Log dir needs to be accessible to the compute nodes too!!!!
        dask_log_dir = '/contrib/dask-logs/'
        cluster = SLURMCluster(
            cores = int(args.cores),
            memory= str(args.memory),
            walltime= '00:55:00',
            log_directory= dask_log_dir,
            env_extra= ['source ' + args.conda_sh + '; conda activate']
        )
        cluster.adapt(minimum = 0, maximum = n_jobs)
        client = Client(cluster)
        backend_params = {'wait_for_workers_timeout': 600}
    else:
        backend_params = {}
        n_jobs = None

    # Create Model Directory
    args.model_dir = args.model_dir.replace('*','')
    os.makedirs(args.model_dir, exist_ok = True)

    # Load Data
    X, Y, inames, onames = load_data_csv_io(args.data, int(args.num_inputs))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # Apply sampling to training set only
    # CAUTION: Template for testing only.
    # Currently works only if Y is an integer
    # which is interpreted as a class number
    # by imblearn.  In the future, generalize this
    # by including a non-target column that is
    # the class of each row so imblearn can be
    # applied more generally.
    #ros = RandomOverSampler()
    #rus = RandomUnderSampler()
    #X_train, Y_train = rus.fit_resample(X_train, Y_train)
    #Y_train = np.expand_dims(Y_train, axis=1)
    
    # Load SuperLearner Configuration
    sys.path.append(os.path.dirname(args.superlearner_conf))
    sl_conf = getattr(
        importlib.import_module(os.path.basename(args.superlearner_conf.replace('.py',''))),
        'SuperLearnerConf'
    )
    # The SL configuration file is needed to load the SL pickle
    try:
        # To prevent same file error!
        shutil.copy(args.superlearner_conf, args.model_dir)
    except:
        pass # FIXME: Add error handling!

    # Running hyperparameter optimization:
    if args.hpo == "True":
        sl_conf_hpo = deepcopy(sl_conf)
        sl_conf_hpo['estimators'] = {}
        for oi, oname in enumerate(onames):
            sl_conf_hpo['estimators'][oname] = {}

            if oname in sl_conf['estimators']:
                estimators = sl_conf['estimators'][oname]
            else:
                estimators = sl_conf['estimators']

            for ename, einfo in estimators.items():
                print('Running HPO for output {} and estimator {}'.format(oname, ename), flush = True)
                sl_conf_hpo['estimators'][oname][ename] = {}

                if 'hpo' not in einfo:
                    sl_conf_hpo['estimators'][oname][ename]['model'] = einfo['model']

                with joblib.parallel_backend(args.backend, **backend_params):
                    sl_conf_hpo['estimators'][oname][ename]['model'] = einfo['hpo'].fit(X_train, Y_train[:, oi]).best_estimator_

        sl_conf = sl_conf_hpo

    # Define SuperLearners:
    SuperLearners = {}
    for oi, oname in enumerate(onames):
        print('Defining estimator for output: ' + oname, flush = True)

        if oname in sl_conf['estimators']:
            estimators = sl_conf['estimators'][oname]
        else:
            estimators = sl_conf['estimators']

        final_estimator = sl_conf['final_estimator']
        if type(final_estimator) == dict:
            if 'oname' in sl_conf['final_estimator']:
                final_estimator = sl_conf['final_estimator'][oname]


        SuperLearners[oname] = StackingRegressor(
            estimators = format_estimators(estimators),
            final_estimator = final_estimator,
            n_jobs = n_jobs
        )

    # Fit SuperLearners:
    for oi, oname in enumerate(onames):
        print('Training estimator for output: ' + oname, flush = True)
        with joblib.parallel_backend(args.backend, **backend_params):
            SuperLearners[oname] = SuperLearners[oname].fit(X_train, Y_train[:, oi])


    with open(args.model_dir + '/SuperLearners.pkl', 'wb') as output:
        pickle.dump(SuperLearners, output, pickle.HIGHEST_PROTOCOL)

    # Cross_val_score:
    if args.cross_val_score == "True":
        cross_val_metrics = {}
        for oi, oname in enumerate(onames):
            cross_val_metrics[oname] = dict.fromkeys(['all', 'mean', 'std'])
            # FIXME: dask bug with cross_val_score!
            with joblib.parallel_backend('threading', **{}):
                scores = cross_val_score(
                    deepcopy(SuperLearners[oname]),
                    X,
                    y = Y[:, oi],
                    n_jobs = n_jobs
                )

            cross_val_metrics[oname]['all'] = list(scores)
            cross_val_metrics[oname]['mean'] = scores.mean()
            cross_val_metrics[oname]['std'] = scores.std()

        print('Cross-validation metrics:', flush = True)
        print(json.dumps(cross_val_metrics, indent = 4), flush = True)
        with open(args.model_dir + '/cross-val-metrics.json', 'w') as json_file:
            json.dump(cross_val_metrics, json_file, indent = 4)
        print('Statistics of the cross-validation metrics:')



    # Evaluate SuperLearners on test set:
    ho_metrics = {}
    for oi, oname in enumerate(onames):
        print('Evaluating estimator for output: ' + oname, flush = True)
        with joblib.parallel_backend(args.backend, **backend_params):
            ho_metrics[oname] = SuperLearners[oname].score(X_test, Y_test[:, oi])

    print('Hold out metrics:', flush = True)
    print(json.dumps(ho_metrics, indent = 4), flush = True)
    with open(args.model_dir + '/hold-out-metrics.json', 'w') as json_file:
        json.dump(ho_metrics, json_file, indent = 4)

    #============================================================
    # Evaluate SuperLearners on training set ("classical validation"):
    ho_metrics = {}
    for oi, oname in enumerate(onames):
        print('Evaluating estimator for output: ' + oname, flush = True)
        with joblib.parallel_backend(args.backend, **backend_params):
            ho_metrics[oname] = SuperLearners[oname].score(X_train, Y_train[:, oi])

    print('Classical metrics:', flush = True)
    print(json.dumps(ho_metrics, indent = 4), flush = True)
    with open(args.model_dir + '/classical-metrics.json', 'w') as json_file:
        json.dump(ho_metrics, json_file, indent = 4)
    #=========================================================

    # For debugging
    if args.backend == 'dask':
        shutil.move(dask_log_dir, args.model_dir)

    #============================================================
    # Save the training and testing data seprately for evaluation
    # (They are randomly split above.)
    print('Save train/test data')
    save_data_csv_io(X_train, Y_train, inames, onames, args.model_dir+'/train.csv')
    save_data_csv_io(X_test, Y_test, inames, onames, args.model_dir+'/test.csv')
