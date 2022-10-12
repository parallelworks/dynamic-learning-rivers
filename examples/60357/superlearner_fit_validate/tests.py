
def get_test_pwargs(test):
    pwargs = {}
    if test == 'fit_mimosuperlearner_with_hold_out_converge':
        pwargs['backend'] = 'dask'
        pwargs['remote_dir'] = '/tmp'
        pwargs['conda_sh'] = '/contrib/Alvaro.Vidal/miniconda3/etc/profile.d/conda.sh'
        pwargs['conda_env'] = 'base'
        pwargs['superlearner_conf'] = './superlearner_fit_validate/sample_io/converge/superlearner_conf.py'
        pwargs['data'] = './superlearner_fit_validate/sample_io/converge/data250.csv'
        pwargs['num_inputs'] = '9'
        pwargs['model_dir'] = './superlearner_fit_validate/sample_io/converge/ConvergeSL'
        pwargs['n_jobs'] = 4
        pwargs['hpo'] = "True"
        pwargs['cross_val_score'] = "True"
    elif test == 'fit_mimosuperlearner_with_hold_out_converge_parsl':
        pwargs['backend'] = 'threading'
        pwargs['remote_dir'] = '/tmp'
        pwargs['conda_sh'] = '/tmp/pworks/.miniconda3/etc/profile.d/conda.sh'
        pwargs['conda_env'] = 'parsl-pw'
        pwargs['superlearner_conf'] = './superlearner_fit_validate/sample_io/converge/superlearner_conf.py'
        pwargs['data'] = './superlearner_fit_validate/sample_io/converge/data250.csv'
        pwargs['num_inputs'] = '9'
        pwargs['model_dir'] = './superlearner_fit_validate/sample_io/converge/ConvergeSL'
        pwargs['n_jobs'] = 4
        pwargs['hpo'] = "False"
        pwargs['cross_val_score'] = "True"
    else:
        raise('Test {} not found!'.format(test))

    return pwargs
