import os
import json
import workflow_json as wjson
import sys
from random import randint

# To run with simlink
sys.path.append(os.getcwd())

# Decide whether to use Parsl or plain Python
if not os.path.isfile('pw.conf'):
    # There is no need to use parsl when running locally
    use_parsl = False
elif wjson.pool_info.is_coaster_pool():
    use_parsl = True
    import parsl
    from parsl.app.app import bash_app
    from parsl.data_provider.files import File
    from parslpw import pwconfig
else:
    use_parsl = False

if __name__ == '__main__':
    # Load arguments
    args = wjson.general.read_command_line_args()

    # Check for predefined tests
    if os.path.isfile('tests.py'):
        if 'sanity_test' in args:
            if not args['sanity_test']:
                args['sanity_test'] = 'None'
            if args['sanity_test'] != 'None':
                import tests
                args = tests.get_test_pwargs(args['sanity_test'])

    if use_parsl:
        parsl.load(pwconfig)
        # Remote directory is chosen for you: /tmp/parsl-task-
        args['remote_dir'] = './'
        args['workflow_host'] = ''
    else: # Plain python
        # Add sandbox directory
        if 'remote_dir' in args:
            args['remote_dir'] += '/run-' + str(randint(0, 99999)).zfill(5)
        else:
            args['remote_dir'] = '/tmp/run-' + str(randint(0, 99999)).zfill(5)
        # Add workflow host to arguments
        args = wjson.plain.add_workflow_host(args)

    print('Command line arguments:', flush = True)
    print(args, flush = True)

    # Load and complete workflow configuration
    with open('workflow.json') as f:
        workflow_conf = wjson.general.complete_workflow_conf(json.load(f), args)

    print(json.dumps(workflow_conf, indent = 4), flush = True)

    if use_parsl:
        # Get workflow command and files:
        # - In parsl staging is different so command is also different
        workflow_conf = wjson.parsl.add_parsl_files(workflow_conf)
        print(workflow_conf, flush = True)
        workflow_cmd = wjson.parsl.get_command(workflow_conf)
    else:
        # Get workflow command:
        workflow_cmd = wjson.plain.get_command(workflow_conf)

    print('Workflow command', flush = True)
    print(workflow_cmd, flush = True)

    if use_parsl:
        # Run workflow command
        workflow_fut = wjson.parsl.run_workflow(
            workflow_cmd,
            inputs = wjson.parsl.get_workflow_files(workflow_conf['inputs']),
            outputs = wjson.parsl.get_workflow_files(workflow_conf['outputs'])
        )
        workflow_fut.result()
    else:
        # Stage input files:
        wjson.plain.stage_files(
            workflow_conf['inputs'],
            destination_host = workflow_conf['workflow_host']
        )

        # Run command:
        wjson.plain.run_and_log_cmd(workflow_cmd)

        # Stage output files
        wjson.plain.stage_files(
            workflow_conf['outputs'],
            origin_host = workflow_conf['workflow_host']
        )
