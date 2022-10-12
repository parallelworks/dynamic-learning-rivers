import os
import json
from random import randint
from subprocess import Popen, PIPE, STDOUT
import traceback
import requests
import argparse
import sys

from . import pool_info


# Determine the workflow host. Can be empty (local) or in the format user@ip
def add_workflow_host(args):
    if 'PW_USER' in os.environ and 'PW_API_KEY' in os.environ:
        # Running in remote host from PW
        args['workflow_host'] =  os.environ['PW_USER'] + '@' + pool_info.get_master_node_ip()
        # Make sure host is not in known_hosts
        try:
            run_and_log_cmd("ssh-keygen -f \"/home/" + os.environ['PW_USER'] + "/.ssh/known_hosts\" -R " + pool_info.get_master_node_ip())
        except:
            pass
        return args
    elif 'workflow_host' in args:
        # Running in remote host from local host
        return args
    else:
        # Running in local host from local host
        args['workflow_host'] = ''
        return args


# Define the workflow command from the workflow configuration
def get_command(workflow_conf):

    def get_command_arg(io, input = True):
        if io['type'] == 'text':
            return io['value']
        elif io['type'] == 'file':
            if input:
                return io['destination']
            else:
                return io['origin']
        else:
            raise(Exception('IO type must be text or file'))

    command = workflow_conf['command']
    if 'workflow_host' in workflow_conf:
        if workflow_conf['workflow_host']:
            command = 'ssh ' + workflow_conf['workflow_host'] + ' ' + command


    for inp_key, inp in workflow_conf['inputs'].items():
        arg = get_command_arg(inp, input = True)
        if arg:
            command += ' --' + inp_key + ' ' + arg

    for out_key, out in workflow_conf['outputs'].items():
        arg = get_command_arg(out, input = False)
        if arg:
            command += ' --' + out_key + ' ' + arg

    return command

# Run and log a command using Popen
def run_and_log_cmd(cmd):
    cmd = cmd.rstrip()
    cmd = [cmd for cmd in cmd.split(' ') if cmd]
    print('\n\nRunning command:\n' + " ".join(cmd), flush = True)
    process = Popen(cmd, stdout = PIPE, stderr = PIPE)

    # Initialize logging loop:
    for line in iter(process.stdout.readline, '\n'):
        if process.poll() == None:
            line_str = line.decode('utf-8').rstrip()
            print(line_str)
        elif process.poll() == 0:
            print('Command finished successfully', flush = True)
            success = True
            break
        else:
            msg = 'Command failed or killed'
            print(msg, flush = True)
            success = False
            raise(Exception(msg))

    for line in process.stdout.readlines():
        print(line.decode('utf-8').rstrip(), flush = True)

    for line in process.stderr.readlines():
        print(line.decode('utf-8').rstrip(), flush = True)

    return success


# Stage files to and from the workflow host and workflow directory
# FIXME: Improve documentation
def stage_files(io, origin_host = '', destination_host = ''):
    mkdir_cmd = 'mkdir -p '
    if not origin_host and not destination_host:
        cp_cmd = 'cp -r '
    else:
        cp_cmd = 'scp -r '
        if origin_host:
            origin_host += ':'
        if destination_host:
            mkdir_cmd =  'ssh ' + destination_host + ' ' + mkdir_cmd
            destination_host += ':'

    stage_commands = []
    for ioval in io.values():
        if ioval['type'] == 'file':
            if '{{' in ioval['origin'] or '{{' in ioval['destination']:
                # File is assume to be an optional parameter that was not replaced
                continue

            if ioval['origin'].startswith('~'):
                ioval['origin'] = ioval['origin'].replace('~', '/home/' + os.environ['PW_USER'])

            if ioval['destination'].startswith('~'):
                ioval['destination'] = ioval['destination'].replace('~', '/home/' + os.environ['PW_USER'])

            stage_commands.append(mkdir_cmd + os.path.dirname(ioval['destination']).replace('../',''))
            stage_commands.append(cp_cmd + origin_host + ioval['origin'] + ' ' + destination_host + ioval['destination'])

    for cmd in stage_commands:
        run_and_log_cmd(cmd)
