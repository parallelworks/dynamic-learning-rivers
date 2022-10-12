import os
import parsl
from parsl.app.app import bash_app
from parsl.data_provider.files import File

@bash_app
def run_workflow(command, inputs = [], outputs = [], stdout = 'std.out', stderr = 'std.err'):
    return command


def add_parsl_files(workflow_conf):
    cwd = os.getcwd()
    prefix = 'file://parslhost/'
    files = {}

    def parsl_file(path):
        if path.startswith('/'):
            return File(prefix + path)
        else:
            return File(prefix + cwd + '/' + path)

    for iname, inp in workflow_conf['inputs'].items():
        if inp['type'] == 'file':
            workflow_conf['inputs'][iname]['parsl_file'] = parsl_file(inp['origin'])

    for oname, out in workflow_conf['outputs'].items():
        if out['type'] == 'file':
            workflow_conf['outputs'][oname]['parsl_file'] = parsl_file(out['destination'])

    return workflow_conf

def get_workflow_files(io):
    files = []
    for pname,param in io.items():
        if 'parsl_file' in param:
            files.append(param['parsl_file'])
    return files


# Gets the file path in the remote host
def remote_path(p):
    if hasattr(p, 'path'):
        if "/./" in p.path: # p.filename in /path/./to/file gives file instead of ./to/file
            return "\"" + p.path.split("/./")[-1] + "\""
        else:
            if p.path.endswith('/'):
                return "\"" + os.path.basename(p.path[:-1]) + "\""
            else:
                return "\"" + os.path.basename(p.path) + "\""
    else:
        return "\"" + str(p) + "\""

# Define the workflow command from the workflow configuration
def get_command(workflow_conf):

    def get_command_arg(io):
        if io['type'] == 'text':
            return io['value']
        elif io['type'] == 'file':
            return remote_path(io['parsl_file'])
        else:
            raise(Exception('IO type must be text or file'))

    command = workflow_conf['command']

    for pkey, param in {**workflow_conf['inputs'], **workflow_conf['outputs']}.items():
        arg = get_command_arg(param)
        if arg:
            command += ' --' + pkey + ' ' + arg

    return command

