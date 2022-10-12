# Python/JSON Workflow Framework
Uses a `workflow.json` file to define a workflow. There are several options to execute the workflow:
1. On a _pasive_ or _cluster_ pool. The workflow is executed using `ssh` and `scp` commands on the head node of the cluster.
2. On an _active_ or _coater_ pool. The workflow is executed using a single Parsl `bash_app` on a single node of the pool. Compatible but not described in the rest of this document.
3. On a local machine

# Installation: Key files and what they do

1. Clone this repository into an empty PW workflow directory (the "top level directory").
2. Create symbolic link in the top level directory to `remotepacks` in the local clone of this repo.  (Note if the workflow needs other packages in remotepacks, `cat` rather than `ln`.)
3. **COPY** `main.py` from this directory into the top level directory.  Copy required b/c [python execution is transfered to origin of link](https://stackoverflow.com/questions/19643223/launching-a-python-script-via-a-symbolic-link).
4. Clone the workflow app into the top level directory.
5. Create symbolic links in the top level directory to `workflow.xml` and `workflow.json` in the workflow app.

top level dir       workflow modules/apps

main.py----ln-->|
remotepacks---->|   workflow_json
                |
-------------------------------------------
                |
workflow.xml--->|   app_folder
workflow.json-->|

# Workflow structure

The `workflow.json` has the following keys:

```
{
    'workflow_host': 'Empty or user@ip for local or remote execution, respectively',
    'command': 'Command to execute directly on the master node',
    'inputs': {
        'input_file_key': {
            'type': 'file',
            'origin': 'path to the input file in origin (typically local machine)',
            'destination': 'path to the input file in destination (typically remote machine)'
        },
        'input_text_key': {
            'type': 'text',
            'value': 'input_text_value'
        }
    },
    'outputs': {
        'output_file_key': {
            'type': 'file',
            'destination': 'path to the output file in origin (typically local machine)',
            'origin': 'path to the output file in destination (typically remote machine)'
        }
    }
}
```

And can be templated using double braces `{{}}`, for example:

```
{
    'workflow_host': '{{workflow_host}}',
    'command': 'python {{remote_dir}}/run.py',
    'inputs': {
        'script': {
            'type': 'file',
            'origin': './run.py',
            'destination': '{{remote_dir}}/run.py'
        },
        'arg1': {
            'type': 'text',
            'value': '{{arg1}}'
        }
    },
    'outputs': {
        'result': {
            'type': 'file',
            'destination': '{{output_path}}',
            'origin': '{{remote_dir}}/result.txt'
        }
    }
}
```

The values between braces will be replaced by the corresponding command line arguments passed to the `main.py` script. For example, the command below will replace `{{arg1}}`, `{{output_path}}` and `{{remote_dir}}` by `arg1_value`, `results.txt` and `/tmp`, respetively, everywhere in the `workflow.json` file.

```
python main.py --arg1 arg1_value --output_path results.txt --remote_dir /tmp
```

Note that when you execute the workflow by clicking the `execute` button in the PW platform the platform will generate this command for you using the `workflow.xml` file.


**Special input keys:**

These argument keys are treated in a special way by the `main.py`:

- The `workflow_host` is a special key that is replaced by the value corresponding to the selected pool if the workflow is executed from the platform. Otherwise, the user must pass the corresponding `user@ip` (and have ssh access to the host) or pass no argument and the workflow is executed in the local machine.
- A sandbox subdirectory is added to the `remote_key` argument. In the example above: `/tmp/run-12345`, where `12345` is a randomly generate integer.
- If the 'sanity_test' key aregument is present the corresponding test is loaded from a `tests.py` script in the workflow directory.


### Steps for local or _cluster_ pool execution:

The `main.py` script goes through the following steps in the `if __name__ == '__main__':` part of the script:

##### 1. Load command line arguments
The command line arguments passed to the `main.py` script are loaded into a python dictionary. For example, the dictionary corresponding to this command:

```
python main.py -argkey1 argval1 --argkey2 argval2
```

is:

```
args = {
    'argkey1': 'argval1',
    'argkey2': 'argval2'
}
```

##### 2. If a sanity test is specified load its corresponding arguments
If a sanity test is specified the arguments dictionary is replaced by the one corresponding to the test.


##### 3. Add sandbox directory to the `remote_dir` argument.
A sandbox subdirectory is added to the `remote_key` argument. In the example above: `/tmp/run-12345`, where `12345` is a randomly generate integer.

##### 4. If running in PW: get the `workflow_host` from the pool
If the environmental variables `PW_USER` and `PW_API_KEY` are detected the value  of the `workflow_host` key is assigned to be `PW_USER@HEAD_NODE_IP`. Otherwise, if no value was passed through the command line it assignes an empty value to this key indicating that the workflow runs in the local machine.

##### 5. Load and complete workflow configuration:
The workflow configuration is loaded and the braced values {{}} are replaced using the command line arguments.

##### 6. Generate the workflow command using the `command` key in the `workflow.json`:
The workflow command is generated using the workflow configuration.

```
ssh PW_USER@HEAD_NODE_IP python /tmp/run-12345/run.py --script /tmp/run-12345/run.py --arg1 arg1_value --result /tmp/run-12345/result.txt
````

Or if the worflow is run in the local machine:

```
python /tmp/run-12345/run.py --script /tmp/run-12345/run.py --arg1 arg1_value --result /tmp/run-12345/result.txt
````

##### 7. Stage input files
Each input file in the `inputs` key of the workflow configuration is copied to the specified destination using a `cp` or `scp` command if the workflow host is local or remote, respectively. This command is generated using the `origin` and `destination` keys together with the `workflow_host` information.

##### 8. Run the command
The command is executed using the function `Popen`. The standard output and errors are logged.

##### 9. Stage output files
The output files are staged with the same function as the input files.
