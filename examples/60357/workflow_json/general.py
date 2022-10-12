import json
import argparse

# Reads command line arguments in the format:
# python main.py --argkey1 argval1 --argkey2 argval2
# Returns a dictionary args in the format:
# args[argkey1] = argval1
def read_command_line_args():
    # GET COMMAND LINE ARGS FROM PW FORM
    parser = argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
    args = vars(parser.parse_args())
    print("\nargs:", flush=True)
    print(json.dumps(args, indent = 4), flush = True)
    return args

# Replace {{argk}} by argv everywhere in the workflow configuration
def complete_workflow_conf(workflow_conf, args):
    if type(workflow_conf) == list:
        for i,wfv in enumerate(workflow_conf):
            workflow_conf[i] = complete_workflow_conf(wfv, args)

    elif type(workflow_conf) == dict:
        for wfk, wfv in workflow_conf.items():
            workflow_conf[wfk] = complete_workflow_conf(wfv, args)

    elif type(workflow_conf) == str:
        for argk, argv in args.items():
            workflow_conf = workflow_conf.replace('{{' + argk + '}}', str(argv))
        return workflow_conf

    return workflow_conf
