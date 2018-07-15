import json
import sys
from utils import listRecursive


def local_1(args):
    input_list = args["input"]

    output_dict = {
        "Name": input_list["Person_Name"],
        "computation_phase": "local_1"
    }
    cache_dict = {}

    computation_output = {"output": output_dict, "cache_dict": cache_dict}

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
