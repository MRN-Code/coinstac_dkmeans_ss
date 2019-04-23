"""
Remote file for single-shot KMeans
"""

import os
import sys
import json
import logging
import configparser
import numpy as np
import remote_computations as remote
import local_computations as local


CONFIG_FILE = 'config.cfg'
DEFAULT_k = 5
DEFAULT_epsilon = 0.00001
DEFAULT_shuffle = True
DEFAULT_learning_rate = 0.001
DEFAULT_verbose = True
DEFAULT_optimization = 'lloyd'


def remote_init_env(config_file=DEFAULT_config_file, k=DEFAULT_k,
                    optimization=DEFAULT_optimization, epsilon=DEFAULT_epsilon, learning_rate=DEFAULT_learning_rate,
                    verbose=DEFAULT_verbose):
    """
        # Description:
            Initialize the remote environment, creating the config file.

        # PREVIOUS PHASE:
            None

        # INPUT:

            |   name            |   type    |   default     |
            |   ---             |   ---     |   ---         |
            |   config_file     |   str     |   config.cfg  |
            |   k               |   int     |   5           |
            |   optimization    |   str     |   lloyd       |
            |   epsilon         |   float   |   0.00001     |
            |   shuffle         |   bool    |   False       |
            |   data_file       |   str     |   data.txt    |
            |   learning_rate   |   float   |   0.001       |
            |   verbose         |   float   |   True        |

        # OUTPUT:
            - config file written to disk
            - k
            - learning_rate
            - optimization
            - shuffle

        # NEXT PHASE:
            local_init_env
    """

    logging.info('REMOTE: Initializing remote environment')
    if not os.path.exists(config_file):
        config = configparser.ConfigParser()
        config['REMOTE'] = dict(k=k, optimization=optimization, epsilon=epsilon,
                                learning_rate=learning_rate, verbose=verbose)
        with open(config_path, 'w') as file:
            config.write(file)
    # output
    computation_output = dict(output=
                              dict(
                                  work_dir=work_dir,
                                  config_file=config_file,
                                  k=k,
                                  learning_rate=learning_rate,
                                  optimization=optimization,
                                  shuffle=shuffle,
                                  computation_phase="remote_init_env"
                                  )
                              )
    return json.dumps(computation_output)


def remote_init_centroids(args, config_file=DEFAULT_config_file):
    """
        # Description:
            Initialize K centroids from locally selected centroids.

        # PREVIOUS PHASE:
            local_init_centroids

        # INPUT:

            |   name             |   type    |   default     |
            |   ---              |   ---     |   ---         |
            |   config_file      |   str     |   config.cfg  |

        # OUTPUT:
            - centroids: list of numpy arrays

        # NEXT PHASE:
            local_compute_optimizer
    """
    logging.info('REMOTE: Initializing centroids')
    # Have each site compute k initial clusters locally
    local_centroids = [cent for site in args for cent in
                       args[site]]
    # and select k random clusters from the s*k pool
    np.random.shuffle(local_centroids)
    remote_centroids = local_centroids[:k]
    computation_output = dict(
        output=dict(
            work_dir=work_dir,
            config_file=config_file,
            centroids=remote_centroids,
            computation_phase="remote_init_centroids"
            ),
        success=True
    )
    return json.dumps(computation_output)


def remote_check_convergence(args, config_file=CONFIG_FILE):
    """
        # Description:
            Check convergence.

        # PREVIOUS PHASE:
            local_check_convergence

        # INPUT:

            |   name               |   type    |   default     |
            |   ---                |   ---     |   ---         |
            |   config_file        |   str     |   config.cfg  |
            |   remote_centroids   |   list    |   config.cfg  |
            |   previous_centroids |   list    |   config.cfg  |

        # OUTPUT:
            - boolean encoded in name of phase

        # NEXT PHASE:
            local_compute_clustering?
    """
    logging.info('REMOTE: Check convergence')
    config = configparser.ConfigParser()
    config.read(config_file)
    local_check = [site['local_check'] for site in args]
    remote_check = any(local_check)
    new_phase = "remote_converged_true" if remote_check else "remote_converged_false"
    computation_output = dict(
        output=dict(
            computation_phase=new_phase
            ),
        success=True
    )
    return json.dumps(computation_output)


def remote_aggregate_output(args):
    """
        Aggregate output. TODO: What needs to be aggregated
    """
    logging.info('REMOTE: Aggregating input')
    computation_output = dict(
        output=dict(
            computation_phase="remote_aggregate_output"
            ),
        success=True
    )
    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = remote_init_env(**parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'remote_aggregate_otpimizer' in phase_key:
        computation_output = remote_optimization_step(parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'local_check_convergence' in phase_key:
        computation_output = remote_check_convergence(parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'remote_converged_true' in phase_key:
        computation_outut = remote_aggregate_output(parsed_args['input'])
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Oops')

