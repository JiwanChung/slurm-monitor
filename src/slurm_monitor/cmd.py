#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
from typing import Optional

from slurm_monitor.parse import parse_user

CMDS = {
    'nodes': 'scontrol show nodes',
    'user': 'squeue',
    'job': 'scontrol show job {}'
}


def run_cmd(cmd) -> Optional[str]:
    try:
        return _run_cmd(cmd)
    except Exception as e:
        print(e)
        return None


def _run_cmd(cmd) -> str:
    """Parse the output of a shell command...
     and if split set to true: split into a list of strings, one per line of output.

    Args:
        cmd (str): the shell command to be executed.
    Returns:
        (list[str]): the strings from each output line.
    """
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    return output


def get_data():
    return run_cmd(CMDS['nodes'])


def get_user():
    user = run_cmd(CMDS['user'])
    if user is None:
        return []
    jobs = parse_user(user)
    outs = []
    for job in jobs:
        out = run_cmd(CMDS['job'].format(job))
        if out is not None:
            outs.append(out)
    return outs
