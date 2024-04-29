#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Credits:
- https://github.com/TengdaHan/slurm_web
- https://github.com/albanie/slurm_gpustat
"""

import re
import subprocess
from collections import defaultdict
from typing import Optional

import humanize
import humanfriendly as hf

INACCESSIBLE = {"drain*", "down*", "drng", "drain", "down"}


def all_info(verbose: bool, partition: Optional[str] = None):
    """Print a collection of summaries about SLURM gpu usage, including: all nodes
    managed by the cluster, nodes that are currently accesible and gpu usage for each
    active user.

    Args:
        partition: the partition/queue (or multiple, comma separated) of interest.
            By default None, which queries all available partitions.
    """
    resources = parse_all_gpus(partition=partition)
    states = node_states(partition=partition)
    summaries = {}
    for mode in ("up", "accessible"):
        summaries[mode] = summary(mode=mode,
                                  resources=resources,
                                  states=states)
    summaries['in_use'] = in_use(resources, partition=partition)
    summaries['available'] = available(resources=resources,
                                       states=states,
                                       verbose=verbose)
    return summaries


def parse_all_gpus(partition: Optional[str] = None,
                   default_gpus: int = 4,
                   default_gpu_name: str = "NONAME_GPU") -> dict:
    """Query SLURM for the number and types of GPUs under management.

    Args:
        partition: the partition/queue (or multiple, comma separated) of interest.
            By default None, which queries all available partitions.
        default_gpus: The number of GPUs estimated for nodes that have incomplete SLURM
            meta data.
        default_gpu_name: The name of the GPU for nodes that have incomplete SLURM meta
        data.

    Returns:
        a mapping between node names and a list of the GPUs that they have available.
    """
    cmd = "sinfo -o '%1000N|%1000G' --noheader"
    if partition:
        cmd += f" --partition={partition}"
    rows = parse_cmd(cmd)
    resources = defaultdict(list)

    # Debug the regular expression below at
    # https://regex101.com/r/RHYM8Z/3
    p = re.compile(r'gpu:(?:(\w*):)?(\d*)(?:\(\S*\))?\s*')

    for row in rows:
        node_str, resource_strs = row.split("|")
        for resource_str in resource_strs.split(","):
            if not resource_str.startswith("gpu"):
                continue
            match = p.search(resource_str)
            gpu_type = match.group(1) if match.group(
                1) is not None else default_gpu_name
            # if the number of GPUs is not specified, we assume it is `default_gpus`
            gpu_count = int(
                match.group(2)) if match.group(2) != "" else default_gpus
            node_names = parse_node_names(node_str)
            for name in node_names:
                resources[name].append({"type": gpu_type, "count": gpu_count})
    return resources


def parse_cmd(cmd, split=True):
    """Parse the output of a shell command...
     and if split set to true: split into a list of strings, one per line of output.

    Args:
        cmd (str): the shell command to be executed.
        split (bool): whether to split the output per line
    Returns:
        (list[str]): the strings from each output line.
    """
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    if split:
        output = [x for x in output.split("\n") if x]
    return output


def split_node_str(node_str):
    """Split SLURM node specifications into node_specs. Here a node_spec defines a range
    of nodes that share the same naming scheme (and are grouped together using square
    brackets).   E.g. 'node[1-3,4,6-9]' represents a single node_spec.

    Examples:
       A `node_str` of the form 'node[001-003]' will be returned as a single element
           list: ['node[001-003]']
       A `node_str` of the form 'node[001-002],node004' will be split into
           ['node[001-002]', 'node004']

    Args:
        node_str (str): a SLURM-formatted list of nodes

    Returns:
        (list[str]): SLURM node specs.
    """
    node_str = node_str.strip()
    breakpoints, stack = [0], []
    for ii, char in enumerate(node_str):
        if char == "[":
            stack.append(char)
        elif char == "]":
            stack.pop()
        elif not stack and char == ",":
            breakpoints.append(ii + 1)
    end = len(node_str) + 1
    return [
        node_str[i:j - 1] for i, j in zip(breakpoints, breakpoints[1:] + [end])
    ]


def parse_node_names(node_str):
    """Parse the node list produced by the SLURM tools into separate node names.

    Examples:
       A slurm `node_str` of the form 'node[001-003]' will be split into a list of the
           form ['node001', 'node002', 'node003'].
       A `node_str` of the form 'node[001-002],node004' will be split into
           ['node001', 'node002', 'node004']

    Args:
        node_str (str): a SLURM-formatted list of nodes

    Returns:
        (list[str]): a list of separate node names.
    """
    names = []
    node_specs = split_node_str(node_str)
    for node_spec in node_specs:
        if "[" not in node_spec:
            names.append(node_spec)
        else:
            head, tail = node_spec.index("["), node_spec.index("]")
            prefix = node_spec[:head]
            subspecs = node_spec[head + 1:tail].split(",")
            for subspec in subspecs:
                if "-" not in subspec:
                    subnames = [f"{prefix}{subspec}"]
                else:
                    start, end = subspec.split("-")
                    num_digits = len(start)
                    subnames = [
                        f"{prefix}{str(x).zfill(num_digits)}"
                        for x in range(int(start),
                                       int(end) + 1)
                    ]
                names.extend(subnames)
    return names


def node_states(partition: Optional[str] = None) -> dict:
    """Query SLURM for the state of each managed node.

    Args:
        partition: the partition/queue (or multiple, comma separated) of interest.
            By default None, which queries all available partitions.

    Returns:
        a mapping between node names and SLURM states.
    """
    cmd = "sinfo --noheader"
    if partition:
        cmd += f" --partition={partition}"
    rows = parse_cmd(cmd)
    states = {}
    for row in rows:
        tokens = row.split()
        state, names = tokens[4], tokens[5]
        node_names = parse_node_names(names)
        states.update({name: state for name in node_names})
    return states


def summary(mode: str, resources: dict = None, states: dict = None):
    """Generate a printed summary of the cluster resources.

    Args:
        mode (str): the kind of resources to query (must be one of 'accessible', 'up').
        resources (dict :: None): a summary of cluster resources, organised by node name.
        states (dict[str: str] :: None): a mapping between node names and SLURM states.
    """
    if not resources:
        resources = parse_all_gpus()
    if not states:
        states = node_states()
    if mode == "accessible":
        res = {
            key: val
            for key, val in resources.items()
            if states.get(key, "down") not in INACCESSIBLE
        }
    elif mode == "up":
        res = resources
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return summary_by_type(res)


def resource_by_type(resources: dict) -> dict:
    """Determine the cluster capacity by gpu type

    Args:
        resources: a summary of the cluster resources, organised by node name.

    Returns:
        resources: a summary of the cluster resources, organised by gpu type
    """
    by_type = defaultdict(list)
    for node, specs in resources.items():
        for spec in specs:
            by_type[spec["type"]].append({
                "node": node,
                "count": spec["count"]
            })
    return by_type


def summary_by_type(resources: dict, tag: str):
    """Print out out a summary of cluster resources, organised by gpu type.

    Args:
        resources (dict): a summary of cluster resources, organised by node name.
        tag (str): a term that will be included in the printed summary.
    """
    summary = {}
    by_type = resource_by_type(resources)
    total = sum(x["count"] for sublist in by_type.values() for x in sublist)
    summary['total'] = {'count': total}
    summary['types'] = {}
    for key, val in sorted(by_type.items(),
                           key=lambda x: sum(y["count"] for y in x[1])):
        gpu_count = sum(x["count"] for x in val)
        summary['types'][key] = {'count': gpu_count}
    return summary


def in_use(resources: dict = None, partition: Optional[str] = None):
    """Print a short summary of the resources that are currently used by each user.

    Args:
        resources: a summary of cluster resources, organised by node name.
    """
    summary = {}
    if not resources:
        resources = parse_all_gpus()
    usage = gpu_usage(resources, partition=partition)
    aggregates = {}
    for user, subdict in usage.items():
        aggregates[user] = {}
        aggregates[user]['n_gpu'] = {
            key: sum([x['n_gpu'] for x in val.values()])
            for key, val in subdict.items()
        }
        aggregates[user]['bash_gpu'] = {
            key: sum([x['bash_gpu'] for x in val.values()])
            for key, val in subdict.items()
        }
    for user, subdict in sorted(aggregates.items(),
                                key=lambda x: sum(x[1]['n_gpu'].values())):

        summary[user] = {
            'total': sum(subdict['n_gpu'].values()),
            'interactive': str(sum(subdict['bash_gpu'].values())),
            'details': summary_str
        }
    return summary


def gpu_usage(resources: dict, partition: Optional[str] = None) -> dict:
    """Build a data structure of the cluster resource usage, organised by user.

    Args:
        resources (dict :: None): a summary of cluster resources, organised by node name.

    Returns:
        (dict): a summary of resources organised by user (and also by node name).
    """
    version_cmd = "sinfo -V"
    slurm_version = parse_cmd(version_cmd, split=False).split(" ")[1]
    if slurm_version.startswith("17"):
        resource_flag = "gres"
    else:
        resource_flag = "tres-per-node"
    if int(slurm_version[0:2]) >= 21:
        gpu_identifier = 'gres:gpu'
    else:
        gpu_identifier = 'gpu'

    cmd = f"squeue -O {resource_flag}:100,nodelist:100,username:100,jobid:100 --noheader"
    if partition:
        cmd += f" --partition={partition}"
    detailed_job_cmd = "scontrol show jobid -dd %s"
    rows = parse_cmd(cmd)
    usage = defaultdict(dict)
    for row in rows:
        tokens = row.split()
        # ignore pending jobs
        if len(tokens) < 4 or not tokens[0].startswith(gpu_identifier):
            continue
        gpu_count_str, node_str, user, jobid = tokens
        gpu_count_tokens = gpu_count_str.split(":")
        if not gpu_count_tokens[-1].isdigit():
            gpu_count_tokens.append("1")
        num_gpus = int(gpu_count_tokens[-1])
        # get detailed job information, to check if using bash
        detailed_output = parse_cmd(detailed_job_cmd % jobid, split=False)
        is_bash = any(
            [f'Command={x}\n' in detailed_output for x in INTERACTIVE_CMDS])
        num_bash_gpus = num_gpus * is_bash
        node_names = parse_node_names(node_str)
        for node_name in node_names:
            # If a node still has jobs running but is draining, it will not be present
            # in the "available" resources, so we ignore it
            if node_name not in resources:
                continue
            node_gpu_types = [x["type"] for x in resources[node_name]]
            if (len(gpu_count_tokens) == 2) or (int(slurm_version[0:2]) >= 21):
                gpu_type = None
            elif len(gpu_count_tokens) == 3:
                gpu_type = gpu_count_tokens[1]
            if gpu_type is None:
                if len(node_gpu_types) != 1:
                    gpu_type = sorted(resources[node_name],
                                      key=lambda k: k['count'],
                                      reverse=True)[0]['type']
                    msg = (
                        f"cannot determine node gpu type for {user} on {node_name}"
                        f" (guessing {gpu_type})")
                    print(f"WARNING >>> {msg}")
                else:
                    gpu_type = node_gpu_types[0]
            if gpu_type in usage[user]:
                usage[user][gpu_type][node_name]['n_gpu'] += num_gpus
                usage[user][gpu_type][node_name]['bash_gpu'] += num_bash_gpus

            else:
                usage[user][gpu_type] = defaultdict(lambda: {
                    'n_gpu': 0,
                    'bash_gpu': 0
                })
                usage[user][gpu_type][node_name]['n_gpu'] += num_gpus
                usage[user][gpu_type][node_name]['bash_gpu'] += num_bash_gpus

    return usage


def occupancy_stats_for_node(node: str) -> dict:
    """Query SLURM for the occupancy of a given node.

    Args:
        (node): the name of the node to query

    Returns:
        a mapping between node names and occupancy stats.
    """
    cmd = f"scontrol show node {node}"
    rows = [x.strip() for x in parse_cmd(cmd)]
    keys = ("AllocTRES", "CfgTRES")
    metrics = {}
    for row in rows:
        for key in keys:
            if row.startswith(key):
                row = row.replace(f"{key}=", "")
                tokens = row.split(",")
                if tokens == [""]:
                    # SLURM sometimes omits information, so we alert the user to its
                    # its exclusion and report nothing for this node
                    print(
                        f"Missing information for {node}: {key}, skipping....")
                    metrics[key] = {}
                else:
                    metrics[key] = {
                        x.split("=")[0]: x.split("=")[1]
                        for x in tokens
                    }
    occupancy = {}
    for metric, alloc_val in metrics["AllocTRES"].items():
        cfg_val = metrics["CfgTRES"][metric]
        if metric == "mem":
            # SLURM appears to sometimes misformat large numbers, producing summary strings
            # like 68G/257669M, rather than 68G/258G. The humanfriendly library provides
            # a more reliable number parser, and the humanize library provides a nice
            # formatter.
            alloc_val = humanize.naturalsize(hf.parse_size(alloc_val),
                                             format="%d")
            cfg_val = humanize.naturalsize(hf.parse_size(cfg_val), format="%d")
        occupancy[metric] = f"{alloc_val}/{cfg_val}"
    return occupancy


def available(
    resources: dict = None,
    states: dict = None,
    verbose: bool = False,
):
    """Print a short summary of resources available on the cluster.

    Args:
        resources: a summary of cluster resources, organised by node name.
        states: a mapping between node names and SLURM states.
        verbose: whether to output a more verbose summary of the cluster state.

    NOTES: Some systems allow users to share GPUs.  The logic below amounts to a
    conservative estimate of how many GPUs are available.  The algorithm is:

      For each user that requests a GPU on a node, we assume that a new GPU is allocated
      until all GPUs on the server are assigned.  If more GPUs than this are listed as
      allocated by squeue, we assume any further GPU usage occurs by sharing GPUs.
    """
    summary = {}
    if not resources:
        resources = parse_all_gpus()
    if not states:
        states = node_states()
    res = {
        key: val
        for key, val in resources.items()
        if states.get(key, "down") not in INACCESSIBLE
    }
    usage = gpu_usage(resources=res)
    for subdict in usage.values():
        for gpu_type, node_dicts in subdict.items():
            for node_name, user_gpu_count in node_dicts.items():
                resource_idx = [x["type"]
                                for x in res[node_name]].index(gpu_type)
                count = res[node_name][resource_idx]["count"]
                count = max(count - user_gpu_count['n_gpu'], 0)
                res[node_name][resource_idx]["count"] = count
    by_type = resource_by_type(res)
    total = sum(x["count"] for sublist in by_type.values() for x in sublist)
    summary['total'] = {'count': total}
    summary['types'] = {}
    for key, counts_for_gpu_type in by_type.items():
        gpu_count = sum(x["count"] for x in counts_for_gpu_type)
        tail = ""
        if verbose:
            summary_strs = []
            for x in counts_for_gpu_type:
                node, count = x["node"], x["count"]
                if count:
                    occupancy = occupancy_stats_for_node(node)
                    users = [
                        user for user in usage
                        if node in usage[user].get(key, [])
                    ]
                    details = [
                        f"{key}: {val}"
                        for key, val in sorted(occupancy.items())
                    ]
                    details = f"[{', '.join(details)}] [{','.join(users)}]"
                    summary_strs.append(
                        f"\n -> {node}: {count} {key} {details}")
            tail = " ".join(summary_strs)

        summary['types'][key] = {'count': gpu_count, 'details': tail}
    return summary
