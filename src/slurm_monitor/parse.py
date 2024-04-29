#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from itertools import zip_longest
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
from tabulate import tabulate

from slurm_monitor.utils import get_print_output

COLUMNS = {
    'partition': 'Partitions',
    'node': 'NodeName',
    'allocated_resources': 'AllocTRES',
    'total_resources': 'CfgTRES',
}

USER_COLUMNS = {
    'scheduled_nodes': 'SchedNodeList',
    'allocated_nodes': 'NodeList',
    'user_resources': 'TRES',
    'command': 'Command',
    'log_path': 'StdOut',
    'id': 'JobId'
}


def get_property(raw_data: str, name: str) -> str:
    res = re.search(f'{name}=([^\\s]+)\\s', raw_data)
    if res is None:
        return None
    else:
        res = res.groups()
        if len(res) < 1:
            return None
        else:
            return res[0]


def parse_resources(txt: str) -> dict:
    if not txt:
        return {}
    dt = dict([v.split('=') for v in txt.strip().split(',')])
    out = {}
    if 'gres/gpu' in dt:
        out['gpu'] = int(dt['gres/gpu'])
    elif 'gpu' in dt:
        out['gpu'] = int(dt['gpu'])
    if 'cpu' in dt:
        out['cpu'] = int(dt['cpu'])
    if 'mem' in dt:
        mem = int(dt['mem'][:-1])
        unit = dt['mem'][-1]
        if unit == 'M':
            mem = mem // 1000  # to GB
        out['mem'] = mem
    return out


def parse_all_infos(row: str) -> dict:
    dt = {name: get_property(row, key) for name, key in COLUMNS.items()}
    dt['total_resources'] = parse_resources(dt['total_resources'])
    dt['allocated_resources'] = parse_resources(dt['allocated_resources'])
    if dt['partition'] is None:
        dt['partition'] = None
    return dt


def parse_user_infos(row: str) -> dict:
    dt = {name: get_property(row, key) for name, key in USER_COLUMNS.items()}
    dt['user_resources'] = parse_resources(dt['user_resources'])
    dt['status'] = 'running'
    if dt['allocated_nodes'] == '(null)':
        dt['status'] = 'scheduled'
        nodes = dt['scheduled_nodes'].split(',')
    else:
        nodes = dt['allocated_nodes'].split(',')
    outs = {
        'status': dt['status'],
        'nodes': nodes,
        'user_resources': dt['user_resources'],
        'command': dt['command'],
        'log_path': dt['log_path'],
        'id': dt['id'],
    }
    return outs


def format_row(row):
    outs = {}
    for key in ['cpu', 'gpu']:
        if 'total_resources' not in row or not row[
                'total_resources'] or key not in row['total_resources']:
            outs[key] = None
            continue

        alloc = 0
        if key in row['allocated_resources']:
            alloc = row['allocated_resources'][key]
        # targeted / user / alloc / total
        outs[key] = [0, 0, alloc, row['total_resources'][key]]

    outs['mem'] = 0
    if 'mem' in row['total_resources']:
        outs['mem'] = row['total_resources']['mem']
    outs['node'] = row['node']

    if 'user' in row:
        for key in ['cpu', 'gpu']:
            outs[key][0] = 2 if row['user']['running'] else 1
            outs[key][1] = row['user'][key]
        outs['job_id'] = row['user']['job_id']
    return outs


def build_total(rows):
    total = {'mem': None, 'node': 'total'}
    for key in ['cpu', 'gpu']:
        flag = 0
        base = np.array([0, 0, 0])
        for row in rows:
            if key in row and row[key] is not None:
                base = base + np.array(row[key][1:])
                flag = max(flag, row[key][0])
        total[key] = [flag, *list(base)]
    return total


def sort_with_firsts(dt, firsts: list):
    total = set(dt.keys())
    firsts = total - set(firsts)
    rests = list(total - firsts)
    firsts = list(sorted(firsts))
    rests = list(sorted(rests))

    return {k: dt[k] for k in [*firsts, *rests]}


def print_node(row, draw: bool = True, filter_irrelevant: bool = True) -> Optional[str]:
    # set gpu
    if row['gpu'] is None:
        return None

    status = ['none', 'scheduled', 'running'][row['gpu'][0]]
    gpu_total = row['gpu'][3]
    gpu_in_use = row['gpu'][2]
    gpu_user = row['gpu'][1]
    gpu_others = gpu_in_use - gpu_user
    gpu_available = gpu_total - gpu_in_use

    if filter_irrelevant:
        if gpu_available < 1 and gpu_user < 1:
            return None

    def str_colored(start, end, bg: str):
        gpu_str = ''.join([str(v) for v in range(start, end)])
        gpu_str = click.style(gpu_str, bg=bg, fg='white')
        return gpu_str

    gpu_str = f'{row['node']:5s} ('
    if gpu_available > 0:
        gpu_str += click.style(f'{gpu_available}', fg='green')
    else:
        gpu_str += f'{gpu_available}'
    gpu_str += f'/{gpu_total}) '

    if draw:
        gpu_str += '['
        start = 0
        end = start + gpu_user
        gpu_str += str_colored(start, end, bg='blue')
        start = end
        end = start + gpu_available
        gpu_str += str_colored(start, end, bg='green')
        start = end
        end = start + gpu_others
        gpu_str += str_colored(start, end, bg='reset')
        gpu_str += ']'

    gpu_str = gpu_str.strip()
    if status == 'running':
        gpu_str += ' ['
        gpu_str += click.style(b'\xf0\x9f\x8f\x83'.decode("utf-8"), fg='blue')
        if 'job_id' in row:
            gpu_str += ' running: '
            gpu_str += click.style(f"{row['job_id']}", fg='magenta')
            gpu_str += ']'
        else:
            gpu_str += ' running]'
    elif status == 'scheduled':
        gpu_str += ' ['
        gpu_str += click.style(b'\xe2\xa7\x97'.decode("utf-8"), fg='yellow')
        if 'job_id' in row:
            gpu_str += ' scheduled: '
            gpu_str += click.style(f"{row['job_id']}", fg='magenta')
            gpu_str += ']'
        else:
            gpu_str += ' scheduled]'

    return gpu_str


def print_partition(name: str, nodes: list, filter_irrelevant: bool = True) -> Optional[str]:
    partition_str = click.style(name, bg='reset', fg='green')
    for node in nodes:
        node_str = print_node(node, node['node'] != 'total', filter_irrelevant=filter_irrelevant)
        if node_str is not None:
            partition_str += f'\n{node_str}'
    if filter_irrelevant and len(partition_str.split('\n')) <= 1:
        partition_str = None
    return partition_str


def wrap_box(text, title: Optional[str] = None):
    text = text.split('\n')
    lengths = [len(get_print_output(v)) for v in text]
    max_len = max(lengths)
    out = ''
    out += '-' * (max_len + 4)
    
    if title is not None:
        out += '\n'
        length = len(get_print_output(title))
        title += ' ' * (max_len - length)
        out += f'| {title} |\n'
        out += '-' * (max_len + 4)

    out += '\n'
    for length, line in zip(lengths, text):
        line += ' ' * (max_len - length)
        out += f'| {line} |\n'
    out += '-' * (max_len + 4)
    return out


def format_user_infos(data: list):
    if len(data) < 1:
        return None
    outs = []
    for row in data:
        out = {
            'id': row['id'],
            'status': row['status'],
            'cmd': Path(row['command']).name,
            'out': Path(row['log_path']).name,
            'nodes': row['nodes'],
            'gpu': row['user_resources'].get('gpu', 0)
        }
        outs.append(out)
    return tabulate(outs, headers='keys', tablefmt='simple_outline')


def concat_strs(*strs):
    strs = [v.split('\n') for v in strs]
    outs = []
    for row in zip_longest(*strs):
        out_row = ''
        for col in row:
            if col is not None:
                out_row += f' {col}'
        out_row = out_row.strip()
        outs.append(out_row)
    return '\n'.join(outs)


def parse_data(data: str, user_data: List[str] = []):
    if not data:
        return "Error retrieving statistics..."

    data = [v.strip() for v in data.split('\n\n') if v]

    infos = {}
    partitions = set()
    for row in data:
        dt = parse_all_infos(row)
        infos[dt['node']] = dt
        partitions.add(dt['partition'])

    user_infos = []
    for _user_data in user_data:
        if _user_data.startswith('JobId'):
            _user_data = [v.strip() for v in _user_data.split('\n\n') if v]

            for row in _user_data:
                dt = parse_user_infos(row)
                user_infos.append(dt)

    # insert user infos
    show_first = set()
    if user_infos:
        for row in user_infos:
            divider = len(row['nodes'])
            gpu = row['user_resources'].get('gpu', None)
            if gpu is None:
                gpu = 0
            cpu = row['user_resources'].get('cpu', None)
            if cpu is None:
                cpu = 0
            # WARNING: assume equal usage
            gpu = gpu // divider
            cpu = cpu // divider

            for node in row['nodes']:
                infos[node]['user'] = {
                    'job_id': row['id'],
                    'cpu': cpu,
                    'gpu': gpu,
                    'running': row['status'] == 'running'
                }
                show_first.add(node)

    partitions = list(partitions)
    data = {k: {} for k in partitions}
    show_first_partitions = set()
    for node, row in infos.items():
        if node in show_first:
            show_first_partitions.add(row['partition'])
        data[row['partition']][node] = format_row(row)
    # sort order
    data = {
        k: list(sort_with_firsts(v, show_first).values())
        for k, v in data.items()
    }
    data = sort_with_firsts(data, show_first_partitions)
    data = {k: [build_total(v), *v] for k, v in data.items()}

    strs = ''
    for k, v in data.items():
        _str = print_partition(k, v)
        if _str is not None:
            strs += f'\n\n{_str}'
    strs = strs.strip()
    strs = wrap_box(strs, title='Partitions (Node - GPUs)')
    gpu_str = strs

    user_str = format_user_infos(user_infos)
    all_str = concat_strs(gpu_str, user_str)
    return all_str


def parse_user(data):
    data = data.split('\n')[1:]
    jobs = []
    for row in data:
        job = re.search('[0-9]+', row)
        if not job:
            continue
        job = job[0]
        jobs.append(job)
    return jobs
