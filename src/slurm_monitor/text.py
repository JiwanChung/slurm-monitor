#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import typer

from slurm_monitor._textual import run_textual
from slurm_monitor.cmd import get_data, get_user
from slurm_monitor.parse import parse_data, print_data

app = typer.Typer()


@app.command()
def cli(name: str):
    print(f'show {name}')


@app.command()
def show():
    data = get_data()
    user_data = get_user()
    data, user_infos = parse_data(data, user_data)
    msg = print_data(data, user_infos)
    click.echo(msg)


@app.command()
def text():
    # alternative UI using textual
    data = get_data()
    user_data = get_user()
    data, user_infos = parse_data(data, user_data)
    run_textual(data, user_infos)


text()
