#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer
import click

from slurm_monitor.parse import parse_data
from slurm_monitor.cmd import get_data, get_user

app = typer.Typer()


@app.command()
def cli(name: str):
    print(f'show {name}')


@app.command()
def show():
    data = get_data()
    user_data = get_user()
    msg = parse_data(data, user_data)
    click.echo(msg)


app()
