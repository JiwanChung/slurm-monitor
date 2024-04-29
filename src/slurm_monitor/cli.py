#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typer

from slurm_monitor.gpustat import all_info
from slurm_monitor.format import format_data

app = typer.Typer()


@app.command()
def cli(name: str):
    print(f'show {name}')


@app.command()
def show(name: str):
    print(f'show {name}')


app()
# typer.run(cli)
