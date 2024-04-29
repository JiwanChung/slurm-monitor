#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from contextlib import redirect_stdout

import click


def get_print_output(txt):
    f = io.StringIO()
    with redirect_stdout(f):
        click.echo(txt, nl=False)
    out = f.getvalue()
    return out
