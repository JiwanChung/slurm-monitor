#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import zip_longest
from typing import Dict

from rich.text import Text
from textual import log, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.events import Key
from textual.widget import Widget
from textual.widgets import (Label, Collapsible, DataTable, Footer, Header,
                             ListItem, ListView, Static)

from slurm_monitor.parse import _format_user_infos

STATUS_CHAR = {
    'none': '-',
    'scheduled': b'\xe2\xa7\x97'.decode("utf-8"),
    'running': b'\xf0\x9f\x8f\x83'.decode("utf-8")
}

CHARS = {'expand_arrow': "\u21B3"}


def beautify_user_info(row):
    row['status'] = STATUS_CHAR[row['status']]
    return row


def format_data(row):
    if row['gpu'] is None:
        return None

    status = ['none', 'scheduled', 'running'][row['gpu'][0]]
    gpu_total = row['gpu'][3]
    gpu_in_use = row['gpu'][2]
    gpu_user = row['gpu'][1]
    gpu_others = gpu_in_use - gpu_user
    gpu_available = gpu_total - gpu_in_use

    out = {
        'node': f"{row['node']:5s}",
        'mine': gpu_user,
        'used': gpu_others,
        'free': gpu_available,
    }
    return out


class UserWidget(Widget):
    """User job info widget."""

    def __init__(self, *args, data: list, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = data

    def compose(self) -> ComposeResult:
        yield Static('User Jobs')
        yield DataTable(show_header=True)

    def on_mount(self) -> None:
        user_infos = self.data
        tables = self.query(DataTable)
        if len(user_infos) > 0 and tables:
            table = tables[0]
            keys = list(user_infos[0].keys())
            table.add_columns(*keys)
            for row in user_infos:
                row = [row[k] for k in keys]
                table.add_row(*row)


class ServerTable(Widget):

    def __init__(self, *args, data: list, **kwargs):
        self.data = data

        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        btn = CHARS['expand_arrow']
        btn = '>'
        yield Horizontal(
            Label(btn, classes='expand_btn'),
            DataTable(fixed_rows=1, classes='server_table collapsed'))

    def on_mount(self) -> None:
        infos = self.data
        tables = self.query(DataTable)

        is_running = [row for row in infos
                      if row['node'] == 'total'][0]['mine'] > 0

        if is_running:
            self.query_one(Label).update(Text(STATUS_CHAR['running']))
        else:
            self.query_one(Label).update(Text(STATUS_CHAR['none']))

        if len(infos) > 0 and tables:
            table = tables[0]
            keys = list(infos[0].keys())
            table.add_columns(*keys)
            for row in infos:
                row = [row[k] for k in keys]
                table.add_row(*row)

    def do_expand(self) -> None:
        table = self.query_one(DataTable)
        table.toggle_class('collapsed')
        if table.has_class('collapsed'):
            self.set_styles(height=3)
        else:
            self.set_styles(height=len(self.data) + 2)


class ServerWidget(Widget):
    """Server resource info widget."""

    def __init__(self, *args, data: list, **kwargs):
        self.data_key, self.data = data

        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Static(self.data_key)
        yield ServerTable(data=self.data)

    def on_click(self) -> None:
        li = self.parent.parent.parent
        li.do_focus(remove=True)
        li.focused = li.key_map[self.data_key]
        li.do_focus()
        self.query_one(ServerTable).do_expand()


def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


class ServersWidget(Widget):

    def __init__(self, *args, data: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.key_map = {v[0]: i for i, v in enumerate(self.data)}
        self.num_cols = 3
        self.focused = 0

    def compose(self) -> ComposeResult:
        columns = list(zip_longest(*divide_chunks(self.data, self.num_cols)))
        self.keys = self.get_keys(columns)
        yield Horizontal(*[
            ScrollableContainer(*[
                ServerWidget(data=row, id=f'server_{row[0]}') for row in column
                if row is not None
            ],
                                classes='_column') for column in columns
        ])

    def get_keys(self, columns: list) -> list:
        keys = [[v[0] for v in vs if v is not None] for vs in columns]
        return keys

    def get_focused(self):
        key = self.data[self.focused][0]
        return self.query_one(f'#server_{key}')

    def do_focus(self, remove: bool = False):
        comp = self.get_focused()
        if remove:
            comp.remove_class('server_selected')
        else:
            comp.add_class('server_selected')

    def on_show(self) -> None:
        if len(self.data) > 0:
            self.do_focus()

    def on_resize(self) -> None:
        num_cols = self.size.width // 36
        if num_cols != self.num_cols:
            self.num_cols = num_cols
        self.refresh(recompose=True)
        if len(self.data) > 0:
            self.do_focus()

    def do_expand(self) -> None:
        if len(self.data) > 0:
            focused = self.get_focused()
            focused.query_one(ServerTable).do_expand()

    def get_coor(self, x):
        col = x // self.num_cols
        row = x % self.num_cols
        max_row = len(self.keys)
        max_col = len(self.keys[row])
        return col, row, max_row, max_col

    def move_down(self) -> None:
        x = self.focused
        col, row, max_row, max_col = self.get_coor(x)

        if col < max_col - 1:
            prev = self.get_focused()
            prev.remove_class('server_selected')
            self.focused = x + max_row
            curr = self.get_focused()
            curr.add_class('server_selected')

    def move_up(self) -> None:
        x = self.focused
        col, row, max_row, max_col = self.get_coor(x)

        if col > 0:
            prev = self.get_focused()
            prev.remove_class('server_selected')
            self.focused = x - max_row
            curr = self.get_focused()
            curr.add_class('server_selected')

    def move_right(self) -> None:
        x = self.focused
        col, row, max_row, max_col = self.get_coor(x)

        if col >= max_col - 1:
            max_row = len(self.data) % self.num_cols
            if max_row == 0:
                max_row = self.num_cols

        if row < max_row - 1:
            prev = self.get_focused()
            prev.remove_class('server_selected')
            self.focused = x + 1
            curr = self.get_focused()
            curr.add_class('server_selected')

    def move_left(self) -> None:
        x = self.focused
        col, row, max_row, max_col = self.get_coor(x)

        if row > 0:
            prev = self.get_focused()
            prev.remove_class('server_selected')
            self.focused = x - 1
            curr = self.get_focused()
            curr.add_class('server_selected')


def run_textual(data: Dict[str, list], user_infos: list):
    user_infos = _format_user_infos(user_infos)
    if user_infos is None:
        user_infos = []
    user_infos = [beautify_user_info(row) for row in user_infos]
    data = {k: [format_data(row) for row in v] for k, v in data.items()}
    data = {k: [row for row in v if row is not None] for k, v in data.items()}
    data = list(data.items())

    class SLMApp(App):
        """A Textual app for SLurm-Monitor."""
        CSS_PATH = 'style.tcss'

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("h", "left", "Left"),
            ("j", "down", "Down"),
            ("k", "up", "Up"),
            ("l", "right", "Right"),
            Binding("enter", "enter", "Expand", priority=True),
            ("d", "toggle_dark", "Toggle dark mode"),
        ]

        def compose(self) -> ComposeResult:
            """Create child widgets for the app."""
            yield Header()
            yield Footer()
            yield UserWidget(data=user_infos)
            yield Static('Servers')
            yield ServersWidget(data=data, classes='servers')

        def action_down(self) -> None:
            servers = self.query_one(ServersWidget)
            servers.move_down()

        def action_up(self) -> None:
            servers = self.query_one(ServersWidget)
            servers.move_up()

        def action_left(self) -> None:
            servers = self.query_one(ServersWidget)
            servers.move_left()

        def action_right(self) -> None:
            servers = self.query_one(ServersWidget)
            servers.move_right()

        def action_enter(self) -> None:
            servers = self.query_one(ServersWidget)
            servers.do_expand()

        def action_toggle_dark(self) -> None:
            """An action to toggle dark mode."""
            self.dark = not self.dark

    SLMApp().run()
