# SLM (SLurm-Monitor)

Slurm resource monitoring tools

[![PyPI - Version](https://img.shields.io/pypi/v/slurm-monitor.svg)](https://pypi.org/project/slurm-monitor)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/slurm-monitor.svg)](https://pypi.org/project/slurm-monitor)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

I recommend using `pipx` for installation, since this package is a CLI tool.

```console
pipx install git+https://github.com/JiwanChung/slurm-monitor
```

## Usage

- A single command to show all available/allocated/your_allocated GPU Nodes by partitions:

```bash
slm show
```

mt also comes with basic summarization of your submitted jobs!

## License

`slurm-monitor` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
