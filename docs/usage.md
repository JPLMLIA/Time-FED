# Getting Started

TimeFED is recommended to be installed via pip at this time.

## Installation

First clone the repository via:

```bash
git clone https://github.jpl.nasa.gov/jamesmo/TimeFED.git
```

Then install:

```bash
pip install -e TimeFED
```

## Usage

TimeFED is a pipeline system that essentially connects independent scripts together to prepare timeseries datasets for machine learning models and then trains and tests models. This is all done via a configuration YAML file and the TimeFED CLI. The CLI is accessed via the `timefed` command in a terminal:

```bash
$ timefed --help
Usage: timefed [OPTIONS] COMMAND [ARGS]...

  TimeFED is a machine learning system for Time series Forecasting, Evaluation
  and Deployment.

Options:
  --help  Show this message and exit.

Commands:
  config  mlky configuration commands
  run     Executes the TimeFED pipeline
```

There are two primary commands, `config` and `run`:
- The `config` command is used for generating new template configuration files, validating an existing config, or simply printing what a patched config may produce.
- The `run` command enters the TimeFED pipeline and executes the various pieces of it, depending on how it is configured.
