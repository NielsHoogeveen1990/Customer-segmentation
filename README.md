# Welcome!

## 0: Installation
In order to install the package and required libraries, go to the project directory and
use the following bash command:
```
$ pip install . 
```

## 1: How to use the CLI

In order to train the model with new data use the CLI tool.
The CLI tool works with Click. 
Provide the options, followed by the arguments. 
For example: 
```
$ navara train-model --data-path 'data/' --model-version 1.1

```

Be aware that underscores cannot be used with the click decorator. 
Therefore, use a dash instead of an underscore.

### Need help?
Use the --help option to see the available options for a function.
```
$ navara train-model --help
Usage: navara train-model [OPTIONS]

Options:
  --data-path PATH
  --model-version INTEGER
  --help   
```