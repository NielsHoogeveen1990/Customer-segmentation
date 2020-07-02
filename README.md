# Welcome!

## 0: Installation
In order to install the package and required libraries, go to the project directory and
use the following bash command:
```
$ pip install . 
```

## 1: How to train the model

In order to train the model with new data use the CLI tool.
The CLI tool works with Click. 
Provide the options, followed by the arguments. 
For example: 
```
$ navara train-model --data-path 'data' --model-version 1.1

```

Be aware that underscores cannot be used with the click decorator. 
Therefore, use a dash instead of an underscore.

## 2: How to create customer segmentation (clusters)

After training the k-means algorithm in (step 1), 
you are able to generate customer segmentation (clusters) on the original data to perform
analyses.

In order to generate cluster, use the following bash command:
```
$ navara get-results --input-path 'data' --output-path 'data'

```

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