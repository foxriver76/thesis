# Learning high-dimensional data and processing in non-stationary environments
This repository contains the datasets, as well as the algorithms which has
been published and used in my PhD thesis.

## Structure
The following folder structure is used in this repository:

- prototype_lvq: Contains experiments and algorithms of Chapter 3
- random_projection: Contains experiments and algorithms of Chapter 4
- coresets: Contains experiments and algorithms of Chapter 5
- datasets: Contains all used datasets except from stream generators

Every folder has its own README.md file, which explains the content and strcuture.
Furthermore, usage examples are provided.

## Requirements
For execution of the experiments you need to have a running Python3.6 installation or higher.

Furthermore, ensure that the following packages are installed:

- scikit-multiflow
- sklearn

## Execution
All files which are not represent a model implementation can be executed to generate resulsts
of the thesis. To do so, execute them via

```bash
python <path-to-file>
```

Execution works from the root folder as well as from the chapters folder.

## LICENSE
The source code is licensed under the MIT license.