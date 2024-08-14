# Pons_py: People-Opinion Networks Python Package

This package allows for the creation of Speaker Landscapes and social networks and bring them together into a multilayer visualisation for the study of group polarisation. Additionally, it allows for the calculation of group polarisation in the Speaker Landscape. 

## Features
- Generate Speaker Landscapes.
- Generate Social Network with a ForceAtlas2 layout for plotting.
- Visualise each layer separately and together with interactive elements to be able to highlight and isolate groups.
- Calculate silhouette scores, group overlap, and other measures of polarisation in the Speaker Landscapes (measures of polarisation in the social network to be implemented in the future).
- Obtain the terms most (and least) correlated to each group from a Speaker Landscape, as a naive way to find the characteristic language of each group, aiding qualitative analysis of the language of groups.

## Installation
For the moment, it is not available for installation through Pipy or conda, but can be installed from Github through pip:

```bash
pip install git+https://github.com/andres-martinez-torres/pons_py
```

## Documentation
The documentation is still being built (as is the library, to which we will add more functions as the project develops). For the moment, you can gave a look at the Jupyter Notebook "Tutorial.ipynb" and the documentation of each function, which should be enough to get you started. 

If you have further questions or find a bug, make sure to raise a Github issue!



