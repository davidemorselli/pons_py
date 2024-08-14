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

## Acknowledgements
The package here is the result of the collaborative project ["PONs - People-Opinion Networks: A study of polarization in word embeddings and social networks in Switzerland and Southern Africa"](https://data.snf.ch/grants/grant/205032). We thank the Swiss National Science Foundation and the South African National Research Foundation.

The original code from which the Speaker Landscapes code is adapted can be found in [Maria Schuld's Github page](https://github.com/mariaschuld/speaker-landscapes) and the paper associated "Speaker landscapes: machine learning opens a window on the everyday language of opinion" (Schuld, Durrheim and Mafunda, 2023) [here](https://doi.org/10.1080/19312458.2023.2277958).

Finally, for the layout of the social networks, a port to python from Gephi of the ForceAtlas2 algorithm. The package used can be found here [FA2_modified](https://github.com/AminAlam/fa2_modified).




