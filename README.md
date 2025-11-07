# flex

**Fourier-Laguerre expansions for images of galaxies.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/michael-petersen/flex/blob/main/LICENSE)




## Installation

Installation of `flex` currently proceeds from local builds after cloning this repository:
```
git clone https://github.com/michael-petersen/flex.git
```

```
pip install .
```


## Quickstart Example

Drawing from a uniform disc.

```python
import numpy as np
import flex

N = 10_000_000

# Generate uniformly in polar coordinates
r = np.sqrt(np.random.uniform(0, 4, N)) 
theta = np.random.uniform(0, 2*np.pi, N)
mass = np.random.uniform(0, 1, N)

# set expansion parameters
rscl,mmax,nmax = 1.0,0,10

F = flex.FLEX(rscl,mmax,nmax,r,theta)
```


## Documentation

Complete documentation, including more example notebooks, will be available at [at this URL]().

## Contributing

The `flex` maintainers welcome contributions to software, examples, and documentation. The maintainers are actively monitoring pull requests and would be happy to collaborate on contributions or ideas. If you have any requests for additional information or find any bugs, please open an issue directly.

## License

`flex` is available under the MIT license. See the LICENSE file for specifics.
