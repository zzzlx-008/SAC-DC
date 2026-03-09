# Open-source release note for this codebase

## Upstream basis

This project is derived in part from the open-source repository:

- `scatseisnet/scatseisnet`
- GitHub: https://github.com/scatseisnet/scatseisnet

Recommended citation for the upstream package:

> Seydoux, L. S., Steinmann, R., Gärtner, M., Tong, F., Esfahani, R., & Campillo, M. (2025).
> *Scatseisnet, a Scattering network for seismic data analysis (0.3).* Zenodo.
> https://doi.org/10.5281/zenodo.15110686

## What this repository changes

Compared with the upstream scattering-network implementation, this codebase adds or modifies:

1. segment preprocessing and timestamp parsing,
2. a DeepCluster iterative pseudo-label training pipeline,
3. attention-based embedding pooling,
4. full-dataset KMeans clustering without PCA,
5. representative centroid-waveform visualization and output export.

## Attribution statement

Suggested text for your repository README:

> This repository includes code adapted from `scatseisnet/scatseisnet`
> (GPL-3.0). We modified the upstream scattering-network workflow and
> integrated it into a DeepCluster-based representation-learning and
> clustering pipeline for seismic / planetary subsurface signal analysis.

## License reminder

The upstream repository page indicates GPL-3.0 licensing. If your released
code is a derivative work of that source code, your repository should
normally remain under a GPL-compatible license, and the attribution notice
should be retained in the relevant source files and README.
