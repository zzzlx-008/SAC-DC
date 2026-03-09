# SAC-DC: Scattering Attention CNN - DeepCluster
SAC-DC is an unsupervised clustering framework for seismic waveform data, which integrates scattering transform for robust time-frequency feature extraction, 1D CNN for feature refinement, and attention pooling for global sequence aggregation, trained via the DeepCluster self-supervised paradigm.

## Project Status
This codebase is currently in the **initial development phase** and is subject to continuous updates and optimizations. The core functionality of seismic waveform unsupervised clustering has been implemented, but the following aspects are still being iterated:

### Current Progress
- ✅ Core architecture implementation (Scattering Transform + 1D CNN + Attention Pooling + DeepCluster)
- ✅ Basic unsupervised clustering pipeline for seismic waveform data
- ✅ Key modules (scattering feature extraction, attention-based embedding learning, iterative pseudo-label training)

### Ongoing & Planned Updates
- 🚧 Optimization of scattering network hyperparameters (wavelet octaves/resolution/quality factor)
- 🚧 Enhancement of regularization strategies (dropout, batch normalization, weight decay tuning)
- 🚧 Support for multi-GPU training and large-scale seismic dataset processing
- 🚧 Addition of evaluation metrics (ARI, NMI, clustering accuracy) for quantitative validation
- 🚧 Refactoring of code structure for better readability and maintainability
- 🚧 Documentation improvement (detailed API reference, training tutorial, dataset preparation guide)

### Notes for Users
1. This initial version is intended for **research and experimental use only**; it has not been fully validated for production environments.
2. Backward compatibility is not guaranteed for early updates—please check the commit log before pulling new changes.
3. If you encounter bugs or have optimization suggestions, feel free to submit issues or pull requests.
4. Key hyperparameters (e.g., scattering network settings, CNN layer configuration, batch size) may need adjustment based on your specific seismic dataset characteristics.

## Getting Started
### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- Cupy (for CUDA-accelerated scattering transform)
- Obspy (for seismic waveform data processing)
- NumPy, Pandas, Scikit-learn

### Basic Usage
```bash
# Clone the repository
git clone [repository-url]
cd SAC-DC

# Install dependencies
pip install -r requirements.txt

# Run the basic clustering pipeline
python train_sac_dc.py --data_path [your_seismic_data_path] --num_clusters [target_cluster_number]
```

## License
This project is released under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- The scattering transform module is built based on wavelet scattering theory for seismic signal processing.
- The DeepCluster paradigm is adapted from the original DeepCluster paper (Caron et al., 2018) for unsupervised representation learning.
- Attention pooling implementation references transformer-based sequence aggregation strategies for time-series data.
