# SinGAN with Python 3.11 & Torch 2.1

This project reimplemented SinGAN in Python 3.11 and Torch 2.1. The hyperparameters have not been carefully tuned, and the results will not be comparable to the original paper. However, it serves as a starting point for running SinGAN on a modern machine.

## Table of Contents
- [Installation](#installation)
- [Training Models](#training-models)
- [Paint-to-Image Application](#paint-to-image-application)
- [Visualization and Logs](#visualization-and-logs)
- [System Requirements](#system-requirements)
- [Contributing](#contributing)
- [License](#license)

## Installation

Give detailed instructions on how to install and set up your project. Include any prerequisites or dependencies.

```bash
# Example command
git clone <repository-url>
cd <repository-name>
# additional installation commands
```

## Training Models

### Quick Start

To quickly train models, run the provided script:

```bash
cd src
bash run_train.sh
```

This script will train 10 models as described in the paper, and is expected to take approximately 3 hours on an RTX 4080 GPU.

### Manual Training

Alternatively, you can manually run each command in `run_train.sh`. For detailed command usage, refer to the comments within the script.

## Paint-to-Image Application

After training at least one model, you can use the paint-to-image application. Modify `test.py` as necessary to suit your needs. Detailed instructions on modifications and usage can be found in the script.

## Visualization and Logs

`vis.ipynb` contains classes and methods for extracting and visualizing information from training logs. You can use Jupyter Notebook to open and run this file for insights into the training process.

## System Requirements

- GPU: RTX 4080 or equivalent
- Additional hardware/software requirements

List any specific hardware or software requirements needed to run your project.

## Contributing

If you're open to contributions, provide instructions on how users can contribute to your project. Include guidelines for code contributions, bug reports, and feature requests.

## License

Specify the license under which your project is released.
