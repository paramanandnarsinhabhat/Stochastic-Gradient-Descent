
# Stochastic Gradient Descent Neural Network

This repository contains an implementation of a simple neural network using stochastic gradient descent (SGD) for optimization. The neural network is built from scratch using NumPy and Matplotlib for demonstration purposes.

## Repository Structure

- `notebook/`: Jupyter notebook with interactive code examples.
  - `sgd.ipynb`: The main notebook containing the SGD implementation and visualizations.
- `sgd/`: Python package for the SGD neural network.
  - `source/`: Source files for the SGD implementation.
    - `sgd.py`: Core script with SGD logic.
- `.gitignore`: Gitignore file for excluding unnecessary files from Git tracking.
- `LICENSE`: The license file for the project.
- `README.md`: The file you are currently reading that provides information about this project.
- `requirements.txt`: A text file listing the project's dependencies.

## Features

- Data preparation and normalization.
- Sigmoid activation function implementation.
- Forward and backward propagation for learning.
- Error calculation and visualization of the learning process.
- Use of momentum in gradient descent optimization.

## Requirements

The code requires the following Python packages:

- `numpy`: For numerical operations on arrays and matrices.
- `matplotlib`: For visualizing the error reduction over epochs.

To install these packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the SGD neural network, navigate to the `source` directory and run the `sgd.py` script:

```bash
python sgd.py
```

Alternatively, you can run the Jupyter notebook `sgd.ipynb` in the `notebook` directory to step through the code interactively.

## Visualization

The training process is visualized using Matplotlib to show the reduction in error over time as the network learns from the data.

## Contribution

Contributions to this project are welcome. You can contribute in several ways:

- Reporting bugs
- Suggesting enhancements
- Submitting pull requests with improvements to the code or documentation

Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- This implementation is for educational purposes and may not be suitable for production environments.
- The code was inspired by common patterns in neural network design and the desire to understand SGD at a deeper level.



