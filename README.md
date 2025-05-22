# Lindblad Dynamics Approximator

This repository accompanies the paper:

> [Inferring Dynamical Generators of Local Quantum Observables from Projective Measurements through Machine Learning](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.21.L041001)  
> *Phys. Rev. Applied 21, L041001 (2024)*

It provides tools and models to approximate Lindblad-type dynamics for quantum systems using machine learning techniques. The method infers a generator for the evolution of local observables based on measurement data, enabling effective simulations and predictions in open quantum systems.

---

## 🔧 Features

- Machine learning-based inference of quantum dynamical generators.
- Trains models to approximate Lindblad dynamics of local observables.
- Includes benchmarks and example workflows in Jupyter notebooks.
- Built using PyTorch, NumPy, and Matplotlib.

---

## 📁 Repository Structure

- `src/lda/`: Source code of the Lindblad Dynamics Approximator.
- `notebooks/training_model.ipynb`: Shows how to train the model to learn the quantum dynamics from synthetic data.
- `notebooks/benchmark.ipynb`: Benchmarks the learned model against the exact dynamics to validate its performance.
- `tests/`: Unit tests for the codebase.
- `environment.yml`: Conda environment specification file.

---

## 🖥️ Installation

### 📦 Using Conda (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/giovannicemin/lindblad_dynamics_approximator.git
   cd lindblad_dynamics_approximator
   ```
   
2. **Create the environment**:
   ```bash
   conda env create -f environment.yml
   ```
   This will install all dependencies listed in the environment.yml file and make the lda package available for import and development.

3. **Activate the environment**:
   ```bash
   conda activate lda
   ```

---

## 🚀 Usage

After installation, you can explore the notebooks:

- Open notebooks/training_model.ipynb to train the machine learning model from scratch.

- Use notebooks/benchmark.ipynb to evaluate how well the learned model approximates known quantum dynamics.

---

## 📚 Citation
If you use this code, please cite the original paper:

```
@article{Cemin2024,
  title = {Inferring Dynamical Generators of Local Quantum Observables from Projective Measurements through Machine Learning},
  author = {Cemin, Giovanni and Carnazza, Francesco and Hauke, Philipp},
  journal = {Phys. Rev. Applied},
  volume = {21},
  number = {4},
  pages = {L041001},
  year = {2024},
  doi = {10.1103/PhysRevApplied.21.L041001}
}
```
