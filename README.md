# DRL-EdgeCloud-MultiObjective-Scheduler
Scientific Reports requires public access.
# DRL-Based Multi-Objective Task Scheduling for Edge–Cloud Computing

Official implementation of the paper:

**“DRL-Based Multi-Objective Task Scheduling for Edge–Cloud Computing: Latency, Energy, and SLA Optimisation”**  
Submitted to *Scientific Reports*.

---

## 🔷 Overview

This repository provides the implementation of a Deep Q-Network (DQN) based multi-objective scheduler designed for heterogeneous edge–cloud environments. The proposed framework dynamically balances latency, energy consumption, and SLA compliance using an adaptive reward mechanism.

### ✨ Key Features

- DQN-based intelligent scheduler  
- Adaptive multi-objective reward weighting  
- Edge–cloud unified state representation  
- Compatible with iFogSim 2.0 and CloudSim++  
- Reproducible experimental setup  
- Support for real and synthetic workloads  

---

## 📁 Repository Structure

# DRL-Based Multi-Objective Task Scheduling for Edge–Cloud Computing

Official implementation of the paper:

**“DRL-Based Multi-Objective Task Scheduling for Edge–Cloud Computing: Latency, Energy, and SLA Optimisation”**

---

## Overview

This repository provides a Deep Q-Network (DQN) based intelligent scheduler for heterogeneous edge–cloud environments. The framework dynamically balances latency, energy consumption, and SLA compliance using an adaptive multi-objective reward mechanism.

The implementation is designed to reproduce the experimental results reported in the paper and to support further research in DRL-based edge–cloud optimisation.

---

## Key Features

- DQN-based edge–cloud scheduler  
- Adaptive multi-objective reward weighting  
- Unified state representation  
- Experience replay and target network stabilisation  
- Reproducible experimental setup  
- Support for synthetic and real workloads  

---

## Repository Structure


DRL-EdgeCloud-MultiObjective-Scheduler/
│
├── src/ # Core implementation
├── config/ # Hyperparameters
├── datasets/ # Dataset instructions
├── simulators/ # iFogSim & CloudSim configs
├── results/ # Sample outputs
├── seeds.txt # Random seeds for reproducibility
├── requirements.txt # Python dependencies
└── README.md


---

## Requirements

- Python 3.8 or higher  
- PyTorch 2.x  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  

Install dependencies using:

```bash
pip install -r requirements.txt
Training the Model

To reproduce the main results:

python src/train.py --episodes 1500
Optional arguments

--episodes : number of training episodes

--seed : random seed

--config : path to configuration file

Datasets

This study uses publicly available datasets:

Google Cluster Trace

Azure Functions Trace

Due to licensing restrictions, raw datasets are not redistributed in this repository.

Please download them from:

https://github.com/google/cluster-data

https://github.com/Azure/AzurePublicDataset

Preprocessing scripts are provided in the src/ directory.

Reproducibility

To ensure reproducibility:

Random seeds are provided in seeds.txt

Default hyperparameters match the paper

Results are reported as mean ± standard deviation over multiple runs

Output

After training, results will be saved in:

results/sample_output.csv

Metrics include:

Average latency

Energy consumption

SLA violation rate

Reward convergence

Citation

If you use this code, please cite:

Padala Sravan and Mohammed Ali Shaik,
“DRL-Based Multi-Objective Task Scheduling for Edge–Cloud Computing,”
Scientific Reports, 2025.

Contact

Padala Sravan
SR University, Warangal
Email: padalasravanwgl@gmail.com

License

This project is released under the MIT License.
