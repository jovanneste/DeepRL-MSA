# Creating Multiple Sequences Alignments with deep reinforcement learning 

## Overview
Multiple Sequence Alignment (MSA) refers to the process of aligning biological sequences (DNA, RNA or proteins) for comparison. This is generally done for evolutionary analysis and, more recently, drug design and discovery. Advanced protein predictor networks such as AlphaFold use MSA to predict the 3D structure of proteins. However, as an NP complete problem, MSA is a difficult. This project proposes an innovative method to create MSAs leveraging deep reinforcement learning using a novel operator layer for pairwise feature extraction. 

## Features
- Single-agent solution: Create an MSA using a single RL agent. 
- Multi-agent solution: Create an MSA using two co-operative RL agents. 
- Ensemble model: Up to five agents with majority voting.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/jovanneste/DeepRL-MSA.git
    cd DeepRL-MSA
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Naviagte to the *src* directory and run a solution.

### Single-agent 
Run the single-agent solution:


```sh
python main_msa.py 
```

### Multi-agent 
Run the multi-agent solution:

```sh
python main_msa.py --multi
```

### Ensemble 
Run the voting solution:

```sh
python main_msa.py --vote
```


## Contact
- Author: [Joachim Vanneste](https://github.com/jovanneste)
- Email: joachimvanneste1@gmail.com

