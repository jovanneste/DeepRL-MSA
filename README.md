# Creating Multiple Sequences Alignments with deep reinforcement learning 

## Overview
Multiple Sequence Alignment (MSA) refers to the process of aligning biological sequences (DNA, RNA or proteins) for comparison. This is generally done for evolutionary analysis and, more recently, drug design and discovery. Advanced protein predictor networks such as AlphaFold use MSA to predict the 3D structure of proteins. However, as an NP complete problem, MSA is a difficult. This project proposes an innovative method to create MSAs leveraging deep reinforcement learning using a novel operator layer for pairwise feature extraction. 

## Features
- Single-agent solution: Create an MSA using a single RL agent. 
- Multi-agent solution: Create an MSA using two RL agents. 


## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your_username/msa-project.git
    cd msa-project
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Single-agent Solution
1. Navigate to the `single_agent` directory:
    ```sh
    cd single_agent
    ```

2. Run the single-agent solution:
    ```sh
    python main.py
    ```

### Multi-agent Solution
1. Navigate to the `multi_agent` directory:
    ```sh
    cd multi_agent
    ```

2. Run the multi-agent solution:
    ```sh
    python main.py
    ```

## Examples
- [Provide examples or screenshots of the output/results]


## Contact
- Author: [Joachim Vanneste](https://github.com/jovanneste)
- Email: joachimvanneste1@gmail.com

