# Artificial Intelligence Projects

A comprehensive collection of AI and Machine Learning projects demonstrating various algorithms, techniques, and applications in artificial intelligence. This repository showcases implementations ranging from classic search algorithms to modern deep learning approaches.

## Table of Contents

- [Projects Overview](#projects-overview)
- [Technologies](#technologies)
- [Getting Started](#getting-started)
- [Project Descriptions](#project-descriptions)

## Projects Overview

| Project                                                                 | Description                                                   | Key Technologies          |
| ----------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------- |
| [Calculator](#calculator)                                               | Expression tree-based calculator with parsing                 | Python, JSON              |
| [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn) | Image classification for cats and dogs                        | TensorFlow/Keras, CNN     |
| [Delivery Route Finder](#delivery-route-finder)                         | AI-powered route optimization with multiple search algorithms | BFS, UCS, A\* Search      |
| [Forward Propagation](#forward-propagation)                             | Neural network forward propagation implementation             | NumPy, ANN                |
| [Markov Decision Process](#markov-decision-process)                     | Reinforcement learning with value iteration                   | Gymnasium, MDP            |
| [ML Binary Classification](#ml-binary-classification)                   | Complete ML pipeline with hyperparameter optimization         | Scikit-learn, Grid Search |
| [Search Algorithms](#search-algorithms)                                 | Classic AI search implementations                             | BFS, DFS                  |

## Technologies

- **Programming Languages**: Python
- **Machine Learning**: Scikit-learn, TensorFlow, Keras
- **Reinforcement Learning**: Gymnasium (OpenAI Gym)
- **Data Processing**: NumPy, Pandas, Matplotlib
- **AI Algorithms**: Search Algorithms, Neural Networks, MDP, Tree-based Learning

## Getting Started

Each project contains its own setup instructions and dependencies. Generally, you'll need:

```bash
# Clone the repository
git clone https://github.com/yourusername/artificial-intelligence.git
cd artificial-intelligence

# Navigate to specific project
cd "Project Name"

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

## Project Descriptions

### Calculator

A sophisticated mathematical expression evaluator using operator trees and JSON-based input.

**Features:**

- Expression tree parsing and evaluation
- Support for multiple mathematical operations
- JSON configuration for test cases

**Files:**

- `calculator.py` - Main calculator implementation
- `CSC480_Calculator/` - Core modules (operand, operator, tree structure)
- `math_expr_*.json` - Test expression files

**Usage:**

```bash
python calculator.py
```

---

### Convolutional Neural Network (CNN)

Deep learning implementation for binary image classification (cats vs. dogs).

**Features:**

- Image preprocessing and augmentation
- CNN architecture for classification
- Training and testing pipelines

**Files:**

- `image-classification.ipynb` - Complete notebook with model training
- `test.py` - Testing script
- `train/` - Training dataset organized by class

**Technologies:**

- TensorFlow/Keras
- Computer Vision
- Deep Learning

---

### Delivery Route Finder

An intelligent delivery route optimization system comparing three AI search algorithms to find optimal paths for multi-stop delivery scenarios.

**Features:**

- Multiple search algorithms (BFS, UCS, A\*)
- Custom admissible heuristics
- Real-world map data (Tegucigalpa, Honduras)
- Performance metrics and comparative analysis

**Key Results:**

- A\* performs ~4x faster than BFS/UCS on complex scenarios
- Guaranteed optimal solutions with UCS and A\*
- Efficient tree pruning with heuristics

**Usage:**

```bash
cd "Delivery Route Finder"
python main.py
```

[View detailed documentation](Delivery%20Route%20Finder/README.md)

---

### Forward Propagation

Implementation of Artificial Neural Network forward propagation from scratch.

**Features:**

- Step-by-step forward propagation visualization
- Manual implementation without high-level frameworks
- Educational Jupyter notebook format

**Files:**

- `ann-forward-propagation.ipynb` - Interactive notebook
- `forward-propagation.jpg` - Architecture diagram

---

### Markov Decision Process

Reinforcement learning project solving the FrozenLake environment using MDP and value iteration.

**Features:**

- Random policy evaluation
- Value iteration for optimal policy
- Visual simulation with statistical analysis
- Comparison across 100 experiments

**Key Concepts:**

- Bellman optimality equation
- Policy evaluation
- Stochastic environment navigation

**Usage:**

```bash
cd "Markov Decision Process"
pip install -r requirements.txt
python main.py
```

[View detailed documentation](Markov%20Decision%20Process/README.md)

---

### ML Binary Classification

A comprehensive machine learning pipeline featuring automated hyperparameter optimization and model evaluation.

**Features:**

- Three classifiers (Decision Tree, Random Forest, Logistic Regression)
- K-fold cross-validation
- Grid search for hyperparameter tuning
- Model persistence with pickle
- Multiple dataset sizes

**Pipeline:**

1. `part_01_cross_validation.py` - Cross-validation
2. `part_02_grid_search.py` - Hyperparameter optimization
3. `part_03_training.py` - Final model training
4. `part_04_testing.py` - Model evaluation

**Usage:**

```bash
cd "ML Binary Classification"
# Cross-validation
python part_01_cross_validation.py hyperparameters.json training_data_small.csv 5

# Grid search
python part_02_grid_search.py hyperparameters.json training_data_{}.csv 5

# Training
python part_03_training.py hyperparameters.json training_data_large.csv scaler.pkl classifier.pkl

# Testing
python part_04_testing.py testing_data.csv scaler.pkl classifier.pkl
```

[View detailed documentation](ML%20Binary%20Classification/README.md)

---

### Search Algorithms

Implementation of classic AI search algorithms for problem-solving.

**BFS/DFS Maze Solver**: Breadth-first and depth-first search for maze navigation

**Features:**

- Multiple test mazes (`maze1.txt`, `maze2.txt`, `maze3.txt`)
- Visual output (`maze.png`)
- Path visualization and solution finding

**Usage:**

```bash
cd "Search Algorithms/BFS_DFS Maze"
python main.py
```

---

## Learning Objectives

This repository demonstrates:

- **Search Algorithms**: BFS, DFS, UCS, A\*
- **Machine Learning**: Classification, regression, hyperparameter tuning
- **Deep Learning**: CNN architecture, image classification
- **Reinforcement Learning**: MDP, value iteration, policy optimization
- **Algorithm Analysis**: Time complexity, space complexity, performance comparison

## Performance Highlights

- **Delivery Route Finder**: A\* achieves sub-second performance on 12-location routes
- **ML Classification**: 77.5% accuracy on binary classification with Random Forest
- **MDP**: Optimal policy convergence in FrozenLake environment

## Contributing

Feel free to explore, learn from, and build upon these implementations. Each project is self-contained with detailed documentation.

## License

This repository is for educational purposes.

## Author

**Jonesh Shrestha**

---

_This repository represents a comprehensive exploration of artificial intelligence techniques, from foundational algorithms to modern machine learning approaches._
