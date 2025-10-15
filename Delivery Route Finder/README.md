# Delivery Route Finder

An intelligent delivery route optimization system that uses AI search algorithms to find optimal paths for multi-stop delivery scenarios. The system compares three different search strategies: Breadth-First Search (BFS), Uniform Cost Search (UCS), and A\* Search with custom heuristics.

## Features

- **Multiple Search Algorithms**: Implements BFS, UCS, and A\* search for route optimization
- **Custom Heuristics**: Admissible heuristic design for A\* that guarantees optimal solutions
- **Real-World Map**: Uses actual street network data from Tegucigalpa, Honduras
- **Performance Metrics**: Tracks nodes explored, nodes reached, solution cost, and execution time
- **Comparative Analysis**: Automatically runs all three algorithms to compare efficiency

## Technical Highlights

- **State Space Search**: Efficient graph traversal with visited target tracking
- **Heuristic Design**: Straight-line distance estimation for admissible A\* heuristic
- **Optimal Solutions**: Guaranteed optimal routes using UCS and A\*
- **Modular Architecture**: Clean separation between problem domain and search algorithms

## How to Run

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd "Delivery Route Finder"
   ```

2. **Install Python 3.x** (if not already installed)

   - Python 3.8 or higher recommended

3. **Run the application**
   ```bash
   python3 main.py
   ```

The program will execute all delivery scenarios defined in `test_cases.json` and display:

- The optimal route for each scenario
- Total distance traveled
- Performance statistics for each algorithm

## Project Structure

```
.
├── delivery_problem/      # Problem domain (map, state, search request)
│   ├── map.py            # City map and distance calculations
│   ├── problem.py        # Problem formulation and heuristics
│   ├── state.py          # State representation
│   └── search_request.py # Delivery scenario definitions
├── search/               # Search algorithms implementation
│   ├── search_algorithms.py  # BFS, UCS, and A* implementations
│   ├── search_result.py      # Result tracking
│   └── search_tree_node.py   # Search tree structure
├── main.py              # Entry point
├── tegucigalpa.json     # City map data
└── test_cases.json      # Delivery scenarios

```

## Algorithm Comparison

| Algorithm | Optimality | Completeness | Space Complexity |
| --------- | ---------- | ------------ | ---------------- |
| BFS       | ✓\*        | ✓            | O(b^d)           |
| UCS       | ✓          | ✓            | O(b^d)           |
| A\*       | ✓          | ✓            | O(b^d)           |

\*BFS optimal only with uniform edge costs

## Example Output

```
Delivery Scenario:
 - Name: Easy_1
 - Starting Location: A1
 - Delivery Locations: ['C2']

 - Running A Star Search
 --> Solution found: [A1] -> C3 -> (C2) -> C3 -> [A1]
 --> Solution Cost: 5.45 miles
 --> Nodes Reached: 14
 --> Nodes Explored: 4
 --> Search Time (s): 0.00005
```

## Technologies Used

- **Python 3**: Core programming language
- **Graph Theory**: State space representation and traversal
- **Search Algorithms**: AI-based pathfinding techniques
- **Data Structures**: Priority queues (heapq) for efficient search

# Search Algorithm Results

## Easy Test Cases

### Test Case: Easy_1

**Starting Location:** A1  
**Delivery Locations:** ['C2']

| Algorithm | Solution Path                | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | ---------------------------- | --------- | ------------- | -------------- | -------- |
| BFS       | [A1] → C3 → (C2) → C3 → [A1] | 5.45      | 27            | 22             | 0.00018  |
| UCS       | [A1] → C3 → (C2) → C3 → [A1] | 5.45      | 56            | 42             | 0.00030  |
| A\*       | [A1] → C3 → (C2) → C3 → [A1] | 5.45      | 14            | 4              | 0.00006  |

### Test Case: Easy_2

**Starting Location:** M1  
**Delivery Locations:** ['D1', 'R3']

| Algorithm | Solution Path                                        | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | ---------------------------------------------------- | --------- | ------------- | -------------- | -------- |
| BFS       | [M1] → D3 → (D1) → D3 → [M1] → M2 → (R3) → M2 → [M1] | 7.15      | 208           | 158            | 0.00113  |
| UCS       | [M1] → M2 → (R3) → M2 → [M1] → D3 → (D1) → D3 → [M1] | 7.15      | 229           | 188            | 0.00129  |
| A\*       | [M1] → M2 → (R3) → M2 → [M1] → D3 → (D1) → D3 → [M1] | 7.15      | 41            | 15             | 0.00015  |

### Test Case: Easy_3

**Starting Location:** D1  
**Delivery Locations:** ['C2', 'A2', 'D3']

| Algorithm | Solution Path                                               | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | ----------------------------------------------------------- | --------- | ------------- | -------------- | -------- |
| BFS       | [D1] → A1 → (A2) → A1 → C3 → (C2) → C3 → [D1] → (D3) → [D1] | 10.95     | 328           | 254            | 0.00175  |
| UCS       | [D1] → (D3) → D4 → C4 → (C2) → C3 → A1 → (A2) → A1 → [D1]   | 10.14     | 490           | 417            | 0.00293  |
| A\*       | [D1] → (D3) → D4 → C4 → (C2) → C3 → A1 → (A2) → A1 → [D1]   | 10.14     | 83            | 40             | 0.00039  |

## Medium Test Cases

### Test Case: Medium_1

**Starting Location:** Q1  
**Delivery Locations:** ['D1', 'R2', 'V4', 'F3']

| Algorithm | Solution Path                                                                                            | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | -------------------------------------------------------------------------------------------------------- | --------- | ------------- | -------------- | -------- |
| BFS       | [Q1] → M4 → M1 → D3 → (D1) → D2 → I3 → I4 → R1 → (R2) → L4 → V2 → (V4) → V1 → Q3 → Q2 → (F3) → Q2 → [Q1] | 19.28     | 1409          | 1321           | 0.00849  |
| UCS       | [Q1] → M4 → M1 → D3 → (D1) → D3 → M1 → M2 → R3 → (R2) → L4 → V2 → (V4) → V1 → Q3 → Q2 → (F3) → Q2 → [Q1] | 18.78     | 1467          | 1516           | 0.00941  |
| A\*       | [Q1] → Q2 → (F3) → Q2 → Q3 → V1 → (V4) → V2 → L4 → (R2) → R3 → M2 → M1 → D3 → (D1) → D3 → M1 → M4 → [Q1] | 18.78     | 580           | 445            | 0.00331  |

### Test Case: Medium_2

**Starting Location:** R2  
**Delivery Locations:** ['A2', 'C2', 'N3', 'K3', 'F1']

| Algorithm | Solution Path                                                                                                                                      | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------------- | -------------- | -------- |
| BFS       | [R2] → L4 → V2 → V1 → Q3 → Q2 → F3 → (F1) → F2 → K2 → (K3) → O3 → O2 → (N3) → N1 → M3 → D4 → C4 → (C2) → C3 → A1 → (A2) → A3 → I3 → I4 → R1 → [R2] | 28.52     | 2968          | 2940           | 0.01665  |
| UCS       | [R2] → R1 → I4 → I3 → A3 → (A2) → A1 → C3 → (C2) → C4 → D4 → M3 → N1 → (N3) → O2 → O3 → (K3) → K2 → F2 → (F1) → F3 → Q2 → Q1 → M4 → M2 → R3 → [R2] | 27.81     | 2959          | 3171           | 0.01882  |
| A\*       | [R2] → R1 → I4 → I3 → A3 → (A2) → A1 → C3 → (C2) → C4 → D4 → M3 → N1 → (N3) → O2 → O3 → (K3) → K2 → F2 → (F1) → F3 → Q2 → Q1 → M4 → M2 → R3 → [R2] | 27.81     | 638           | 518            | 0.00382  |

### Test Case: Medium_3

**Starting Location:** L3  
**Delivery Locations:** ['I1', 'S2', 'T4', 'X4', 'E4', 'U1']

| Algorithm | Solution Path                                                                                                                                               | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------------- | -------------- | -------- |
| BFS       | [L3] → I4 → (I1) → I3 → D2 → D3 → M1 → M4 → Q1 → Q4 → F4 → U4 → (U1) → U2 → E2 → (E4) → W3 → (X4) → X1 → (T4) → T1 → S1 → (S2) → S1 → T1 → T2 → L2 → [L3]   | 32.82     | 5534          | 5343           | 0.03019  |
| UCS       | [L3] → I4 → (I1) → I4 → [L3] → L4 → V2 → V1 → Q3 → W2 → W4 → U4 → (U1) → U2 → E2 → (E4) → W3 → (X4) → X1 → (T4) → T1 → S1 → (S2) → S1 → T1 → T2 → L2 → [L3] | 32.50     | 5854          | 6406           | 0.03750  |
| A\*       | [L3] → L4 → V2 → V1 → Q3 → W2 → W4 → U4 → (U1) → U2 → E2 → (E4) → W3 → (X4) → X1 → (T4) → T1 → S1 → (S2) → S1 → T1 → T2 → L2 → [L3] → I4 → (I1) → I4 → [L3] | 32.50     | 2208          | 1979           | 0.01477  |

## Hard Test Cases

### Test Case: hard_1

**Starting Location:** Q2  
**Delivery Locations:** ['A2', 'S3', 'G2', 'W2', 'E2', 'J3', 'H2']

| Algorithm | Solution Path                                                                                                                                                                                                                       | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------------- | -------------- | -------- |
| BFS       | [Q2] → F3 → F1 → (J3) → J2 → (H2) → J2 → J1 → K2 → P4 → P1 → M3 → D4 → D1 → A1 → (A2) → G1 → (G2) → G4 → I1 → I4 → L3 → L2 → T2 → T1 → S1 → (S3) → S1 → T1 → T4 → X1 → X4 → W3 → E4 → (E2) → E4 → W3 → (W2) → Q3 → [Q2]             | 48.01     | 11668         | 11496          | 0.06602  |
| UCS       | [Q2] → Q1 → M4 → M1 → D3 → D1 → A1 → (A2) → G1 → (G2) → G4 → I1 → I4 → L3 → L2 → T2 → T1 → S1 → (S3) → S1 → T1 → T4 → X1 → X3 → W1 → (W2) → W4 → U4 → U2 → (E2) → U2 → U1 → B4 → B3 → (J3) → J2 → (H2) → J2 → (J3) → F1 → F3 → [Q2] | 44.62     | 11721         | 12491          | 0.07588  |
| A\*       | [Q2] → Q1 → M4 → M1 → D3 → D1 → A1 → (A2) → G1 → (G2) → G4 → I1 → I4 → L3 → L2 → T2 → T1 → S1 → (S3) → S1 → T1 → T4 → X1 → X3 → W1 → (W2) → W4 → U4 → U2 → (E2) → U2 → U1 → B4 → B3 → (J3) → J2 → (H2) → J2 → (J3) → F1 → F3 → [Q2] | 44.62     | 6864          | 7092           | 0.05276  |

### Test Case: hard_2

**Starting Location:** P2  
**Delivery Locations:** ['N3', 'C4', 'D4', 'A2', 'X2', 'B2', 'J2', 'F2']

| Algorithm | Solution Path                                                                                                                                                                                    | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | ------------- | -------------- | -------- |
| BFS       | [P2] → P1 → N4 → (N3) → N1 → M3 → (D4) → (C4) → C3 → A1 → (A2) → A3 → I3 → I4 → L3 → L4 → V2 → V4 → (X2) → X3 → W1 → W4 → F4 → (F2) → F1 → B3 → (B2) → B3 → J3 → (J2) → J1 → K2 → P4 → [P2]      | 35.58     | 23175         | 22843          | 0.13724  |
| UCS       | [P2] → P4 → K2 → (F2) → F1 → J3 → (J2) → J3 → B3 → (B2) → B4 → U1 → U4 → W4 → W1 → X3 → (X2) → V4 → V2 → L4 → L3 → I4 → I3 → A3 → (A2) → A1 → C3 → (C4) → (D4) → M3 → N1 → (N3) → N4 → P1 → [P2] | 34.57     | 23318         | 24451          | 0.15451  |
| A\*       | [P2] → P4 → K2 → (F2) → F1 → J3 → (J2) → J3 → B3 → (B2) → B4 → U1 → U4 → W4 → W1 → X3 → (X2) → V4 → V2 → L4 → L3 → I4 → I3 → A3 → (A2) → A1 → C3 → (C4) → (D4) → M3 → N1 → (N3) → N4 → P1 → [P2] | 34.57     | 9741          | 9395           | 0.07621  |

### Test Case: hard_3

**Starting Location:** O1  
**Delivery Locations:** ['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2']

| Algorithm | Solution Path                                                                                                                                                                                                                         | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------------- | -------------- | -------- |
| BFS       | [O1] → N4 → N1 → M3 → D4 → C4 → (C2) → C3 → A1 → (A2) → G1 → (G2) → G4 → I1 → (I2) → I3 → (D2) → D4 → M3 → P1 → P4 → F3 → (F2) → F4 → U4 → U2 → (E2) → U2 → U1 → B4 → (B2) → B3 → J3 → J2 → (H2) → H1 → J4 → K4 → K3 → O3 → [O1]      | 45.04     | 46678         | 46463          | 0.29525  |
| UCS       | [O1] → P3 → P4 → K2 → (F2) → F1 → J3 → J2 → (H2) → J2 → J3 → B3 → (B2) → B4 → U1 → U2 → (E2) → U2 → U4 → F4 → Q4 → Q1 → M4 → M1 → D3 → (D2) → I3 → (I2) → I1 → G4 → (G2) → G1 → (A2) → A1 → C3 → (C2) → C4 → D4 → M3 → N1 → N4 → [O1] | 42.75     | 46817         | 50882          | 0.33802  |
| A\*       | [O1] → N4 → N1 → M3 → D4 → C4 → (C2) → C3 → A1 → (A2) → G1 → (G2) → G4 → I1 → (I2) → I3 → (D2) → D3 → M1 → M4 → Q1 → Q4 → F4 → U4 → U2 → (E2) → U2 → U1 → B4 → (B2) → B3 → J3 → J2 → (H2) → J2 → J3 → F1 → (F2) → K2 → P4 → P3 → [O1] | 42.75     | 21208         | 19998          | 0.16998  |

### Test Case: hard_4

**Starting Location:** V3  
**Delivery Locations:** ['B4', 'D1', 'F4', 'G1', 'H2', 'I3', 'J4', 'K1', 'X1', 'W1', 'R1', 'P3']

| Algorithm | Solution Path                                                                                                                                                                                                 | Cost (mi) | Nodes Reached | Nodes Explored | Time (s) |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------------- | -------------- | -------- |
| BFS       | [V3] → V2 → R4 → (R1) → I4 → (I3) → A3 → A2 → (G1) → A2 → A1 → (D1) → D4 → M3 → P1 → (P3) → K3 → (K1) → K4 → (J4) → H1 → (H2) → J2 → J3 → B3 → (B4) → B3 → F1 → (F4) → W4 → (W1) → X3 → (X1) → X2 → V4 → [V3] | 39.05     | 364289        | 361047         | 3.99215  |
| UCS       | [V3] → V4 → X2 → (X1) → X3 → (W1) → W4 → (F4) → U4 → U1 → (B4) → B3 → J3 → J2 → (H2) → H1 → (J4) → K4 → (K1) → K3 → (P3) → P1 → M3 → D4 → (D1) → A1 → A2 → (G1) → A2 → A3 → (I3) → I4 → (R1) → R4 → V2 → [V3] | 38.86     | 361878        | 386478         | 3.61029  |
| A\*       | [V3] → V4 → X2 → (X1) → X3 → (W1) → W4 → (F4) → U4 → U1 → (B4) → B3 → J3 → J2 → (H2) → H1 → (J4) → K4 → (K1) → K3 → (P3) → P1 → M3 → D4 → (D1) → A1 → A2 → (G1) → A2 → A3 → (I3) → I4 → (R1) → R4 → V2 → [V3] | 38.86     | 98013         | 85230          | 0.95041  |

## Performance Summary

The experimental results demonstrate that search time, nodes reached, and nodes explored grow exponentially as the number of delivery locations increases. When delivery locations exceed approximately 7 locations, BFS and UCS become significantly more computationally expensive than A\*.

**Key Findings:**

- **BFS and UCS** explore very large state spaces, making them slower and computationally intensive
- **A\*** runs in under a second for all test cases, performing almost **4x faster** than BFS and UCS on hard_4
- **A\***'s superior performance is due to its use of the straight-line heuristic, which enables effective tree pruning
- Tree pruning reduces frontier size, which significantly decreases exploration cost

## Author

**Jonesh Shrestha**

Original framework by Kenny Davila Castellanos

---

_This project demonstrates practical applications of artificial intelligence in route optimization and logistics planning._
