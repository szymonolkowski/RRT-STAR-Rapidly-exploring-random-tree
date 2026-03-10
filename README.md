# RRT* Path Planner with Potential Fields Support

An advanced implementation of the **RRT*** (Rapidly-exploring Random Tree Star) algorithm in Python. This algorithm is designed for determining optimal and collision-free paths in a 2D space (NED coordinates - North-East), dedicated to mobile robotics, drones, or autonomous vehicles (e.g., ASV - Autonomous Surface Vehicles).

It stands out from the classic RRT* by integrating an Artificial Potential Fields mechanism and intelligent space sampling using Delaunay triangulation.

## Key Features

* **Asymptotic Optimality (RRT*)**: The algorithm doesn't just find the first available path (like standard RRT), but optimizes it over time by "rewiring" nodes to minimize the total path cost.
* **Artificial Potential Fields**: Obstacles generate "influence zones". Paths running too close to obstacles receive drastic cost penalties, which naturally pushes the path away from hazards.
* **Intelligent Sampling (Delaunay Heuristics)**: Instead of randomly sampling points across the entire space, the algorithm can utilize a provided Delaunay mesh with weights, focusing on triangles located in the general direction of the goal.
* **Dynamic Target Node Count**: Calculates the optimal number of target nodes based on the search area size and the specified step size (`step_size`).
* **Early Stopping**: The algorithm can halt before reaching the node limit if the best cost hasn't improved for a sufficiently long time, saving computational resources.
* **Path Smoothing**: A post-processing module that removes unnecessary "zig-zags" typical of RRT* using raycasting and the reduction of densely packed points.
* **Starting Collision Tolerance**: If the starting point is located inside an obstacle, the algorithm ignores that specific obstacle only for the first move, allowing the robot to "drive out" of the collision state.

## Requirements

The algorithm uses standard Python environment libraries:

* `numpy`
* `math` (built-in)
* `random` (built-in)

## Data Structure

For the algorithm to work, it requires objects to be passed in a specific format:

1. **Start and Goal (`start`, `goal`)**: Objects must have a `ned` attribute, containing `n` (North) and `e` (East) coordinates.
2. **Obstacles (`potential_fields`)**: A list of objects possessing center coordinates `n`, `e`, and a radius `r`.

## Usage (Quick Start)

```python
import numpy as np
# Assuming the Node and RRTStar classes are in the rrt_star.py file
from rrt_star import RRTStar

# 1. Preparing mock structures for start and goal
class Coordinate:
    def __init__(self, n, e):
        self.n = n
        self.e = e

class Waypoint:
    def __init__(self, n, e):
        self.ned = Coordinate(n, e)

class Obstacle:
    def __init__(self, n, e, r):
        self.n = n
        self.e = e
        self.r = r

start_node = Waypoint(0.0, 0.0)
goal_node = Waypoint(20.0, 20.0)
obstacles = [Obstacle(10.0, 10.0, 3.0), Obstacle(5.0, 15.0, 2.0)]

# 2. Planner initialization
planner = RRTStar(
    start=start_node,
    goal=goal_node,
    delaunay=None,            # Optional object with scipy.spatial.Delaunay triangulation
    weighted_random=None,     # Optional weights for the triangles
    potential_fields=obstacles,
    step_size=1.0,
    iter_scale=10,
    safety_margin=0.5,        # Additional safety buffer around obstacles
    repulsive_weight=20.0     # "Repulsive" strength of the potential fields
)

# 3. Running the planner (with optional path smoothing)
planner.plan(post_processing=True)

# 4. Retrieving results
if planner.path:
    print(f"Path found! Length (nodes): {len(planner.path)}")
    print(f"Best cost: {planner.best_cost:.2f}")
    for node in planner.path:
        print(f" -> N: {node.n:.2f}, E: {node.e:.2f}")
else:
    print("Failed to find a path.")

```

## `RRTStar` Class Parameterization

Here is a detailed description of all the parameters accepted by the `RRTStar` constructor:

* **`start`**: *(Object)* The starting waypoint/node. It must contain a `ned` attribute with `n` and `e` coordinates.
* **`goal`**: *(Object)* The target destination waypoint/node. It must contain a `ned` attribute with `n` and `e` coordinates.
* **`delaunay`**: *(scipy.spatial.Delaunay or None)* Optional Delaunay triangulation object used to bound, filter, and guide the random sampling of new nodes.
* **`weighted_random`**: *(list/numpy.ndarray or None)* Optional list of probabilities/weights corresponding to Delaunay triangles, allowing prioritized sampling in specific areas (e.g., towards the goal).
* **`potential_fields`**: *(list)* List of obstacle objects. Each object must have `n`, `e`, and `r` (radius) attributes to act as repulsive zones.
* **`step_size`**: *(float, default=0.25)* Maximum length of a single branch (edge) added to the RRT* tree per iteration.
* **`iter_scale`**: *(int, default=5)* A multiplier affecting the dynamic calculation of the maximum target number of nodes, scaled based on the calculated search area.
* **`logger`**: *(logging.Logger or None)* Optional logger object for outputting debug, warning, and info messages during the planning process.
* **`check_delaunay`**: *(bool, default=False)* If set to `True`, the algorithm strictly verifies that newly generated nodes and paths fall within the valid bounds of the provided Delaunay simplices.
* **`safety_margin`**: *(float, default=0.4)* An extra safety buffer distance added to the defined radius (`r`) of every obstacle to prevent clipping.
* **`repulsive_weight`**: *(float, default=15.0)* The penalty weight applied by the artificial potential fields. It scales the added cost when passing close to obstacles. Influences the cost formula: $penalty = 0.5 \times weight \times term^2$.
* **`goal_region_radius`**: *(float, default=0.35)* The acceptance radius around the target goal. A node generated within this distance from the goal is considered to have successfully reached it.
* **`search_radius`**: *(float, default=1.0)* The maximum radius within which the RRT* algorithm searches for neighboring nodes during the tree optimization ("rewiring") phase.

Would you like me to review your Python code for any potential optimizations, or is the documentation ready to be pushed to your repository?
