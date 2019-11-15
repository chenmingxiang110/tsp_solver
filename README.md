# Travelling Salesman Problem (TSP) Solver Using Iterations
Solving tsp (Travelling Salesman Problem) using ruin &amp; recreate method.

The following library is required to use the script:
- Numpy

The following library is required to plot the route:
- Matplotlib

Example Code:

```
import numpy as np
from iter_solver import calculate_distance_matrix, auto_solver, plot_route

# Randomize 50 points for testing
coords = []
for _ in range(50):
    coords.append(np.random.random(2))
coords = np.array(coords)

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(coords)

# Solve the TSP problem with ruin & recreate method
d, r = auto_solver(distance_matrix, n_iter=1000, local_search_iter=100,
                   init_route=None, back_to_origin=False, verbose_step=1)
print(d)
print(r)
plot_route(coords, r)
```

By using the auto_solver function, a sub-optimal route and its distance can be found. If verbose_step is None (default), then the solver will remain silent. If, for instance, it is set to 100 and the total number of iteration is 100,000, then it will print out the current best solution's distance every 100 step.

The main steps of solving a tsp include 4 parts:

1. Obtain the distance matrix

The Distance Matrix can be calculated using the function "calculate_distance_matrix" which is provided with the script. The shape of the matrix is (number_of_nodes, number_of_nodes). Let's say we have three points A(0,0), B(0,3), and C(4,0), then the distance matrix would be:

```
D = [[0,3,4],
     [3,0,5],
     [4,5,0]]
```

You can also input a customized distance matrix. It do not need to follow the triangle rule, and the distance from A to B and B to A can be different.

2. Initialize with a initial route. 

The route can be obtained using some greedy algorithm. For example: https://github.com/dmishin/tsp-solver. But the final result will be much better especially when number of vertices is large. In my tests, the average improvement (in 1000 iterations) comparing to dmishin's result is as follows (the coordinates of points are initialized randomly between 0 and 1):

|# of Points |TSP Solver |Average Distance |
|--- |--- |--- |
|20|dmishin's|4.004|
||R&R|3.822|
|50|dmishin's|6.094|
||R&R|5.676|
|100|dmishin's|8.233|
||R&R|7.671|
|200|dmishin's|11.475|
||R&R|10.801|

If the init_route is None, then it will be initialized with a random route.

3. Iteration.

If back_to_origin is set to true, then the route should start from the first point, and going back to the point after traveled all the nodes. If back_to_origin is set to false, then the starting point is the first one, and the end point is the last one. The algorithm will optimize the route through iteration with ruin & recreate strategy, which means, in every single iteration, delete some nodes from the route and insert them back using some sorting mechanism. The main idea of this algorithm is similar to this paper: Slack Induction by String Removals for Vehicle Routing Problems, https://lirias.kuleuven.be/handle/123456789/624431.

4. Optimization.

After iteration, in case there is some better local solutions to some sub-routes, the algorithm will break the route into pieces and try to do some quick iterations. If local_search_iter is set to 0, then this step will be skipped.
