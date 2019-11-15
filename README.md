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
