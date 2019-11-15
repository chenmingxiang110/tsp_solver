import numpy as np
from iter_solver import calculate_distance_matrix, auto_solver, plot_route

coords = []
for _ in range(50):
    coords.append(np.random.random(2))
coords = np.array(coords)

distance_matrix = calculate_distance_matrix(coords)
d, r = auto_solver(distance_matrix, n_iter=1000, local_search_iter=100,
                   init_route=None, back_to_origin=False, verbose_step=1)
print(d)
print(r)
plot_route(coords, r)