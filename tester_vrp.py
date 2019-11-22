import time
import numpy as np
from iter_solver import sisr_vrp

def parse_vrp_question(file_path):
    data = []
    vehicle_capcity = -1
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            sline = line.strip()
            if len(sline)==0: continue
            if i==3: vehicle_capcity = int(sline.split()[1])
            if i>=6:
                data.append([int(i) for i in sline.split()][1:])
            i+=1
    return vehicle_capcity, np.array(data)

vehicle_capcity, data = parse_vrp_question("data/homberger_200_customer_instances/C1_2_1.TXT")

print(vehicle_capcity)
print(data.shape)
print("------------------")

start_time = time.time()
np.random.seed(0)
d, best_routes = sisr_vrp(data, vehicle_capcity, n_iter=1000, init_T=100.0, final_T=1.0,
                          n_iter_fleet=50, fleet_gap=100, obj_n_routes=20,
                          init_route = None, verbose_step=10)
time_cost = time.time()-start_time
print("time_cost", time_cost)
print("distance", d)