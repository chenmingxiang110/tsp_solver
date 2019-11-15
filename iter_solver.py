import time
import numpy as np

def calculate_distance_matrix(coords):
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    return distance_matrix

def sisr_tsp_e2e(distance_matrix,
                 n_iter=10000,
                 init_T=100.0,
                 final_T=1,
                 init_dropRate=0.8,
                 final_dropRate=0.4,
                 init_route=None,
                 record=False,
                 verbose_step=None):
    
    ###########################################################################
    # Solving TSP problems with simulated annealing and string removal.       #
    ###########################################################################
    
    def get_route_distance(distance_matrix, route):
        return np.sum([distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1)])
    
    def get_scores():
        s = np.ones(distance_matrix.shape[0])
        s[0] = 0
        s[-1] = 0
        s/=np.sum(s)
        return s
    
    def destroy_nodes(n_destroy, last_route, scores):
        destroyed = list(np.random.choice(len(scores), n_destroy, replace=False, p=scores))
        new_route = [x for x in last_route if x not in set(destroyed)]
        return destroyed, new_route
    
    def sort_absent_nodes(absent_nodes, scores):
        ab_scores = scores[absent_nodes]
        ab_scores/= np.sum(ab_scores)
        absent_nodes_indices = list(np.random.choice(len(absent_nodes),
                                                     len(absent_nodes),
                                                     replace=False, p=ab_scores))
        return [absent_nodes[i] for i in absent_nodes_indices]
    
    def adding_node(current_route, node, distance_matrix):
        cost = distance_matrix[current_route[0],node] + \
               distance_matrix[node,current_route[1]] - \
               distance_matrix[current_route[0],current_route[1]]
        index = 1
        for i in range(2,len(current_route)):
            current_cost = distance_matrix[current_route[i-1],node] + \
                           distance_matrix[node,current_route[i]] - \
                           distance_matrix[current_route[i-1],current_route[i]]
            if current_cost<cost:
                cost = current_cost
                index = i
        return current_route[:index]+[node]+current_route[index:]
    
    ###########################################################################
    
    alpha_T = (final_T/init_T)**(1.0/n_iter)
    alpha_drop = (final_dropRate/init_dropRate)**(1.0/n_iter)
    
    if init_route is None:
        best_route = [x for x in range(distance_matrix.shape[0])]
    else:
        best_route = init_route
    
    best_distance = get_route_distance(distance_matrix, best_route)
    last_route = best_route
    temperature = init_T
    dropRate = init_dropRate
    recorded_dists = []
    
    for i_iter in range(n_iter):
        if verbose_step is not None and (i_iter+1)%verbose_step==0:
            print(i_iter+1, np.round((i_iter+1)/n_iter*100,3), "%:", best_distance)
        scores = get_scores()
        n_destroy = min(distance_matrix.shape[0]-1, max(1, int(dropRate*distance_matrix.shape[0])))
        absent_nodes, current_route = destroy_nodes(n_destroy, last_route, scores)
        sorted_absent_nodes = sort_absent_nodes(absent_nodes, scores)
        for node in sorted_absent_nodes:
            current_route = adding_node(current_route, node, distance_matrix)
        current_distance = get_route_distance(distance_matrix, current_route)
        if record: recorded_dists.append(current_distance)
        if current_distance<(best_distance-temperature*np.log(np.random.random())):
            if current_distance<best_distance:
                best_distance = current_distance
                best_route = current_route
            last_route = current_route
        temperature*=alpha_T
        dropRate*=alpha_drop
    if verbose_step is not None: print(i_iter+1, "100 %:", best_distance)
    
    bri = best_route.index(0)
    if record:
        return best_distance, recorded_dists, best_route[bri:]+best_route[:bri]+[0]
    return best_distance, best_route

def sisr_tsp(distance_matrix,
             n_iter=10000,
             init_T=100.0,
             final_T=1,
             init_dropRate=0.8,
             final_dropRate=0.4,
             init_route=None,
             record=False,
             verbose_step=None):

    def get_route_distance(distance_matrix, route):
        return np.sum([distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1)])

    def get_scores():
        return np.ones(distance_matrix.shape[0])/float(distance_matrix.shape[0])

    def destroy_nodes(n_destroy, last_route, scores):
        destroyed = list(np.random.choice(len(scores), n_destroy, replace=False, p=scores))
        new_route = [x for x in last_route if x not in set(destroyed)]
        return destroyed, new_route

    def sort_absent_nodes(absent_nodes, scores):
        ab_scores = scores[absent_nodes]
        ab_scores/= np.sum(ab_scores)
        absent_nodes_indices = list(np.random.choice(len(absent_nodes),
                                                     len(absent_nodes),
                                                     replace=False, p=ab_scores))
        return [absent_nodes[i] for i in absent_nodes_indices]

    def adding_node(current_route, node, distance_matrix):
        cost = distance_matrix[current_route[-1],node] + \
               distance_matrix[node,current_route[0]] - \
               distance_matrix[current_route[-1],current_route[0]]
        index = 0
        for i in range(1,len(current_route)):
            current_cost = distance_matrix[current_route[i-1],node] + \
                           distance_matrix[node,current_route[i]] - \
                           distance_matrix[current_route[i-1],current_route[i]]
            if current_cost<cost:
                cost = current_cost
                index = i
        if index == 0:
            return [node]+current_route
        return current_route[:index]+[node]+current_route[index:]

    ###########################################################################

    alpha_T = (final_T/init_T)**(1.0/n_iter)
    alpha_drop = (final_dropRate/init_dropRate)**(1.0/n_iter)

    if init_route is None:
        best_route = [x for x in range(distance_matrix.shape[0])]
    else:
        best_route = init_route

    best_distance = get_route_distance(distance_matrix, best_route+[best_route[0]])
    last_route = best_route
    temperature = init_T
    dropRate = init_dropRate
    recorded_dists = []

    for i_iter in range(n_iter):
        if verbose_step is not None and (i_iter+1)%verbose_step==0:
            print(i_iter+1, np.round((i_iter+1)/n_iter*100,3), "%:", best_distance)
        scores = get_scores()
        n_destroy = min(distance_matrix.shape[0]-1, max(1, int(dropRate*distance_matrix.shape[0])))
        absent_nodes, current_route = destroy_nodes(n_destroy, last_route, scores)
        sorted_absent_nodes = sort_absent_nodes(absent_nodes, scores)
        for node in sorted_absent_nodes:
            current_route = adding_node(current_route, node, distance_matrix)
        current_distance = get_route_distance(distance_matrix, current_route+[current_route[0]])
        if record: recorded_dists.append(current_distance)
        if current_distance<(best_distance-temperature*np.log(np.random.random())):
            if current_distance<best_distance:
                best_distance = current_distance
                best_route = current_route
            last_route = current_route
        temperature*=alpha_T
        dropRate*=alpha_drop
    if verbose_step is not None: print(i_iter+1, "100 %:", best_distance)

    bri = best_route.index(0)
    if record:
        return best_distance, recorded_dists, best_route[bri:]+best_route[:bri]+[0]
    return best_distance, best_route[bri:]+best_route[:bri]+[0]

def optimize_route(distance_matrix, route, n_iter, verbose=False):
    
    def get_route_distance(distance_matrix, route):
        return np.sum([distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1)])

    if route[0]==route[-1]:
        best_route = route[:-1]
    else:
        best_route = route
    best_distance = get_route_distance(distance_matrix, best_route+[best_route[0]])
    
    multiple = 2
    route_length = int(distance_matrix.shape[0]/multiple)
    lower_bnd = min(n_iter/5, 100)
    while route_length>=20:
        i_unchanged = 0
        current_n_iter = int(max(2*n_iter/multiple, lower_bnd))
        while i_unchanged<(multiple*2):
            if verbose: print("Optimizing:", route_length, i_unchanged+1, '/', multiple*2)
            start = int(route_length/2)
            best_route = (best_route+best_route)[start:start+distance_matrix.shape[0]]
            sub_route = np.array(best_route[:route_length])
            sub_D = distance_matrix[sub_route][:,sub_route]
            sub_dist = get_route_distance(distance_matrix, sub_route)
            
            propose_dist, tsp_propose = sisr_tsp_e2e(sub_D, n_iter=current_n_iter)
            propose_route = sub_route[tsp_propose]
            
            if np.sum(np.abs(propose_route-sub_route))!=0 and propose_dist<sub_dist:
                best_route = list(propose_route)+best_route[route_length:]
                best_distance = get_route_distance(distance_matrix, best_route+[best_route[0]])
                i_unchanged=0
            else:
                i_unchanged+=1
        multiple *= 2
        route_length = int(distance_matrix.shape[0]/multiple)
    return best_distance, best_route

def optimize_route_e2e(distance_matrix, route, n_iter, verbose=False):
    def get_route_distance(distance_matrix, route):
        return np.sum([distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1)])

    best_route = route
    best_distance = get_route_distance(distance_matrix, best_route)
    
    multiple = 2
    route_length = int(distance_matrix.shape[0]/multiple)
    lower_bnd = min(n_iter/5, 100)
    while route_length>=20:
        isChanged = False
        current_n_iter = int(max(2*n_iter/multiple, lower_bnd))
        start = 0
        gap = int(max(route_length/2,1))
        flag = True
        while flag:
            if verbose: print("Optimizing:", route_length, start, "isChanged", isChanged)
            if (start+route_length)>len(best_route):
                end = len(best_route)
                flag = False
            else:
                end = start+route_length
            sub_route = np.array(best_route[start:end])
            sub_D = distance_matrix[sub_route][:,sub_route]
            sub_dist = get_route_distance(distance_matrix, sub_route)
            propose_dist, tsp_propose = sisr_tsp_e2e(sub_D, n_iter=current_n_iter)
            propose_route = sub_route[tsp_propose]
            
            if np.sum(np.abs(propose_route-sub_route))!=0 and propose_dist<sub_dist:
                best_route = best_route[:start]+list(propose_route)+best_route[end:]
                best_distance = get_route_distance(distance_matrix, best_route)
                isChanged = True
            
            start+=gap
        if not isChanged:
            multiple *= 2
            route_length = int(distance_matrix.shape[0]/multiple)
    return best_distance, best_route

def plot_route(coords, route):
    import matplotlib.pyplot as plt
    plt.plot(coords[0,0], coords[0,1], 'C2o')
    plt.plot(coords[1:,0], coords[1:,1], 'C1o')
    for i in range(len(coords)): plt.annotate(i, (coords[i,0],coords[i,1]))
    for i in range(len(route)-1):
        plt.plot([coords[route[i],0],coords[route[i+1],0]],
                 [coords[route[i],1],coords[route[i+1],1]],'C0-')
    plt.show()

def auto_solver(distance_matrix,
                n_iter=10000,
                local_search_iter=400,
                init_route=None,
                back_to_origin=True,
                verbose_step=None):
    
    assert len(distance_matrix.shape)==2
    assert distance_matrix.shape[0]==distance_matrix.shape[1]
    assert distance_matrix.shape[0]>2
    
    if back_to_origin:
        d, r = sisr_tsp(distance_matrix, n_iter=n_iter,
                        init_route=init_route, verbose_step=verbose_step)
        if local_search_iter>0:
            d, r = optimize_route(distance_matrix, r, n_iter, verbose_step is not None)
        return d, r
    else:
        d, r = sisr_tsp_e2e(distance_matrix, n_iter=n_iter,
                            init_route=init_route, verbose_step=verbose_step)
        if local_search_iter>0:
            d, r = optimize_route_e2e(distance_matrix, r, n_iter, verbose_step is not None)
        return d, r