import time
import copy
import numpy as np

def calculate_distance_matrix(coords):
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    return distance_matrix

def sisr_vrp(data,
             vehicle_capcity,
             n_iter,
             n_iter_fleet=None,
             fleet_gap=None,
             init_T=100.0,
             final_T=1.0,
             c_bar=10.0,
             L_max=10.0,
             m_alpha=0.01,
             obj_n_routes=None,
             init_route=None,
             verbose_step=None,
             test_obj=None):
    """
    data: An numpy array with a shape of (N, 6), where N is the number of customers.
          Each column represents x coord, y coord, demand, ready time, due time, and
          service time.
    """
    
    def get_route_distance(distance_matrix, route):
        r = [0]+route+[0]
        result = np.sum([distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)])
        return result
    
    def get_routes_distance(distance_matrix, routes):
        total_distance = 0
        for route in routes:
            r = [0]+route+[0]
            total_distance += np.sum([distance_matrix[r[i],r[i+1]] for i in range(len(r)-1)])
        return total_distance
    
    def get_neighbours(distance_matrix):
        n_vertices = distance_matrix.shape[0]
        neighbours = []
        for i in range(n_vertices):
            index_dist = [(j, distance_matrix[i][j]) for j in range(n_vertices)]
            sorted_index_dist = sorted(index_dist, key=lambda x: x[1])
            neighbours.append([x[0] for x in sorted_index_dist])
        return neighbours
    
    def ruin(last_routes, neighbours, in_absents=None, isHugeRuin=False):

        def remove_nodes(tr, l_t, c, m):
            def string_removal(tr, l_t, c):
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t)
                range2 = min(i_c, len(tr)-l_t)+1
                start = np.random.randint(range1, range2)
                return tr[start:start+l_t]
            def split_removal(tr, l_t, c, m):
                additional_l = min(m, len(tr)-l_t)
                l_t_m = l_t+additional_l
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t_m)
                range2 = min(i_c, len(tr)-l_t_m)+1
                start = np.random.randint(range1, range2)
                potential_removal = tr[start:start+l_t_m]
                return [potential_removal[i] for i in np.random.choice(l_t_m, l_t, replace=False)]
            if np.random.random()<0.5:
                newly_removed = string_removal(tr, l_t, c)
            else:
                newly_removed = split_removal(tr, l_t, c, m)
                if m<(len(tr)-l_t) or np.random.random()>m_alpha:
                    m+=1
            return m, newly_removed
        
        def find_t(last_routes, c):
            for i in range(len(last_routes)):
                if c in last_routes[i]: return i
            return None
        
        def routes_summary(last_routes, absents):
            current_routes = []
            for r in last_routes:
                new_r = [x for x in r if x not in absents]
                if len(new_r)>0:
                    current_routes.append(new_r)
            return current_routes
        
        m = 1
        l_s_max = min(L_max, np.mean([len(x) for x in last_routes]))
        k_s_max = 4.0*c_bar/(1.0+l_s_max)-1.0
        k_s = int(np.random.random()*k_s_max+1.0)
        c_seed = int(np.random.random()*len(data))
        if in_absents is None:
            absents = []
        else:
            absents = copy.deepcopy(in_absents)
        ruined_t_indices = set([])
        for c in neighbours[c_seed]:
            if len(ruined_t_indices) >= k_s: break
            if c not in absents and c!=0:
                t = find_t(last_routes, c)
                if t in ruined_t_indices: continue
                if isHugeRuin and np.random.random()>0.5:
                    newly_removed = last_routes[t]
                else:
                    l_t_max = min(l_s_max, len(last_routes[t]))
                    l_t = int(np.random.random()*l_t_max+1.0)
                    m, newly_removed = remove_nodes(last_routes[t], l_t, c, m)
                absents = absents+newly_removed
                ruined_t_indices.add(t)
        current_routes = routes_summary(last_routes, absents)
        return current_routes, absents
    
    def recreate(data, distance_matrix, current_routes, absents, last_length=None):
        
        def checkValid_legacy(complete_r):
            t_current = 0
            for j in range(0,len(complete_r)-1):
                t_current+=distance_matrix[complete_r[j]][complete_r[j+1]]
                t_current=max(data[complete_r[j+1]][3], t_current)
                if t_current<=data[complete_r[j+1]][4]:
                    t_current+=data[complete_r[j+1]][-1]
                else:
                    return False
            return True
        
        def getValid_legacy(r,c):
            dist = get_route_distance(distance_matrix, r)
            complete_r = [0]+r+[0]
            valids = []
            tmp_t=0
            for i in range(len(complete_r)-1):
                tmp_t=max(tmp_t, data[complete_r[i]][3])
                tmp_t+=data[complete_r[i]][-1]
                if tmp_t+distance_matrix[complete_r[i]][c]>data[c][4]: break
                new_r = r[:i]+[c]+r[i:]
                if checkValid_legacy([0]+new_r+[0]):
                    new_d = get_route_distance(distance_matrix, new_r)
                    valids.append((i,new_d-dist))
                tmp_t+=distance_matrix[complete_r[i]][complete_r[i+1]]
            return valids
        
        def route_add(current_routes, c, adding_position):
            if adding_position[0]==-1:
                current_routes = current_routes+[[c]]
            else:
                chg_r = current_routes[adding_position[0]]
                new_r = chg_r[:adding_position[1]]+[c]+chg_r[adding_position[1]:]
                current_routes[adding_position[0]] = new_r
            return current_routes
        
        # 1. Sort the absent nodes with some methods, here only use random as a placeholder
        absents = [absents[i] for i in np.random.choice(len(absents), len(absents), replace=False)]
        if last_length is not None: rests = []
        for c in absents:
            probable_place = []
            for ir,r in enumerate(current_routes):
                # assert capacity
                if (np.sum([data[x][2] for x in r])+data[c][2])>vehicle_capcity: continue
                valids = getValid_legacy(r,c)
                for v in valids:
                    probable_place.append((ir,v[0],v[1]))
            if len(probable_place)==0:
                if last_length is not None and len(current_routes)>=last_length:
                    rests.append(c)
                    continue
                adding_position = (-1,-1,1)
            else:
                adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
#             probable_place.append((-1,-1,2*distance_matrix[0,c]))
#             adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
            current_routes = route_add(current_routes, c, adding_position)
        if last_length is not None: return current_routes, rests
        return current_routes
    
    def fleet_min(n, data, distance_matrix, neighbours, routes, verbose_step):
        absents = []
        absent_c = np.zeros(distance_matrix.shape[0])
        best_routes = copy.deepcopy(routes)
        last_routes = copy.deepcopy(routes)
        for i in range(n):
            current_routes, new_absents = ruin(last_routes, neighbours, absents)
            current_routes, new_absents = recreate(data, distance_matrix,
                                                              current_routes, new_absents,
                                                              last_length=len(last_routes))
            if len(new_absents)==0:
                best_routes = copy.deepcopy(current_routes)
                sumabs = sorted([(t, np.sum(absent_c[t])) for t in best_routes], key=lambda x: x[1])
                absents = sumabs[0][0]
                last_routes = [x[0] for x in sumabs[1:]]
                current_routes = copy.deepcopy(last_routes)
            elif len(new_absents)<len(absents) or np.sum(absent_c[new_absents])<np.sum(absent_c[absents]):
                last_routes = copy.deepcopy(current_routes)
                absents = copy.deepcopy(new_absents)
            if len(new_absents)>0:
                tmp = np.zeros(absent_c.shape)
                tmp[new_absents]=1
                absent_c = absent_c+tmp
            if verbose_step is not None and (i+1)%verbose_step==0:
                print("fleet_min", i+1, np.round((i+1)/n*100,4), "%:", len(best_routes))
        if verbose_step is not None and n%verbose_step!=0: print(i+1, "100.0 %:", len(best_routes))
        return best_routes
    
    if obj_n_routes is not None: assert fleet_gap is not None
    alpha_T = (final_T/init_T)**(1.0/n_iter)
    
    coords = data[:,:2]
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    
    if init_route is not None:
        best_routes = copy.deepcopy(init_route)
    else:
        best_routes = [[i] for i in range(1,len(data))]
    best_distance = get_routes_distance(distance_matrix, best_routes)
    last_routes = copy.deepcopy(best_routes)
    last_distance = get_routes_distance(distance_matrix, best_routes)
    neighbours = get_neighbours(distance_matrix)
    print(len(best_routes), best_distance)
    
#     if n_iter_fleet is None: n_iter_fleet=int(max(n_iter*0.1, 1))
    if (n_iter_fleet is not None) and (n_iter_fleet > 0):
        last_routes = fleet_min(n_iter_fleet, data, distance_matrix, neighbours, best_routes, verbose_step)
        last_distance = get_routes_distance(distance_matrix, last_routes)

        if last_distance<best_distance:
            best_distance = last_distance
            best_routes = last_routes
        print(len(best_routes), best_distance)
    temperature = init_T
    
    for i_iter in range(n_iter):
        if obj_n_routes is not None and len(last_routes)>obj_n_routes and (i_iter+1)%fleet_gap==0:
            current_routes = fleet_min(n_iter_fleet, data, distance_matrix, neighbours, last_routes, verbose_step)
        else:
            current_routes, absents = ruin(last_routes, neighbours)
            current_routes = recreate(data, distance_matrix, current_routes, absents)
        
        current_distance = get_routes_distance(distance_matrix, current_routes)
        if len(current_routes)<len(best_routes) or \
           (current_distance<(last_distance-temperature*np.log(np.random.random())) and \
            len(current_routes)<=len(best_routes)):
            
            if len(current_routes)<len(best_routes) or current_distance<best_distance:
                best_distance = current_distance
                best_routes = current_routes
                if test_obj is not None and best_distance<test_obj:
                    break
            last_distance = current_distance
            last_routes = current_routes
        temperature*=alpha_T
        
        if verbose_step is not None and (i_iter+1)%verbose_step==0:
            print(i_iter+1, np.round((i_iter+1)/n_iter*100,4), "%:",
                  len(best_routes), len(last_routes), len(current_routes),
                  best_distance, last_distance, current_distance)
    
    if verbose_step is not None and n_iter%verbose_step!=0: print(i_iter+1, "100.0 %:", best_distance)
    return best_distance, best_routes

def sisr_cvrp(data,
              vehicle_capcity,
              n_iter,
              n_iter_fleet=None,
              fleet_gap=None,
              init_T=100.0,
              final_T=1.0,
              c_bar=10.0,
              L_max=10.0,
              m_alpha=0.01,
              obj_n_routes=None,
              init_route=None,
              verbose_step=None,
              test_obj=None):
    """
    data: An numpy array with a shape of (N, 3), where N is the number of customers.
          Each column represents x coord, y coord, demand, ready time, due time, and
          service time.
    """
    
    def get_route_distance(distance_matrix, route):
        r = [0]+route+[0]
        result = np.sum([distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)])
        return result
    
    def get_routes_distance(distance_matrix, routes):
        total_distance = 0
        for route in routes:
            r = [0]+route+[0]
            total_distance += np.sum([distance_matrix[r[i],r[i+1]] for i in range(len(r)-1)])
        return total_distance
    
    def get_neighbours(distance_matrix):
        n_vertices = distance_matrix.shape[0]
        neighbours = []
        for i in range(n_vertices):
            index_dist = [(j, distance_matrix[i][j]) for j in range(n_vertices)]
            sorted_index_dist = sorted(index_dist, key=lambda x: x[1])
            neighbours.append([x[0] for x in sorted_index_dist])
        return neighbours
    
    def ruin(last_routes, neighbours, in_absents=None, isHugeRuin=False):

        def remove_nodes(tr, l_t, c, m):
            def string_removal(tr, l_t, c):
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t)
                range2 = min(i_c, len(tr)-l_t)+1
                start = np.random.randint(range1, range2)
                return tr[start:start+l_t]
            def split_removal(tr, l_t, c, m):
                additional_l = min(m, len(tr)-l_t)
                l_t_m = l_t+additional_l
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t_m)
                range2 = min(i_c, len(tr)-l_t_m)+1
                start = np.random.randint(range1, range2)
                potential_removal = tr[start:start+l_t_m]
                return [potential_removal[i] for i in np.random.choice(l_t_m, l_t, replace=False)]
            if np.random.random()<0.5:
                newly_removed = string_removal(tr, l_t, c)
            else:
                newly_removed = split_removal(tr, l_t, c, m)
                if m<(len(tr)-l_t) or np.random.random()>m_alpha:
                    m+=1
            return m, newly_removed
        
        def find_t(last_routes, c):
            for i in range(len(last_routes)):
                if c in last_routes[i]: return i
            return None
        
        def routes_summary(last_routes, absents):
            current_routes = []
            for r in last_routes:
                new_r = [x for x in r if x not in absents]
                if len(new_r)>0:
                    current_routes.append(new_r)
            return current_routes
        
        m = 1
        l_s_max = min(L_max, np.mean([len(x) for x in last_routes]))
        k_s_max = 4.0*c_bar/(1.0+l_s_max)-1.0
        k_s = int(np.random.random()*k_s_max+1.0)
        c_seed = int(np.random.random()*len(data))
        if in_absents is None:
            absents = []
        else:
            absents = copy.deepcopy(in_absents)
        ruined_t_indices = set([])
        for c in neighbours[c_seed]:
            if len(ruined_t_indices) >= k_s: break
            if c not in absents and c!=0:
                t = find_t(last_routes, c)
                if t in ruined_t_indices: continue
                if isHugeRuin and np.random.random()>0.5:
                    newly_removed = last_routes[t]
                else:
                    l_t_max = min(l_s_max, len(last_routes[t]))
                    l_t = int(np.random.random()*l_t_max+1.0)
                    m, newly_removed = remove_nodes(last_routes[t], l_t, c, m)
                absents = absents+newly_removed
                ruined_t_indices.add(t)
        current_routes = routes_summary(last_routes, absents)
        return current_routes, absents
    
    def recreate(data, dist_m, current_routes, absents, last_length=None):
        
        def route_add(dist_m, current_routes, c, adding_position):
            if adding_position[0]==-1:
                current_routes = current_routes+[[c]]
            else:
                chg_r = current_routes[adding_position[0]]
                new_r = chg_r[:adding_position[1]]+[c]+chg_r[adding_position[1]:]
                current_routes[adding_position[0]] = new_r
            return current_routes
        
        # 1. Sort the absent nodes with some methods, here only use random as a placeholder
        absents = [absents[i] for i in np.random.choice(len(absents), len(absents), replace=False)]
        if last_length is not None: rests = []
        for c in absents:
            probable_place = []
            for ir,r in enumerate(current_routes):
                # assert capacity
                if (np.sum([data[x][2] for x in r])+data[c][2])>vehicle_capcity: continue
                complete_r = [0]+r+[0]
                for iri in range(len(r)+1):
                    probable_place.append((ir,iri,dist_m[iri,c]+dist_m[c,iri+1]-dist_m[iri,iri+1]))
            if len(probable_place)==0:
                if last_length is not None and len(current_routes)>=last_length:
                    rests.append(c)
                    continue
                adding_position = (-1,-1,1)
            else:
                adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
#             probable_place.append((-1,-1,2*distance_matrix[0,c]))
#             adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
            current_routes = route_add(dist_m, current_routes, c, adding_position)
        if last_length is not None: return current_routes, rests
        return current_routes
    
    def fleet_min(n, data, distance_matrix, neighbours, routes, verbose_step):
        absents = []
        absent_c = np.zeros(distance_matrix.shape[0])
        best_routes = copy.deepcopy(routes)
        last_routes = copy.deepcopy(routes)
        for i in range(n):
            current_routes, new_absents = ruin(last_routes, neighbours, absents)
            current_routes, new_absents = recreate(data, distance_matrix,
                                                              current_routes, new_absents,
                                                              last_length=len(last_routes))
            if len(new_absents)==0:
                best_routes = copy.deepcopy(current_routes)
                sumabs = sorted([(t, np.sum(absent_c[t])) for t in best_routes], key=lambda x: x[1])
                absents = sumabs[0][0]
                last_routes = [x[0] for x in sumabs[1:]]
                current_routes = copy.deepcopy(last_routes)
            elif len(new_absents)<len(absents) or np.sum(absent_c[new_absents])<np.sum(absent_c[absents]):
                last_routes = copy.deepcopy(current_routes)
                absents = copy.deepcopy(new_absents)
            if len(new_absents)>0:
                tmp = np.zeros(absent_c.shape)
                tmp[new_absents]=1
                absent_c = absent_c+tmp
            if verbose_step is not None and (i+1)%verbose_step==0:
                print("fleet_min", i+1, np.round((i+1)/n*100,4), "%:", len(best_routes))
        if verbose_step is not None and n%verbose_step!=0: print(i+1, "100.0 %:", len(best_routes))
        return best_routes
    
    alpha_T = (final_T/init_T)**(1.0/n_iter)
    
    coords = data[:,:2]
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    
    if init_route is not None:
        best_routes = copy.deepcopy(init_route)
    else:
        best_routes = [[i] for i in range(1,len(data))]
    best_distance = get_routes_distance(distance_matrix, best_routes)
    last_routes = copy.deepcopy(best_routes)
    last_distance = get_routes_distance(distance_matrix, best_routes)
    neighbours = get_neighbours(distance_matrix)
    print(len(best_routes), best_distance)
    
#     if n_iter_fleet is None: n_iter_fleet=int(max(n_iter*0.1, 1))
    if (n_iter_fleet is not None) and (n_iter_fleet > 0):
        last_routes = fleet_min(n_iter_fleet, data, distance_matrix, neighbours, best_routes, verbose_step)
        last_distance = get_routes_distance(distance_matrix, last_routes)

        if last_distance<best_distance:
            best_distance = last_distance
            best_routes = last_routes
        print(len(best_routes), best_distance)
    temperature = init_T
    
    for i_iter in range(n_iter):
        if obj_n_routes is not None and len(last_routes)>obj_n_routes and (i_iter+1)%fleet_gap==0:
            current_routes = fleet_min(n_iter_fleet, data, distance_matrix, neighbours, last_routes, verbose_step)
        else:
            current_routes, absents = ruin(last_routes, neighbours)
            current_routes = recreate(data, distance_matrix, current_routes, absents)
        
        current_distance = get_routes_distance(distance_matrix, current_routes)
        if len(current_routes)<len(best_routes) or \
           (current_distance<(last_distance-temperature*np.log(np.random.random())) and \
            len(current_routes)<=len(best_routes)):
            
            if len(current_routes)<len(best_routes) or current_distance<best_distance:
                best_distance = current_distance
                best_routes = current_routes
                if test_obj is not None and best_distance<test_obj:
                    break
            last_distance = current_distance
            last_routes = current_routes
        temperature*=alpha_T
        
        if verbose_step is not None and (i_iter+1)%verbose_step==0:
            print(i_iter+1, np.round((i_iter+1)/n_iter*100,4), "%:",
                  len(best_routes), len(last_routes), len(current_routes),
                  best_distance, last_distance, current_distance)
    
    if verbose_step is not None and n_iter%verbose_step!=0: print(i_iter+1, "100.0 %:", best_distance)
    return best_distance, best_routes

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
    
    def adding_node(curr_r, node, dist_m):
        cost = dist_m[curr_r[0],node] + dist_m[node,curr_r[1]] - dist_m[curr_r[0],curr_r[1]]
        index = 1
        curr_c = dist_m[curr_r[1:-1],node]+dist_m[node,curr_r[2:]]-dist_m[curr_r[1:-1],curr_r[2:]]
        index_c = np.argmin(curr_c)
        if curr_c[index_c]>cost:
            return curr_r[:1]+[node]+curr_r[1:]
        return curr_r[:(index_c+2)]+[node]+curr_r[(index_c+2):]
    
#         for i in range(2,len(current_route)):
#             current_cost = distance_matrix[current_route[i-1],node] + \
#                            distance_matrix[node,current_route[i]] - \
#                            distance_matrix[current_route[i-1],current_route[i]]
#             if current_cost<cost:
#                 cost = current_cost
#                 index = i
#         return current_route[:index]+[node]+current_route[index:]
    
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

    def adding_node(curr_r, node, dist_m):
        cost = dist_m[curr_r[-1],node] + dist_m[node,curr_r[0]] - dist_m[curr_r[-1],curr_r[0]]
        index = 0
        curr_c = dist_m[curr_r[:-1],node]+dist_m[node,curr_r[1:]]-dist_m[curr_r[:-1],curr_r[1:]]
        index_c = np.argmin(curr_c)
        if curr_c[index_c]>cost:
            return [node]+curr_r
        return curr_r[:(index_c+1)]+[node]+curr_r[(index_c+1):]

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
    scores = get_scores()

    for i_iter in range(n_iter):
        if verbose_step is not None and (i_iter+1)%verbose_step==0:
            print(i_iter+1, np.round((i_iter+1)/n_iter*100,3), "%:", best_distance)
            
        # start_time = time.time()
        n_destroy = min(distance_matrix.shape[0]-1, max(1, int(dropRate*distance_matrix.shape[0])))
        # print(time.time()-start_time)
        
        # start_time = time.time()
        absent_nodes, curr_r = destroy_nodes(n_destroy, last_route, scores)
        # print(time.time()-start_time)
        
        # start_time = time.time()
        s_nodes = sort_absent_nodes(absent_nodes, scores)
        # print(time.time()-start_time)
        
        # start_time = time.time()
        for node in s_nodes:
            curr_r = adding_node(curr_r, node, distance_matrix)
        # print(time.time()-start_time)
        
        # start_time = time.time()
        current_distance = get_route_distance(distance_matrix, curr_r+[curr_r[0]])
        if record: recorded_dists.append(current_distance)
        if current_distance<(best_distance-temperature*np.log(np.random.random())):
            if current_distance<best_distance:
                best_distance = current_distance
                best_route = curr_r
            last_route = curr_r
        temperature*=alpha_T
        dropRate*=alpha_drop
        # print(time.time()-start_time)
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
            if verbose: print("Optimizing:", route_length, i_unchanged+1, '/', multiple*2, best_distance)
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
            if verbose: print("Optimizing:", route_length, start, "isChanged", isChanged, best_distance)
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
    if verbose_step is not None:
        assert verbose_step>=1
    
    if back_to_origin:
        d, r = sisr_tsp(distance_matrix, n_iter=n_iter,
                        init_route=init_route, verbose_step=verbose_step)
        if local_search_iter>0:
            d, r = optimize_route(distance_matrix, r, local_search_iter, verbose_step is not None)
        return d, r
    else:
        d, r = sisr_tsp_e2e(distance_matrix, n_iter=n_iter,
                            init_route=init_route, verbose_step=verbose_step)
        if local_search_iter>0:
            d, r = optimize_route_e2e(distance_matrix, r, local_search_iter, verbose_step is not None)
        return d, r