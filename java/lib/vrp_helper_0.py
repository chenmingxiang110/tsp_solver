import time
import copy
import math
import tqdm
from tqdm import trange
import numpy as np

# The following is used to shut up tensorflow
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_distance(vec1, vec2):
    return np.sum((vec1-vec2)**2)**0.5

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

def parse_vrp_answer(file_path):
    routes = []
    start = False
    with open(file_path, 'r', encoding = "ISO-8859-1") as f:
        for line in f:
            if start:
                r = line.strip().split(":")[1]
                r = [int(x) for x in r.split() if len(x)>0]
                routes.append(r)
            else:
                if "Solution" in line:
                    start=True
    return routes

def getQADict(folderPath_q, folderPath_a):
    result = {}
    if folderPath_q[-1]!="/": folderPath_q = folderPath_q+"/"
    if folderPath_a[-1]!="/": folderPath_a = folderPath_a+"/"
    que_s = [x for x in os.listdir(folderPath_q) if x[0]!='.']
    ans_s = [x for x in os.listdir(folderPath_a) if x[0]!='.']
    for que in que_s:
        pa = ""
        for ans in ans_s:
            name = que.lower()
            name = name.split(".")[0]
            name = name.split("-")[0]
            if name == ans[:len(name)]:
                pa = ans
                break
        if len(pa)>0:
            result[folderPath_q+que] = folderPath_a+pa
    return result

def check_vrp_routes_validity(vehicle_capcity, data, routes):
    coords = data[:,:2]
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    
    for r in routes:
        R = [0]+r+[0]
        t = 0
        demands = np.sum([data[x,2] for x in R])
        if demands>vehicle_capcity: return False
        for i in range(len(R)):
            if i==0:
                arrive_t = t
            else:
                arrive_t = t+distance_matrix[R[i-1],R[i]]
            start_t = max(data[R[i],3], arrive_t)
            due_t = data[R[i],4]
            serve_t = data[R[i],5]
            end_t = start_t+serve_t
            t=end_t
            if due_t<start_t: return False
    return True

def check_vrp_route_validity(vehicle_capcity, data, r):
    coords = data[:,:2]
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5

    R = [0]+r+[0]
    t = 0
    demands = np.sum([data[x,2] for x in R])
    if demands>vehicle_capcity:
        print("demands",demands,"vehicle_capcity",vehicle_capcity,"NOT VALID!")
    else:
        print("demands",demands,"vehicle_capcity",vehicle_capcity)

    print("Node", "arrive_time", "start_time", "due_time", "service_time", "end_time")
    for i in range(len(R)):
        if i==0:
            arrive_t = t
        else:
            arrive_t = t+distance_matrix[R[i-1],R[i]]
        start_t = max(data[R[i],3], arrive_t)
        due_t = data[R[i],4]
        serve_t = data[R[i],5]
        end_t = start_t+serve_t
        t=end_t
        if due_t>=start_t:
            print(R[i], np.round(arrive_t,2), np.round(start_t,2),
                  due_t, serve_t, np.round(end_t,2))
        else:
            print(R[i], np.round(arrive_t,2), np.round(start_t,2),
                  due_t, serve_t, np.round(end_t,2), "NOT VALID!")

def check_cvrp_route_validity(vehicle_capcity, data, r):
    R = [0]+r+[0]
    t = 0
    demands = np.sum([data[x,2] for x in R])
    if demands>vehicle_capcity:
        print("demands",demands,"vehicle_capcity",vehicle_capcity,"NOT VALID!")
    else:
        print("demands",demands,"vehicle_capcity",vehicle_capcity)

def get_routes_distance(coords, routes, distance_matrix=None):
    if distance_matrix is None:
        distance_matrix = np.zeros([len(coords),len(coords)])
        for i in range(len(coords)):
            coord = coords[i]
            distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5

    result = 0.0
    for route in routes:
        r = [0]+route+[0]
        result += np.sum([distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)])
    return result

def calculate_distance_matrix(coords):
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    return distance_matrix

def sisr_cvrp(data, capacity, n_iter, init_T=100.0, final_T=1.0, init_routes=None, verbose_step=None):
    alter = np.zeros([data.shape[0],3])
    alter[:,1] = 1000000
    data_reshape = np.concatenate([data, alter], axis=1)
    d, best_routes = sisr_vrp(data_reshape, capacity, n_iter=n_iter, init_T=init_T, final_T=final_T,
                              init_route=init_routes, verbose_step=verbose_step)
    return d, best_routes

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
    if n_iter>0: alpha_T = (final_T/init_T)**(1.0/n_iter)

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
#     print(len(best_routes), best_distance)

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
            current_routes = fleet_min(n_iter_fleet, data, distance_matrix, neighbours,
                                       last_routes, verbose_step)
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

def random_cvrp_legacy(n_nodes, n_clusters=None, demand_lowerBnd=1, demand_upperBnd=9):
    data = []
    # 如果 node 数量小于1000，那么边长为1000
    side_limit = int(max(100.01, n_nodes/10))
    
    if n_clusters is not None:
        assert n_clusters<n_nodes
        while len(data) < n_clusters:
            coord = [np.random.randint(0,side_limit), np.random.randint(0,side_limit)]
            flag = False
            for d in data:
                if coord[0] == d[0] and coord[1] == d[1]:
                    flag = True
                    break
            if flag: continue
            data.append([coord[0], coord[1],
                         np.random.randint(demand_lowerBnd, demand_upperBnd+1),])
        
        while len(data) < n_nodes:
            rnd = np.array([np.random.randint(-3,4), np.random.randint(-3,4)])
            coord = data[np.random.randint(len(data))][:2]+rnd
            if coord[0]<0 or coord[1]<0 or coord[0]>=side_limit or coord[1]>=side_limit: continue
            flag = False
            for d in data:
                if coord[0] == d[0] and coord[1] == d[1]:
                    flag = True
                    break
            if flag: continue
            data.append([coord[0], coord[1],
                         np.random.randint(demand_lowerBnd, demand_upperBnd+1),])
    else:
        while len(data) < n_nodes:
            coord = [np.random.randint(0,side_limit), np.random.randint(0,side_limit)]
            flag = False
            for d in data:
                if coord[0] == d[0] and coord[1] == d[1]:
                    flag = True
                    break
            if flag: continue
            data.append([coord[0], coord[1],
                         np.random.randint(demand_lowerBnd, demand_upperBnd+1),])
    data = np.array(data)
    return data

def random_vrptw_legacy(n_nodes, n_clusters=None, demand_lowerBnd=1, demand_upperBnd=9):
    
    DEPOT_END = 300
    SERVICE_TIME = 10
    TW_WIDTH = 30
    
    def get_distance(vec1, vec2):
        return np.sum((vec1-vec2)**2)**0.5

    def random_tw(dist_to_depot,service_time,depot_end,tw_width):
        _tmp_0 = math.ceil(dist_to_depot)
        # _tmp_1 = int((_tmp_0+depot_end)/2)
        _tmp_1 = 200
        start = np.random.randint(_tmp_0, _tmp_1)
        end = start + tw_width
        if end < dist_to_depot or end + service_time + dist_to_depot > depot_end:
            start = 0
            end = depot_end
        return start,end
    
    data = random_cvrp(n_nodes, n_clusters, demand_lowerBnd, demand_upperBnd)
    tw = [[0,DEPOT_END,0]]
    for i in range(1, len(data)):
        dist_to_depot = get_distance(data[0], data[i])
        start,end = random_tw(dist_to_depot,SERVICE_TIME,DEPOT_END,TW_WIDTH)
        tw.append([start,end,SERVICE_TIME])
    tw = np.array(tw)
    result = np.concatenate((data, tw), axis=1)
    return result

def random_vrptw(n_nodes,
                 n_clusters=None,
                 capacity=None,
                 time_gap=None,
                 time_limit=None,
                 start0_prob=None,
                 service_time=None):
    """
    随机产生cvrptw问题。坐标值为整数且任意两点不重叠。出发点(depot)位于全图的中心。
    需求(demand)，和时间窗(time window)等信息均为整数。
    n_nodes: 客户(nodes)的数量。
    n_clusters: =0:客户点(nodes)会被放置在随机位置，完全没有偏好。
                =n: 客户(nodes)会汇聚成n个群落。群落内客户(nodes)数量随机。
                =None: 随机选择上述两种情况之一(概率各为50%)。
    """
    def sigmoid(x):
        return 1.0/(1.0+np.exp(-x))
    assert n_nodes>=0
    
    data = []
    
    # 参数初始化
    if n_nodes>200:
        side_limit = int(max(140, n_nodes*0.5))
    else:
        side_limit = int(max(10, n_nodes*0.7))
    if capacity is None:
        capacity = int(np.random.randint(-2, 16)*100)
        if capacity<200: capacity=200
        if capacity>1000: capacity=1000
    demand_lowerBnd = 5
    demand_upperBnd = 36
    demand_func = lambda x: np.random.normal(20, 11, 1)[0]
    if time_gap is None:
        if np.random.random()>0.5:
            time_gap = (np.random.randint(10)+1)*10
        else:
            time_gap = np.random.randint(100)*10
    if time_limit is None:
        time_limit_min = int(max(side_limit*3+90+time_gap, n_nodes))
        time_limit_max = min(time_limit_min*5, max(n_nodes*10, 2000))
        time_limit = np.random.randint(min(time_limit_min, time_limit_max), max(time_limit_min, time_limit_max))
    if start0_prob is None:
        start0_prob = max(0, sigmoid(np.random.random()*20-15)-0.2)
    if service_time is None:
        if time_limit>(max((n_nodes/100-1), 1)**0.5*1000):
            service_time = int(10+np.random.randint(2)*80)
        else:
            service_time = 10
    if n_clusters is None:
        if np.random.random()>0.5:
            n_clusters = 0
        else:
            n_clusters = min(n_nodes, np.random.randint(5, int(max(6,n_nodes/5))))
    
    if n_clusters > 0:
        assert n_clusters<=n_nodes
        while len(data) < n_clusters:
            coord = [np.random.randint(0,side_limit), np.random.randint(0,side_limit)]
            flag = False
            for d in data:
                if coord[0] == d[0] and coord[1] == d[1]:
                    flag = True
                    break
            if flag: continue
            data.append([coord[0], coord[1],])
        
        while len(data) < n_nodes:
            rnd = np.array([np.random.randint(-8,9), np.random.randint(-8,9)])
            coord = data[np.random.randint(len(data))][:2]+rnd
            if coord[0]<0 or coord[1]<0 or coord[0]>=side_limit or coord[1]>=side_limit: continue
            flag = False
            for d in data:
                if coord[0] == d[0] and coord[1] == d[1]:
                    flag = True
                    break
            if flag: continue
            data.append([coord[0], coord[1],])
    else:
        while len(data) < n_nodes:
            coord = [np.random.randint(0,side_limit), np.random.randint(0,side_limit)]
            flag = False
            for d in data:
                if coord[0] == d[0] and coord[1] == d[1]:
                    flag = True
                    break
            if flag: continue
            data.append([coord[0], coord[1],])
    for i in range(len(data)):
        demand = int(demand_func(None))
        if demand<1 or demand>45: demand = np.random.randint(demand_lowerBnd, demand_upperBnd+1)
        if i==0:
            data[i]=[int(side_limit/2), int(side_limit/2), 0, 0, time_limit, 0]
        else:
            travel_time = int(get_distance(np.array(data[i][:2]), np.array(data[0][:2])))+1
            rand_limit = time_limit-service_time-time_gap-travel_time*2
            if np.random.random()<start0_prob:
                end_time = rand_limit+travel_time+time_gap
                data[i]+=[demand, 0, end_time, service_time]
            else:
                if np.random.random()>0.5:
                    start_time = np.random.randint(rand_limit/2)
                else:
                    start_time = np.random.randint(rand_limit)
                start_time+=travel_time
                end_time = start_time+time_gap
                data[i]+=[demand, start_time, end_time, service_time]
    data = np.array(data)
    return capacity, data