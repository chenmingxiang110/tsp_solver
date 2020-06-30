import os
import copy
import numpy as np

def get_distMat(data):
    coords = data[:,:2]
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    return distance_matrix

def get_neighbours(distance_matrix):
    n_vertices = distance_matrix.shape[0]
    neighbours = []
    for i in range(n_vertices):
        index_dist = [(j, distance_matrix[i][j]) for j in range(n_vertices)]
        sorted_index_dist = sorted(index_dist, key=lambda x: x[1])
        neighbours.append([x[0] for x in sorted_index_dist])
    return neighbours

def get_solDist(distance_matrix, routes):
    total_distance = 0
    for route in routes:
        r = [0]+route+[0]
        total_distance += np.sum([distance_matrix[r[i],r[i+1]] for i in range(len(r)-1)])
    return total_distance

def env(data, vehicle_capcity, distance_matrix, last_routes, nodes):
    
    def get_route_distance(distance_matrix, route):
        r = [0]+route+[0]
        result = np.sum([distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)])
        return result
    
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
    
    current_routes = []
    
    # delete
    for r in last_routes:
        new_r = [x for x in r if x not in nodes]
        if len(new_r)>0:
            current_routes.append(new_r)
    
    # insert
    for c in nodes:
        probable_place = []
        for ir,r in enumerate(current_routes):
            # assert capacity
            if (np.sum([data[x][2] for x in r])+data[c][2])>vehicle_capcity: continue
            valids = getValid_legacy(r,c)
            for v in valids:
                probable_place.append((ir,v[0],v[1]))
        if len(probable_place)==0:
            adding_position = (-1,-1,1)
        else:
            adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
        current_routes = route_add(current_routes, c, adding_position)
    return current_routes

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
            if start and len(line.strip())>0:
                r = line.strip().split(":")[1]
                r = [int(x) for x in r.split() if len(x)>0]
                routes.append(r)
            else:
                if "Solution" == line.strip():
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
            ans_name = ans.lower()
            ans_name = ans_name.split(".")[0]
            ans_name = ans_name.split("-")[0]
            if name == ans_name:
                pa = ans
                break
        if len(pa)>0:
            result[folderPath_q+que] = folderPath_a+pa
    return result

def get_inbalance_distMat(distMat, percentage=0.2):
    assert percentage>=0
    a = (np.random.random(distMat.shape) * percentage + 1 )
    b = np.random.randint(2, size=distMat.shape)*2-1
    return a**b * distMat

def find_ruins(n_ruins, anchor, coefficient, neighbour, routes, ruined):
    """
    与 SISR 规则类似。区别在于:
        1. 路线中被删去部分的比例是输入项，而非通过 l_max, c_bar 等参数随机。
        2. 如果下一个 neighbour 所在的线路被删除过，该点依然可以被删除，且一样会寻找线路前后的点删除。
    以上修改使得 coefficient 和删去点的空间聚集度具有更严格的正相关性
    """
    def find_t(routes, c):
        for i in range(len(routes)):
            if c in routes[i]: return i
        return None
    
    def remove_nodes(route, c, num):
        nodes = []
        i = route.index(c)
        nodes.append(route[i])
        j = i+1
        k = i-1
        num = min(num, len(route))
        while len(nodes)<num:
            if j>=len(route):
                j=0
            if k<0:
                k = len(route)-1
            nodes.append(route[j])
            if len(nodes)<num:
                nodes.append(route[k])
            j+=1
            k-=1
        return nodes

    coefficient = max(coefficient, 1e-8)
    ruin_nodes = []
    # ruined_t_indices = set([])
    for c in neighbour[anchor]:
        if c not in ruin_nodes and c!=0:
            t = find_t(routes, c)
            # if t in ruined_t_indices: continue
            num = int(np.ceil(len(routes[t]) * coefficient))
            newly_removed = remove_nodes(routes[t], c, num)
            for node in newly_removed:
                if node not in ruin_nodes and node not in ruined and len(ruin_nodes)<n_ruins: ruin_nodes.append(node)
            # ruined_t_indices.add(t)
            if len(ruin_nodes)>=n_ruins:
                break
    ruin_nodes = ruin_nodes[:n_ruins]
#     print(ruined, ruin_nodes)
    return ruin_nodes

def find_degree(a, b, c):
    ab = b-a
    bc = c-b
    cosine = max(min(ab.dot(bc)/(np.linalg.norm(ab)*np.linalg.norm(bc)+1e-16),0.999),-0.999)
    try:
        result = np.arccos(cosine)/np.pi
    except RuntimeError:
        print(cosine)
    return result

def get_embeddings(cap, data, distMat, routes, _e=None):
    """
    return: 含 depot (第零个 data)
    """
    
    d_max = np.max(distMat)
    if _e is None:
        dim0 = data[:,3]/data[0,4] # normalized start_time
        dim1 = data[:,4]/data[0,4] # normalized end_time
        dim2 = data[:,5]/data[0,4] # normalized service_time
        dim3 = (data[:,4]-data[:,3])/(data[0,4]-data[0,3]) # 时间窗跨度除以 depot's 时间窗跨度
        dim4 = np.zeros(len(data)) # total demand normalized by cap
        dim5 = np.zeros(len(data)) # cumulative demand normalized by cap
        dim6 = np.zeros(len(data)) # current demand normalized by cap
        dim7 = np.zeros(len(data)) # total distance normalized by max total distance
        dim8 = np.zeros(len(data)) # cumulative distance normalized by max total distance
        dim9 = np.zeros(len(data)) # forward shift normalized by max total distance
        dim10 = np.zeros(len(data)) # 消去后距离变化量 normalized by 2* max single distance
        dim11 = np.zeros(len(data)) # 路径上这一点的角度

        d_mean = np.mean(distMat)
        d_std = np.std(distMat)
        d_mean_n = (np.mean(distMat, 0)+np.mean(distMat, 1))/2.0
        sym_mat = (distMat+np.transpose(distMat))/2.0

        # 总平均距离以内点数比例
        dim12 = (np.sum((sym_mat<d_mean).astype(int), 1)-1)/(len(data)-1)
        # 总平均距离+1标准差以内点数比例
        dim13 = (np.sum((sym_mat<(d_mean+d_std)).astype(int), 1)-1)/(len(data)-1)
        # 总平均距离-1标准差以内点数比例
        dim14 = (np.sum((sym_mat<(d_mean-d_std)).astype(int), 1)-1)/(len(data)-1)
        # ave distance to depot (from and to) normalized by max single distance
        dim15 = sym_mat[0,:]/d_max
        
    else:
        dim0 = _e[:,0]
        dim1 = _e[:,1]
        dim2 = _e[:,2]
        dim3 = _e[:,3]
        dim4 = np.zeros(len(data))
        dim5 = np.zeros(len(data))
        dim6 = np.zeros(len(data))
        dim7 = np.zeros(len(data))
        dim8 = np.zeros(len(data))
        dim9 = np.zeros(len(data))
        dim10 = np.zeros(len(data))
        dim11 = np.zeros(len(data))
        dim12 = _e[:,12]
        dim13 = _e[:,13]
        dim14 = _e[:,14]
        dim15 = _e[:,15]
    
    max_dist = 0.0
    for r in routes:
        complete_r = [0]+list(r)+[0]
        cum_demd = 0.0
        cum_dist = 0.0
        for i in range(1, len(complete_r)-1):
            demd = data[complete_r[i], 2]
            dist = distMat[complete_r[i-1], complete_r[i]]
            cum_demd += demd
            cum_dist += dist
            dim6[complete_r[i]] = demd/float(cap) # normalized current demand
            dim5[complete_r[i]] = cum_demd/float(cap) # normalized cumulative demand
            dim8[complete_r[i]] = cum_dist
            
            d01 = distMat[complete_r[i-1],complete_r[i]]
            d12 = distMat[complete_r[i],complete_r[i+1]]
            d02 = distMat[complete_r[i-1],complete_r[i+1]]
            dim10[complete_r[i]] = (d01+d12-d02)/(2*d_max)
            
            coord_a = data[complete_r[i-1], :2]
            coord_b = data[complete_r[i], :2]
            coord_c = data[complete_r[i+1], :2]
            dim11[complete_r[i]] = find_degree(coord_a, coord_b, coord_c)
            
            cum_dist += distMat[complete_r[i], complete_r[i+1]]

        for i in range(1, len(complete_r)-1):
            _id = complete_r[i]
            dim4[_id] = cum_demd/float(cap) # normalized total demand
            dim7[_id] = cum_dist

        if cum_dist>max_dist: max_dist=cum_dist
        
        # 计算 forward shift
        D = np.zeros(len(complete_r)-1)
        for i in range(1, len(complete_r)-1):
            start_t = max(data[complete_r[i],3], D[i-1]+distMat[complete_r[i-1], complete_r[i]])
            D[i] = start_t+data[complete_r[i],5]
        for i in range(1, len(complete_r)-1):
            s_plus = []
            for j in range(i+1, len(complete_r)):
                t_sum = np.sum([distMat[complete_r[k],complete_r[k+1]] for k in range(i,j)])
                l_j = data[complete_r[j],4]
                s_plus.append(l_j-(D[i]+t_sum))
            dim9[complete_r[i]] = np.min(s_plus)
    dim7 /= max_dist
    dim8 /= max_dist
    dim9 /= max_dist
    return np.transpose(np.array([dim0, dim1, dim2, dim3,
                                  dim4, dim5, dim6, dim7,
                                  dim8, dim9, dim10, dim11,
                                  dim12, dim13, dim14, dim15,]))