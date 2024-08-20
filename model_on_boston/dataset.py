import networkx as nx
import os
import pickle
import numpy as np
import pandas as pd
import shutil
import gc

import argparse
from tqdm import tqdm


# Function to preprocess the map of Boston
def preprocess_data(origin_data):
    E = len(origin_data['edge_id'])

    # Record the coordinates of the nodesand calculate the bounds of weights
    #! 0-indexing
    pos = {}
    edges = []
    for i in range(E):
        u, v = origin_data['from_node_id'][i], origin_data['to_node_id'][i]
        w = origin_data['length'][i] / origin_data['speed_limit'][i]
        edges.append((u, v, w, 10*w))
        if u not in pos:
            pos[u] = (origin_data['from_lon'][i], origin_data['from_lat'][i])
        if v not in pos:
            pos[v] = (origin_data['to_lon'][i], origin_data['to_lat'][i])

    return edges, pos

# Function to read the data of cities
def read_city(city, path='./data/'):
    if city != 'boston' and city != 'paris':
        raise ValueError('Invalid city name!')
    origin_data = pd.read_csv(path + city + '_data.csv').to_dict(orient='list')
    edges, pos = preprocess_data(origin_data)

    return edges, pos


def get_capacity(capacity_scale = 10, edge_num = 100):
    edge_capacity = np.random.uniform(1,capacity_scale,edge_num)
    return edge_capacity

# get weighted adjacency matrix and adjacency table
def get_weighted_adjacency(edges, pos, capacity, normalization = True, quantization_scale = None, max_connection = 4):
    weighted_adj_matrix = np.zeros([len(pos),len(pos)])
    for i in range(len(edges)):
        weighted_adj_matrix[edges[i][0],edges[i][1]] = capacity[i]
    max_connection_ = np.max([np.sum(weighted_adj_matrix[i]!=0) for i in range(len(pos))])
    if max_connection_ > max_connection:
        raise ValueError('max_connection is too small, need {max_connection_} but only {max_connection} is given')
    
    adj_table = np.zeros([weighted_adj_matrix.shape[0],max_connection, 2]) # [node, connection, [target_node, weight]]
    for i in range(weighted_adj_matrix.shape[0]):
        for j in range(weighted_adj_matrix.shape[1]):
            if weighted_adj_matrix[i,j] != 0:
                adj_table[i,np.sum(adj_table[i,:,0]!=0)] = [j,weighted_adj_matrix[i,j]] # [target_node, weight], add to the first empty slot
    if normalization:
        weighted_adj = weighted_adj_matrix/np.max(weighted_adj_matrix)
        adj_table[:,:,1] = adj_table[:,:,1]/np.max(adj_table[:,:,1])
    if quantization_scale:
        weighted_adj = np.ceil(weighted_adj*quantization_scale)
        adj_table[:,:,1] = np.ceil(adj_table[:,:,1]*quantization_scale)
        
    return weighted_adj, adj_table



# transfer node, wrighted_adj to graph
def transfer_graph(adj_table):
    G = nx.DiGraph()
    for i in range(len(adj_table)):
        G.add_node(i)
    for i in range(len(adj_table)):
        for j in range(len(adj_table[i])):
            if adj_table[i,j,1] != 0:
                G.add_edge(i,adj_table[i,j,0],weight=adj_table[i,j,1])
    return G


# get shortest traj
def generate_trajectory_list(adj_table, node_num, trajectory_num = 1):
    trajectory_list = []
    OD_list = np.random.randint(1,node_num+1,[trajectory_num,2])
    G = transfer_graph(adj_table)
    for i in range(len(OD_list)):
        while nx.has_path(G, (OD_list[i][0]-1), (OD_list[i][1]-1)) == False or OD_list[i][0] == OD_list[i][1]:
            OD_list[i] = np.random.randint(1,node_num+1,2)
        trajectory = nx.shortest_path(G, (OD_list[i][0]-1), (OD_list[i][1]-1), weight='weight')
        for j in range(len(trajectory)):
            trajectory[j] = trajectory[j]+1
        trajectory_list.append(trajectory)
    return trajectory_list, OD_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--simulation_num', type=int, default=2000000)
    parser.add_argument('--total_trajectories', type=int, default=1)
    parser.add_argument('--capacity_scale', type=int, default=10)
    parser.add_argument('--weight_quantization_scale', type=int, default=None)
    parser.add_argument('--max_connection', type=int, default=4)
    args = parser.parse_args()

    data_dir = args.data_dir
    simulation_num = args.simulation_num
    total_trajectories = args.total_trajectories
    capacity_scale = args.capacity_scale
    weight_quantization_scale = args.weight_quantization_scale
    max_connection = args.max_connection

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    #simulation for simulation_num times

    edges, pos = read_city('boston', path='./data_city/')
    shutil.copyfile('./data_city/boston_data.csv', data_dir+'/boston_data.csv')
    node_num = len(pos)
    edge_num = len(edges)

    # print('Start generating data...')
    # for t in tqdm(range(simulation_num), desc=f'Generating data'):
    #     # print(f'simulation {t}')
    #     # Generate and save 10 trajectories
    #     all_encoded_trajectories = []

    #     edge_capacity = get_capacity(capacity_scale,edge_num)
    #     weighted_adj, adj_table = get_weighted_adjacency(edges, pos, edge_capacity, normalization = True, quantization_scale = weight_quantization_scale, max_connection = max_connection)
    #     trajectory_list, OD_list = generate_trajectory_list(adj_table, node_num, total_trajectories)

    #     # Save all trajectories to a single file
    #     with open(data_dir+'/trajectory_list_single.pkl', 'ab') as file:
    #         pickle.dump(trajectory_list, file) #! 1-indexing
    #     # with open(data_dir+'/weighted_adj_list.pkl', 'ab') as file:
    #     #     pickle.dump(weighted_adj, file) #! 0-indexing
    #     with open(data_dir+'/adj_table_list_single.pkl', 'ab') as file:
    #         pickle.dump(adj_table, file) #! 0-indexing

    # print(f'one by one saved successfully!')
    # del weighted_adj, adj_table, trajectory_list, file
    # gc.collect()

    # # transfer spreead data to one file

    # print('Start merging data...')
    # print('Merging trajectory_list...')
    # trajectory_list = []
    # with open(data_dir+'/trajectory_list_single.pkl', 'rb') as file:
    #     while True:
    #         try:
    #             trajectory_list.append(pickle.load(file))
    #         except EOFError:
    #             break
    # with open(data_dir+'/trajectory_list.pkl', 'wb') as file:
    #         pickle.dump(trajectory_list, file)
    # del trajectory_list, file
    # gc.collect()
    # os.remove(data_dir+'/trajectory_list_single.pkl')
    # print('Success')

    # # print('Merging weighted_adj_list...')
    # # weighted_adj_list = []
    # # with open(data_dir+'/weighted_adj_list.pkl', 'rb') as file:
    # #     while True:
    # #         try:
    # #             weighted_adj_list.append(pickle.load(file))
    # #         except EOFError:
    # #             break
    # # with open(data_dir+'/weighted_adj_list.pkl', 'wb') as file:
    # #         pickle.dump(weighted_adj_list, file)
    # # del weighted_adj_list
    # # print('Success')

    print('Merging adj_table_list...')
    adj_table_list = []
    with open(data_dir+'/adj_table_list_single.pkl', 'rb') as file:
        for i in range(simulation_num):
            try:
                adj_table_list.append(pickle.load(file))
            except EOFError:
                break
    print('data_loaded')
    with open(data_dir+'/adj_table_list.pkl', 'wb') as file:
            pickle.dump(adj_table_list, file)
    del adj_table_list
    gc.collect()
    # os.remove(data_dir+'/adj_table_list_single.pkl')
    print('Success')

    print(f'Merged successfully!')

