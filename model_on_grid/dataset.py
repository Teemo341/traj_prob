import networkx as nx
import os
import pickle
import numpy as np
import copy

import argparse
from tqdm import tqdm


def get_capacity(capacity_scale = 10, grid_size = 10):
    grid_capacity = np.random.uniform(1,capacity_scale,grid_size*grid_size)
    return grid_capacity

def get_adjacency(grid_size=10):
    adj = np.zeros([grid_size*grid_size,grid_size*grid_size],dtype= int)
    for i in range(grid_size):
        for j in range(grid_size):
            if i>0:
                adj[(i-1)*grid_size+j,i*grid_size+j] = 1
            if i<grid_size-1:
                adj[(i+1)*grid_size+j,i*grid_size+j] = 1
            if j>0:
                adj[i*grid_size+j-1,i*grid_size+j] = 1
            if j<grid_size-1:
                adj[i*grid_size+j+1,i*grid_size+j] = 1
    return adj

def get_length(road_num):
    return np.ones(road_num,dtype= int)

# get weighted adjacency matrix and adjacency table
def get_weighted_adjacency(adj, grid_capacity, length, normalization = True, quantization_scale = None, max_connection = 4):
    weighted_adj_matrix = copy.deepcopy(adj)
    adj_table = np.zeros([weighted_adj_matrix.shape[0],max_connection, 2]) # [node, connection, [target_node, weight]]
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i,j] == 1:
                weighted_adj_matrix[i,j] = (grid_capacity[j]+1)*length[j]
                adj_table[i,np.sum(adj_table[i,:,0]!=0)] = [j,weighted_adj_matrix[i,j]] # [target_node, weight], add to the first empty slot
    if normalization:
        weighted_adj = weighted_adj_matrix/np.max(weighted_adj_matrix)
        adj_table[:,:,1] = adj_table[:,:,1]/np.max(adj_table[:,:,1])
    if quantization_scale:
        weighted_adj = np.ceil(weighted_adj*quantization_scale)
        adj_table[:,:,1] = np.ceil(adj_table[:,:,1]*quantization_scale)
        
    return weighted_adj, adj_table



# transfer node, wrighted_adj to graph
def transfer_graph(weighted_adj, grid_size):
    G = nx.DiGraph()
    for i in range(grid_size*grid_size):
        G.add_node(i)
    for i in range(grid_size*grid_size):
        for j in range(grid_size*grid_size):
            if weighted_adj[i,j] != 0:
                G.add_edge(i,j,weight=weighted_adj[i,j])
    return G


# get shortest traj
def generate_trajectory_list(OD_list, weighted_adj, grid_size):
    trajectory_list = []
    G = transfer_graph(weighted_adj, grid_size)
    for i in range(len(OD_list)):
        trajectory = nx.shortest_path(G, (OD_list[i][0][0]-1)*grid_size+OD_list[i][0][1]-1, (OD_list[i][1][0]-1)*grid_size+OD_list[i][1][1]-1, weight='weight')
        for j in range(len(trajectory)):
            trajectory[j] = (trajectory[j]//grid_size+1,trajectory[j]%grid_size+1)
        trajectory_list.append(trajectory)
    return trajectory_list
        

def get_OD_list(grid_size=10, trajectory_num = 10):
    # define OD [trajectory_num,2,2] (N*[[x_start,y_start],[x_end,y_end]])
    # 1-indexing
    OD_list = np.random.randint(1,grid_size+1,[trajectory_num,2,2])
    return OD_list


def make_codebook(grid_size):
    codebook = {}
    max_value = grid_size  # max(y) = 10 in this case

    # Populate the codebook with grid cells
    for x in range(1, grid_size + 1):
        for y in range(1, grid_size + 1):
            code = (x - 1) * max_value + y
            codebook[(x, y)] = code
    codebook[(0,0)] = grid_size*grid_size+1
    codebook[(grid_size+1,grid_size+1)] = grid_size*grid_size+2
    return codebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--simulation_num', type=int, default=2000000)
    parser.add_argument('--grid_size', type=int, default=10)
    parser.add_argument('--total_trajectories', type=int, default=1)
    parser.add_argument('--capacity_scale', type=int, default=10)
    parser.add_argument('--weight_quantization_scale', type=int, default=None)
    parser.add_argument('--max_connection', type=int, default=4)
    args = parser.parse_args()

    data_dir = args.data_dir
    simulation_num = args.simulation_num
    grid_size = args.grid_size
    total_trajectories = args.total_trajectories
    capacity_scale = args.capacity_scale
    weight_quantization_scale = args.weight_quantization_scale
    max_connection = args.max_connection


    # Generate the codebook
    codebook = make_codebook(grid_size)

    # Save the codebook to a file
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(data_dir+'/codebook.txt', 'w') as file:
        for key, value in codebook.items():
            file.write(f'{key}: {value}\n')


    #simulation for simulation_num times
    all_trajectory_list = []
    all_weighted_adj_list = []
    all_adj_table_list = []

    print('Start generating data...')
    for t in tqdm(range(simulation_num), desc=f'Generating data'):
        # print(f'simulation {t}')
        # Generate and save 10 trajectories
        all_encoded_trajectories = []

        OD_list = get_OD_list(grid_size=grid_size, trajectory_num= total_trajectories)
        # grid_capacity = get_capacity(current_trajectory_list=OD_list[:,0,:],grid_size=grid_size)
        grid_capacity = get_capacity(capacity_scale,grid_size)
        adj = get_adjacency(grid_size=grid_size)
        road_len = get_length(grid_size*grid_size)
        weighted_adj, adj_table = get_weighted_adjacency(adj=adj,grid_capacity=grid_capacity,length=road_len, normalization=True, quantization_scale=weight_quantization_scale, max_connection = max_connection)
        trajectory_list = generate_trajectory_list(OD_list, weighted_adj, grid_size)

        for i in range(total_trajectories):
            trajectory = trajectory_list[i]
            encoded_trajectory = [codebook[(x, y)] for x, y in trajectory]
            # # Append '0' at the end of each trajectory
            # encoded_trajectory.append('0')
            all_encoded_trajectories.append(encoded_trajectory)

        # Save all trajectories to a single file
        all_trajectory_list.append(all_encoded_trajectories)
        all_weighted_adj_list.append(weighted_adj)
        all_adj_table_list.append(adj_table)
        # print(f'simulation {t}')

    with open(data_dir+'/trajectory_list.pkl', 'wb') as file:
        pickle.dump(all_trajectory_list, file)
    with open(data_dir+'/weighted_adj_list.pkl', 'wb') as file:
        pickle.dump(all_weighted_adj_list, file)
    with open(data_dir+'/adj_table_list.pkl', 'wb') as file:
        pickle.dump(all_adj_table_list, file)
    print(f'saved successfully!')