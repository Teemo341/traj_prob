from matplotlib.pyplot import grid
import torch
import time
import pickle
import numpy as np

from dataset import get_capacity, get_adjacency, get_length, get_weighted_adjacency, generate_trajectory_list, get_OD_list, make_codebook


# Function to read the encoded data from a file and save it as a list of integers
def read_encoded_trajectory(filename, block_size = 10):
    all_encoded_trajectories = []
    all_condition = []
    all_special_mask = []
    with open(filename, 'r') as file:
        for line in file:
            trajectory = line.strip().split()
            trajectory = [int(code) for code in trajectory]
            condition = [trajectory[0],trajectory[-1]]
            special_mask = np.ones(block_size)
            if len(trajectory) > block_size:
                raise ValueError(f'Trajectory length {len(trajectory)} is greater than block size {block_size}')
            elif len(trajectory) < block_size:
                special_mask[len(trajectory)+1:] = 0
                trajectory += [0] * (block_size - len(trajectory))
            all_encoded_trajectories.append(trajectory)
            all_condition.append(condition)
            all_special_mask.append(special_mask)

    # all_encoded_trajectories: NxT, T can be different for each trajectory
    # all_condition: Nx2
    # all_special_mask: NxT
    return all_encoded_trajectories, all_condition, all_special_mask


def refine_trajectory(trajectory,block_size):
    all_encoded_trajectories = []
    all_condition = []
    all_special_mask = []
    for i in range(len(trajectory)):
        traj = trajectory[i]
        traj = [int(code) for code in traj]
        condition = [traj[0],traj[-1]]
        special_mask = np.ones(block_size)
        if len(traj) > block_size:
            raise ValueError(f'Trajectory length {len(traj)} is greater than block size {block_size}')
        elif len(traj) < block_size:
            special_mask[len(traj)+1:] = 0
            traj += [0] * (block_size - len(traj))
        all_encoded_trajectories.append(traj)
        all_condition.append(condition)
        all_special_mask.append(special_mask)
    
    return all_encoded_trajectories, all_condition, all_special_mask
    

def read_adjacency(filename):
    return torch.tensor(np.load(filename))

def read_data_list(simulation_num, block_size):
    print('Loading the encoded trajectory...')
    time_ = time.time()
    trajectory_list = []
    weighted_adj_list = []
    adj_table_list = []
    condition_list = []
    special_mask_list = []
    for i in range(simulation_num):
        all_encoded_trajectories, all_condition, all_special_mask = read_encoded_trajectory(f'./data/trajectory_{i}.txt', block_size=block_size)
        all_weighted_adj = read_adjacency(f'./data/weighted_adjacency_{i}.npy')
        all_adj_table = read_adjacency(f'./data/adj_table_{i}.npy')
        trajectory_list.append(all_encoded_trajectories)
        weighted_adj_list.append(all_weighted_adj)
        adj_table_list.append(all_adj_table)
        condition_list.append(all_condition)
        special_mask_list.append(all_special_mask)

    assert len(trajectory_list) == simulation_num and len(weighted_adj_list) == simulation_num and len(condition_list) == simulation_num
    print('Encoded trajectory loaded, time:', time.time()-time_)
    # trajectory_list: [simulation_num, trajectory_num, block_size]
    # weighted_adj_list: [simulation_num, grid_size*grid_size, grid_size*grid_size]
    # condition_list: [simulation_num, trajectory_num, 2]
    # special_mask_list: [simulation_num, trajectory_num, block_size]
    return trajectory_list, weighted_adj_list, adj_table_list, condition_list, special_mask_list


def read_data_pkl(simulation_num, block_size, root = './data'):
    time_ = time.time()
    trajectory_list = []
    weighted_adj_list = []
    adj_table_list = []
    condition_list = []
    special_mask_list = []
    
    with open(f'{root}/trajectory_list.pkl', 'rb') as file:
        all_trajectory_list = pickle.load(file)
    with open(f'{root}/weighted_adj_list.pkl', 'rb') as file:
        all_weighted_adj_list = pickle.load(file)
    with open(f'{root}/adj_table_list.pkl', 'rb') as file:
        all_adj_table_list = pickle.load(file)

    print('data num:',simulation_num)
    for i in range(simulation_num):
        all_encoded_trajectories, all_condition, all_special_mask = refine_trajectory(all_trajectory_list[i], block_size=block_size)
        trajectory_list.append(all_encoded_trajectories)
        condition_list.append(all_condition)
        special_mask_list.append(all_special_mask)
    weighted_adj_list = all_weighted_adj_list[:simulation_num]
    adj_table_list = all_adj_table_list[:simulation_num]

    assert len(trajectory_list) == simulation_num and len(weighted_adj_list) == simulation_num and len(condition_list) == simulation_num
    print('Encoded trajectory loaded, time:', time.time()-time_)
    # trajectory_list: [simulation_num, trajectory_num, block_size]
    # weighted_adj_list: [simulation_num, grid_size*grid_size, grid_size*grid_size]
    # condition_list: [simulation_num, trajectory_num, 2]
    # special_mask_list: [simulation_num, trajectory_num, block_size]
    return trajectory_list, weighted_adj_list, adj_table_list, condition_list, special_mask_list


def generate_new_data(codebook, grid_size=10,trajectory_num = 1, block_size = 10, weight_quantization_scale = None, max_connection = 4):
    OD_list = get_OD_list(grid_size=grid_size, trajectory_num=trajectory_num)
    grid_capacity = get_capacity(grid_size=grid_size)
    adj = get_adjacency(grid_size=grid_size)
    road_len = get_length(grid_size*grid_size)
    weighted_adj, adj_table = get_weighted_adjacency(adj=adj,grid_capacity=grid_capacity,length=road_len, normalization=True, quantization_scale=weight_quantization_scale, max_connection = max_connection)
    trajectory_list = generate_trajectory_list(OD_list, weighted_adj, grid_size)

    all_encoded_trajectories = []
    all_condition = []
    all_special_mask = []
    for i in range(trajectory_num):
        trajectory = trajectory_list[i]
        encoded_trajectory = [int(codebook[(x, y)]) for x, y in trajectory]
        condition = [encoded_trajectory[0],encoded_trajectory[-1]]
        special_mask = np.ones(block_size)
        if len(encoded_trajectory) > block_size:
            raise ValueError(f'Trajectory length {len(encoded_trajectory)} is greater than block size {block_size}')
        elif len(encoded_trajectory) < block_size:
            special_mask[len(encoded_trajectory)+1:] = 0
            encoded_trajectory += [0] * (block_size - len(encoded_trajectory))
        all_encoded_trajectories.append(encoded_trajectory)
        all_condition.append(condition)
        all_special_mask.append(special_mask)

    # all_encoded_trajectories: [trajectory_num, block_size]
    # all_weighted_adj: [grid_size*grid_size, grid_size*grid_size]
    # all_condition: [trajectory_num, 2]
    # all_special_mask: [trajectory_num, block_size]
    return all_encoded_trajectories, weighted_adj, adj_table, all_condition, all_special_mask

def generate_data_list(simulation_num, grid_size ,total_trajectories, block_size, codebook = None, weight_quantization_scale = None, max_connection = 4):
    trajectory_list = []
    weighted_adj_list = []
    adj_table_list = []
    condition_list = []
    special_mask_list = []
    # print('Generating new encoded trajectory...')
    time_ = time.time()
    for i in range(simulation_num):
        all_encoded_trajectories, weighted_adj, adj_table, all_condition, all_special_mask = generate_new_data(codebook, grid_size=grid_size, trajectory_num=total_trajectories, block_size=block_size, weight_quantization_scale = weight_quantization_scale, max_connection = max_connection)
        trajectory_list.append(all_encoded_trajectories)
        weighted_adj_list.append(weighted_adj)
        adj_table_list.append(adj_table)
        condition_list.append(all_condition)
        special_mask_list.append(all_special_mask)
    assert len(trajectory_list) == simulation_num and len(weighted_adj_list) == simulation_num and len(condition_list) == simulation_num
    # print('New encoded trajectory generated, time:', time.time()-time_)

    # trajectory_list: [simulation_num, trajectory_num, block_size]
    # weighted_adj_list: [simulation_num, grid_size*grid_size, grid_size*grid_size]
    # condition_list: [simulation_num, trajectory_num, 2]
    return trajectory_list, weighted_adj_list, adj_table_list, condition_list, special_mask_list
    
# dataloaders
class data_loader():
    def __init__(self, use_given_data = True, simulation_num = 400, test_simulation_num = 100, grid_size = 10, block_size = 20):
        self.simulation_num = simulation_num
        self.test_simulation_num = test_simulation_num
        self.grid_size = grid_size
        self.block_size = block_size
        if use_given_data:
            self.trajectory_list, self.weighted_adj_list, self.adj_table_list, self.condition_list, self.special_mask_list = read_data_pkl(simulation_num, block_size, root = './data')
            self.index = 0
            self.test_trajectory_list, self.test_weighted_adj_list, self.test_adj_table_list, self.test_condition_list, self.test_special_mask_list = read_data_pkl(test_simulation_num, block_size, root = './data_test')
            self.index_test = 0
        self.codebook = make_codebook(grid_size)
        
    def load_train_batch(self, batch_size = 32, total_trajectories = 1):
        idx = self.index * batch_size % self.simulation_num
        self.index += 1
        # Get the trajectory and the weighted adjacency matrix
        trajectory = self.trajectory_list[idx:idx+batch_size] # B, N, L
        weighted_adj = self.weighted_adj_list[idx:idx+batch_size] # B, V, V
        adj_table = self.adj_table_list[idx:idx+batch_size] # B, V, E, 2
        condition = self.condition_list[idx:idx+batch_size] # B, N, 2
        special_mask = self.special_mask_list[idx:idx+batch_size] # B, N, L
        return torch.tensor(trajectory), torch.tensor(np.array(weighted_adj)).float(), torch.tensor(np.array(adj_table)).float(), torch.tensor(condition), torch.tensor(np.array(special_mask))
    
    def load_test_batch(self, batch_size = 32, total_trajectories = 1):
        idx = self.index_test * batch_size % self.test_simulation_num
        self.index_test += 1
        # Get the trajectory and the weighted adjacency matrix
        trajectory = self.test_trajectory_list[idx:idx+batch_size]
        weighted_adj = self.test_weighted_adj_list[idx:idx+batch_size]
        adj_table = self.test_adj_table_list[idx:idx+batch_size]
        condition = self.test_condition_list[idx:idx+batch_size]
        special_mask = self.test_special_mask_list[idx:idx+batch_size]
        return torch.tensor(trajectory), torch.tensor(np.array(weighted_adj)).float(), torch.tensor(np.array(adj_table)).float(), torch.tensor(condition), torch.tensor(np.array(special_mask))
    
    def generate_batch(self, batch_size = 32, total_trajectories = 1):
        trajectory, weighted_adj, adj_table, condition, special_mask = generate_data_list(batch_size, self.grid_size, total_trajectories, self.block_size, self.codebook, weight_quantization_scale = None, max_connection = 4)
        return torch.tensor(trajectory), torch.tensor(np.array(weighted_adj)).float(), torch.tensor(np.array(adj_table)).float(), torch.tensor(condition), torch.tensor(np.array(special_mask))

if __name__ == '__main__':
    # test the dataloader
    loader = data_loader(use_given_data = True, simulation_num = 400, test_simulation_num = 100, grid_size = 10, block_size = 20)
    trajectory, weighted_adj, adj_table, condition, special_mask = loader.load_train_batch(batch_size = 32, total_trajectories = 1)
    print(trajectory.size(), weighted_adj.size(), adj_table.size(), condition.size(), special_mask.size())
    trajectory, weighted_adj, adj_table, condition, special_mask = loader.load_test_batch(batch_size = 32, total_trajectories = 1)
    print(trajectory.size(), weighted_adj.size(), adj_table.size(), condition.size(), special_mask.size())
    trajectory, weighted_adj, adj_table, condition, special_mask = loader.generate_batch(batch_size = 32, total_trajectories = 1)
    print(trajectory.size(), weighted_adj.size(), adj_table.size(), condition.size(), special_mask.size())