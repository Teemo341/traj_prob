import torch
import time
import pickle
import numpy as np

from dataset import get_capacity, get_weighted_adjacency, generate_trajectory_list, read_city


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


def read_data_pkl(simulation_num, block_size, root = './data'):
    time_ = time.time()
    trajectory_list = []
    # weighted_adj_list = []
    adj_table_list = []
    condition_list = []
    special_mask_list = []
    
    with open(f'{root}/trajectory_list.pkl', 'rb') as file:
        all_trajectory_list = pickle.load(file)
    # with open(f'{root}/weighted_adj_list.pkl', 'rb') as file:
    #     all_weighted_adj_list = pickle.load(file)
    with open(f'{root}/adj_table_list.pkl', 'rb') as file:
        all_adj_table_list = pickle.load(file)

    print('data num:',simulation_num)
    for i in range(simulation_num):
        all_encoded_trajectories, all_condition, all_special_mask = refine_trajectory(all_trajectory_list[i], block_size=block_size)
        trajectory_list.append(all_encoded_trajectories)
        condition_list.append(all_condition)
        special_mask_list.append(all_special_mask)
    # weighted_adj_list = all_weighted_adj_list[:simulation_num]
    adj_table_list = all_adj_table_list[:simulation_num]

    if not len(trajectory_list) == simulation_num and len(adj_table_list) == simulation_num and len(condition_list) == simulation_num:
        raise ValueError(f"length not match, expact {simulation_num}, got {len(trajectory_list)} for trajectory_list, {len(adj_table_list)} for adj_table_list, {len(condition_list)} for condition_list")
    print('Encoded trajectory loaded, time:', time.time()-time_)
    # trajectory_list: [simulation_num, trajectory_num, block_size]
    # weighted_adj_list: [simulation_num, node_num, node_num]
    # condition_list: [simulation_num, trajectory_num, 2]
    # special_mask_list: [simulation_num, trajectory_num, block_size]
    return trajectory_list, adj_table_list, condition_list, special_mask_list


def generate_new_data(edges, pos, trajectory_num = 1, block_size = 10, weight_quantization_scale = None, max_connection = 4):
    node_num = len(pos)
    edge_num = len(edges)
    edge_capacity = get_capacity(edge_num = edge_num)
    weighted_adj, adj_table = get_weighted_adjacency(edges, pos, capacity=edge_capacity, normalization=True, quantization_scale=weight_quantization_scale, max_connection = max_connection)
    trajectory_list, OD_list = generate_trajectory_list(adj_table, node_num, trajectory_num)

    all_encoded_trajectories = []
    all_condition = []
    all_special_mask = []
    for i in range(trajectory_num):
        trajectory = trajectory_list[i]
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

    # all_encoded_trajectories: [trajectory_num, block_size]
    # all_weighted_adj: [node_num, node_num]
    # all_condition: [trajectory_num, 2]
    # all_special_mask: [trajectory_num, block_size]
    return all_encoded_trajectories, adj_table, all_condition, all_special_mask

def generate_data_list(simulation_num, edges, pos, total_trajectories, block_size, weight_quantization_scale = None, max_connection = 4):
    trajectory_list = []
    # weighted_adj_list = []
    adj_table_list = []
    condition_list = []
    special_mask_list = []
    # print('Generating new encoded trajectory...')
    for i in range(simulation_num):
        all_encoded_trajectories, adj_table, all_condition, all_special_mask = generate_new_data(edges, pos, trajectory_num = total_trajectories, block_size = block_size, weight_quantization_scale = weight_quantization_scale, max_connection = max_connection)
        trajectory_list.append(all_encoded_trajectories)
        # weighted_adj_list.append(weighted_adj)
        adj_table_list.append(adj_table)
        condition_list.append(all_condition)
        special_mask_list.append(all_special_mask)
    assert len(trajectory_list) == simulation_num and len(adj_table_list) == simulation_num and len(condition_list) == simulation_num
    # print('New encoded trajectory generated, time:', time.time()-time_)

    # trajectory_list: [simulation_num, trajectory_num, block_size]
    # weighted_adj_list: [simulation_num, node_num, node_num]
    # condition_list: [simulation_num, trajectory_num, 2]
    return trajectory_list, adj_table_list, condition_list, special_mask_list
    
# dataloaders
class data_loader():
    def __init__(self, city = 'boston', data_dir = './data/', test_data_dir = './data_test/', use_given_data = True, simulation_num = 400, test_simulation_num = 100, block_size = 20):
        self.edges, self.pos = read_city(city, path = data_dir)
        self.simulation_num = simulation_num
        self.test_simulation_num = test_simulation_num
        self.block_size = block_size
        if use_given_data:
            self.trajectory_list, self.adj_table_list, self.condition_list, self.special_mask_list = read_data_pkl(simulation_num, block_size, root = data_dir)
            self.index = 0
            self.test_trajectory_list, self.test_adj_table_list, self.test_condition_list, self.test_special_mask_list = read_data_pkl(test_simulation_num, block_size, root = test_data_dir)
            self.index_test = 0
        
    def load_train_batch(self, batch_size = 32, total_trajectories = 1):
        idx = self.index * batch_size % self.simulation_num
        self.index += 1
        # Get the trajectory and the weighted adjacency matrix
        trajectory = self.trajectory_list[idx:idx+batch_size] # B, N, L
        adj_table = self.adj_table_list[idx:idx+batch_size] # B, V, E, 2
        condition = self.condition_list[idx:idx+batch_size] # B, N, 2
        special_mask = self.special_mask_list[idx:idx+batch_size] # B, N, L
        return torch.tensor(np.array(trajectory)), torch.tensor(np.array(adj_table)).float(), torch.tensor(np.array(condition)), torch.tensor(np.array(special_mask))
    
    def load_test_batch(self, batch_size = 32, total_trajectories = 1):
        idx = self.index_test * batch_size % self.test_simulation_num
        self.index_test += 1
        # Get the trajectory and the weighted adjacency matrix
        trajectory = self.test_trajectory_list[idx:idx+batch_size]
        adj_table = self.test_adj_table_list[idx:idx+batch_size]
        condition = self.test_condition_list[idx:idx+batch_size]
        special_mask = self.test_special_mask_list[idx:idx+batch_size]
        return torch.tensor(np.array(trajectory)), torch.tensor(np.array(adj_table)).float(), torch.tensor(np.array(condition)), torch.tensor(np.array(special_mask))
    
    def generate_batch(self, batch_size = 32, total_trajectories = 1, max_connection = 4):
        trajectory, adj_table, condition, special_mask = generate_data_list(batch_size, self.edges, self.pos, total_trajectories, self.block_size, weight_quantization_scale = None, max_connection = max_connection)
        return torch.tensor(np.array(trajectory)), torch.tensor(np.array(adj_table)).float(), torch.tensor(np.array(condition)), torch.tensor(np.array(special_mask))
    

class data_loader_random_start(data_loader):
    def __init__(self, city = 'boston', data_dir = './data/', test_data_dir = './data_test/', use_given_data = True, simulation_num = 400, test_simulation_num = 100, block_size = 20):
        super(data_loader_random_start, self).__init__(city, data_dir, test_data_dir, use_given_data, simulation_num, test_simulation_num, block_size)
        if use_given_data:
            self.start_idx = np.zeros([self.simulation_num, len(self.trajectory_list[0])], dtype = np.int32)
            self.test_start_idx = np.zeros([self.test_simulation_num, len(self.test_trajectory_list[0])], dtype = np.int32)
            self.randomize_condition()

    def randomize_start_and_condition(self, observe_prob = 0.2):
        self.randomize_start()
        self.randomize_condition(observe_prob = observe_prob)

    def randomize_start(self):
        start_time = time.time()
        for i in range(self.simulation_num):
            for j in range(len(self.trajectory_list[i])):
                traj = np.array(self.trajectory_list[i][j])
                traj = np.roll(traj, -self.start_idx[i][j]) # return to the original traj
                traj_len = np.sum(traj != 0)
                self.start_idx[i][j] = np.random.randint(0, len(traj)-traj_len+1)
                self.trajectory_list[i][j] = np.roll(traj, self.start_idx[i][j])
                self.special_mask_list[i][j] = np.zeros(self.block_size)
                self.special_mask_list[i][j][self.start_idx[i][j]:self.start_idx[i][j]+traj_len] = 1.0
        for i in range(self.test_simulation_num):
            for j in range(len(self.test_trajectory_list[i])):
                traj = np.array(self.test_trajectory_list[i][j])
                traj = np.roll(traj, -self.test_start_idx[i][j])
                traj_len = np.sum(traj != 0)
                self.test_start_idx[i][j] = np.random.randint(0, len(traj)-traj_len+1)
                self.test_trajectory_list[i][j] = np.roll(traj, self.test_start_idx[i][j])
                self.test_special_mask_list[i][j] = np.zeros(self.block_size)
                self.test_special_mask_list[i][j][self.test_start_idx[i][j]:self.test_start_idx[i][j]+traj_len] = 1.0
        print(f'Start time and condition randomized in {(time.time()-start_time)/60}m')

    def randomize_condition(self, observe_prob = 0.2):
        start_time = time.time()
        for i in range(self.simulation_num):
            for j in range(len(self.trajectory_list[i])):
                condition_idx = np.random.choice(2, len(self.trajectory_list[i][j]), p=[1-observe_prob, observe_prob])
                condition_idx[self.start_idx[i][j]] = 1 # always show the first checkpoint
                self.condition_list[i][j] = self.trajectory_list[i][j]*condition_idx
        for i in range(self.test_simulation_num):
            for j in range(len(self.test_trajectory_list[i])):
                condition_idx = np.random.choice(2, len(self.test_trajectory_list[i][j]), p=[1-observe_prob, observe_prob])
                condition_idx[self.start_idx[i][j]] = 1
                self.test_condition_list[i][j] = self.test_trajectory_list[i][j]*condition_idx
    
    def generate_batch(self, batch_size=32, total_trajectories=1, max_connection=4):
        trajectory, adj_table, condition, special_mask = generate_data_list(batch_size, self.edges, self.pos, total_trajectories, self.block_size, weight_quantization_scale = None, max_connection = max_connection)
        for i in range(batch_size):
            for j in range(total_trajectories):
                traj = np.array(trajectory[i][j])
                traj_len = np.sum(traj != 0)
                # start = np.random.randint(0, len(traj)-traj_len+1)
                start = 0
                trajectory[i][j] = np.roll(traj, start)
                condition_idx = np.random.choice(2, len(traj), p=[0.8, 0.2])
                condition_idx[start] = 1
                condition[i][j] = trajectory[i][j]*condition_idx
                special_mask[i][j] = np.zeros(self.block_size)
                special_mask[i][j][start:start+traj_len] = 1.0
        return torch.tensor(np.array(trajectory)), torch.tensor(np.array(adj_table)).float(), torch.tensor(np.array(condition)), torch.tensor(np.array(special_mask))
    


if __name__ == '__main__':
    # test the dataloader
    loader = data_loader_random_start(city = 'boston', data_dir= './data/', test_data_dir= './data_test', use_given_data = True, simulation_num = 400, test_simulation_num = 100, block_size = 50)
    loader.randomize_start_and_condition()
    trajectory, adj_table, condition, special_mask = loader.load_train_batch(batch_size = 32, total_trajectories = 1)
    print(trajectory.size(), adj_table.size(), condition.size(), special_mask.size())
    print(trajectory[0][0], condition[0][0], special_mask[0][0])
    trajectory, adj_table, condition, special_mask = loader.load_test_batch(batch_size = 32, total_trajectories = 1)
    print(trajectory.size(), adj_table.size(), condition.size(), special_mask.size())
    trajectory, adj_table, condition, special_mask = loader.generate_batch(batch_size = 32, total_trajectories = 1)
    print(trajectory.size(), adj_table.size(), condition.size(), special_mask.size())
    loader.randomize_start_and_condition()
    trajectory, adj_table, condition, special_mask = loader.load_train_batch(batch_size = 32, total_trajectories = 1)
    print(trajectory[0][0], condition[0][0], special_mask[0][0])