import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# test function whether a trajectory skip a node
def test_trajectory_error(pred, weighted_adj, trajectory):
    i = 0
    while pred[i] != 0 and pred[i+1] != 0 and i < len(pred)-1:
        if weighted_adj[pred[i]-1,pred[i+1]-1] == 0: # currently the codebook is 1-indexing
            return True
        i += 1
    j = 0 
    while trajectory[j] != 0 and trajectory[j+1] != 0 and j < len(trajectory)-1:
        j += 1
    if pred[i] != trajectory[j]:
        return True
    else:
        return False

# Calculate the length of a trajectory
def calculate_length(trajectory, weighted_adj):
    length = 0
    for i in range(len(trajectory)-1):
        if trajectory[i+1] != 0:
            length += weighted_adj[trajectory[i]-1,trajectory[i+1]-1] # currently the codebook is 1-indexing
        else:
            break
    return length

def calculate_length_(trajectory, weighted_adj):
    length = 0
    for i in range(len(trajectory)-1):
        length += weighted_adj[trajectory[i],trajectory[i+1]]
    return length

def transfer_graph(weighted_adj, grid_size):
    G = nx.DiGraph()
    for i in range(grid_size*grid_size):
        G.add_node(i)
    for i in range(grid_size*grid_size):
        for j in range(grid_size*grid_size):
            if weighted_adj[i,j] != 0:
                G.add_edge(i,j,weight=weighted_adj[i,j])
    return G


def print_trajectory(predict_trajectory, test_trajectory):
    for i in range(len(predict_trajectory)):
        result_txt = f"weighted graph {i+1:<3}, "
        for j in range(len(predict_trajectory[i])):
            result_txt_predict = result_txt+f"pred {j+1:<3}: {predict_trajectory[i][j]}"
            print(result_txt_predict)
            
            result_txt_test = result_txt+f"true {j+1:<3}: {test_trajectory[i][j]}"
            print(result_txt_test)
        

# remove the special token, input one traj as numpy, return a list without special token
def remove_special_token(trajectory):
    new_trajectory = []
    for i in range(len(trajectory)):
        if trajectory[i] == 0:
            break
        new_trajectory.append(trajectory[i]-1) # 0-indexing
    return new_trajectory

# transfer grid into position
def transfer_position(grid_size):
    pos = {}
    for i in range(grid_size):
        for j in range(grid_size):
            pos[i*grid_size+j] = (i,j)
    return pos

# Function to calculate the bounds of the graph
def calculate_bounds(G, pos):
    x_values = [pos[node][0] for node in G]
    y_values = [pos[node][1] for node in G]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_min, x_max = x_min - 0.1 * abs(x_max - x_min), x_max + 0.1 * abs(x_max - x_min) 
    y_min, y_max = y_min - 0.1 * abs(y_max - y_min), y_max + 0.1 * abs(y_max - y_min) 

    return x_min, x_max, y_min, y_max

# Plot the trajectories on the graph
def plot_trajs(ax, G, pos, weighted_adj, traj, traj_ = None, ground_truth=False):

    if ax is None:
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='gray')
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
    else:
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='gray',ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', ax=ax)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f'{weighted_adj[i, j]:.2f}' for i, j in G.edges()}, label_pos=0.7, ax=ax)
        

    # Plot trajectories
    x = [pos[node][0]+np.random.normal(0,0.01) for node in traj]
    y = [pos[node][1]+np.random.normal(0,0.01) for node in traj]
    traj_len = calculate_length_(traj, weighted_adj)
    ax.plot(x, y, marker='o', linewidth = 2.0, alpha=0.5, markersize=10, label=f'Generated Trajectory {traj_len}', color='blue')
    ax.plot(x[0], y[0], marker='*', markersize=20, color='black')

    if traj_ is not None:
        x = [pos[node][0]+np.random.normal(0,0.01) for node in traj_]
        y = [pos[node][1]+np.random.normal(0,0.01) for node in traj_]
        traj_len = calculate_length_(traj_, weighted_adj)
        ax.plot(x, y, marker='o',  linewidth = 2.0, alpha=0.5, markersize=10, label=f'Ground Truth Trajectory {traj_len}', color='red')

    ax.legend()


# Plot the training and validation losses
def plot_losses(train_losses, val_losses, eval_interval):
    x = range(0, len(train_losses)*eval_interval, eval_interval)
    plt.plot(x, train_losses, label='train')
    plt.plot(x, val_losses, label='val')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()