import pickle
import h5py
import pandas as pd


# Function to preprocess the map of Boston
def preprocess_data(origin_data):
    E = len(origin_data['edge_id'])

    # Record the coordinates of the nodesand calculate the bounds of weights
    pos = {}
    edges = []
    for i in range(E):
        u, v = origin_data['from_node_id'][i]+1, origin_data['to_node_id'][i]+1
        w = origin_data['length'][i] / origin_data['speed_limit'][i]
        edges.append((u, v, w, 10*w))
        if u not in pos:
            pos[u] = (origin_data['from_lon'][i], origin_data['from_lat'][i])
        if v not in pos:
            pos[v] = (origin_data['to_lon'][i], origin_data['to_lat'][i])

    return edges, pos

# Function to read the data of cities
def read_city(city, path='./'):
    if city != 'boston' and city != 'paris':
        raise ValueError('Invalid city name!')
    origin_data = pd.read_csv(path + city + '_data.csv').to_dict(orient='list')
    edges, pos = preprocess_data(origin_data)

    return edges, pos


# Save positions of the Boston graph
def save_positions(pos, path='./'):
    with open(path + 'positions.pkl', 'wb') as file:
        pickle.dump(pos, file)
    print('Positions saved!')


# Save weights of the Boston graph
def save_weights(weights, path='./'):
    with h5py.File(path + 'weights.h5', 'w') as file:
        file.create_dataset('weights', data=weights)
    print('Weights saved!')


# Save the generated trajectories
def save_trajs(trajs, path='./'):
    with h5py.File(path + 'trajs.h5', 'w') as file:
        file.create_dataset('trajs', data=trajs)
    print('Trajectories saved!')


# Function to read the data
def read_data(path='./'):
    with h5py.File(path + 'trajs.h5', 'r') as file:
        trajs = file['trajs'][:]
    return trajs


# Function to read positions from files
def read_positions(path='./'):
    with open(path + 'positions.pkl', 'rb') as file:
        pos = pickle.load(file)

    return pos
        

# Function to read the graph data
def read_graphs(path='./'):
    with h5py.File(path + 'weights.h5', 'r') as file:
        weights = file['weights'][:]

    return weights


# Function to read the edges only
def read_edges(path='./'):
    with h5py.File(path + 'weights.h5', 'r') as file:
        edges = file['weights'][:, :2]

    return edges


# Function to save the hyperparameters
def save_hyperparams(hyperparams, path='./'):
    with open(path + 'hyperparams.pkl', 'wb') as file:
        pickle.dump(hyperparams, file)


# Function to save train loss and validation loss
def save_loss(train_loss, val_loss, eval_interval, path='./'):
    with open(path + 'loss.pkl', 'wb') as file:
        pickle.dump({'train_loss': train_loss, 'val_loss': val_loss, 'eval_interval': eval_interval}, file)

    print('Loss saved!')


# Function to read train loss and validation loss
def read_loss(path='./'):
    with open(path + 'loss.pkl', 'rb') as file:
        loss = pickle.load(file)
    return loss['train_loss'], loss['val_loss'], loss['eval_interval']


# Function to read the hyperparameters
def read_hyperparams(path='./'):
    with open(path + 'hyperparams.pkl', 'rb') as file:
        hyperparams = pickle.load(file)
    return hyperparams
