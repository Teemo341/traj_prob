import pickle
import os
import argparse
from tqdm import tqdm
import torch
import time

from model import no_diffusion_model
from data_loader import data_loader
from utils import test_trajectory_error, calculate_length


# ------------------------------------------------------------------------------------------------------------------------------------------
# Hyperparameters

parser = argparse.ArgumentParser()

# about dataset
parser.add_argument('--simulation_num', type=int, default=1000000)
parser.add_argument('--test_simulation_num', type=int, default=1000)
parser.add_argument('--grid_size', type=int, default=10)
parser.add_argument('--total_trajectories', type=int, default=1)
parser.add_argument('--capacity_scale', type=int, default=10)
parser.add_argument('--weight_quantization_scale', type=int, default=None)
parser.add_argument('--max_connection', type=int, default=4)
parser.add_argument('--train_data_dir', type=str, default='./data/')
parser.add_argument('--test_data_dir', type=str, default='./data_test/')

# about dataloader
parser.add_argument('--use_given_data', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--test_size', type=int, default=1000)
parser.add_argument('--path_num', type=int, default=1)
parser.add_argument('--block_size', type=int, default=25)
parser.add_argument('--special_mask_value', type=float, default=0.01)

# about model
parser.add_argument('--n_embd', type=int, default=64)
parser.add_argument('--n_head', type=int, default=16)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--use_adj_table', type=bool, default=True)

# about training
parser.add_argument('--max_iters', type=int, default=20000)
parser.add_argument('--learning_rate', type=float, default=1e-1)
parser.add_argument('--lr_drop_rate', type=float, default=0.5)
parser.add_argument('--eval_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda')

# about record and resume training
parser.add_argument('--load_dir', type=str, default='./checkpoints/')
parser.add_argument('--load_dir_id', type=int, default=None)

args = parser.parse_args()
if not os.path.exists(args.load_dir):
    os.makedirs(args.load_dir)
with open(f'{args.load_dir}args.pkl', 'wb') as f:
    pickle.dump(args, f)

# about dataset
simulation_num = args.simulation_num
test_simulation_num = args.test_simulation_num
use_given_data = args.use_given_data
if simulation_num == 0:
    use_given_data = False
grid_size = args.grid_size
vocab_size = grid_size*grid_size+1 # 0-indexing, 0 means the end placeholder

# about dataloader
batch_size = args.batch_size # how many independent djkastra graph will we process in parallel?
test_size = args.test_size # how many test data will we use?
path_num = args.path_num # on each weighted graph, how many shortest path will we consider?
block_size = args.block_size # The max length of all shortest path
special_mask_value = args.special_mask_value # The value of special mask

# about model
n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout

# about training
max_iters = args.max_iters
learning_rate = args.learning_rate
lr_drop_rate = args.lr_drop_rate
eval_iters = int(max_iters/args.eval_freq)
# eval_iters = 1
save_iters = max_iters//args.save_freq
max_iters = save_iters*args.save_freq

if torch.cuda.is_available() and args.device == 'cuda':
    device = 'cuda:3'
else:
    device = 'cpu'

use_adj_table = args.use_adj_table

# about record and resume training
load_dir_id = args.load_dir_id
load_dir = args.load_dir
# load_dir_id = 10000


# ------------------------------------------------------------------------------------------------------------------------------------------
# Load model
start_time = time.time()

model= no_diffusion_model(vocab_size, n_embd, n_embd, n_layer, n_head, block_size, dropout, weight_quantization_scale = None, use_adj_table=use_adj_table, use_ne=True, use_ge=True, use_agent_mask=False, norm_position='prenorm', device=device)
model = model.to(device)

if load_dir_id:
    # model.load_state_dict(torch.load(f'{load_dir}model_{load_dir_id}.pth'))
    # print('Model loaded from', f'{load_dir}model_{load_dir_id}.pth')
    model = torch.load(f'{load_dir}complete_model_{load_dir_id}.pth')
    print('Model loaded from', f'{load_dir}complete_model_{load_dir_id}.pth')
    learning_rate *= lr_drop_rate**int(max_iters//load_dir_id)

print('Model loaded in', time.time()-start_time, 'seconds')


# ------------------------------------------------------------------------------------------------------------------------------------------
# Load data
start_time = time.time()

loader = data_loader(use_given_data, simulation_num, test_simulation_num, grid_size, block_size)

print('Data loaded in', time.time()-start_time, 'seconds')


# ------------------------------------------------------------------------------------------------------------------------------------------
# set optimizer and lr scheduler

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-15)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-15)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

# lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=save_iters, gamma=lr_drop_rate)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, eta_min=0, last_epoch=-1 if not load_dir_id else load_dir_id)

# ------------------------------------------------------------------------------------------------------------------------------------------
# train

# make loggers
logger_train_loss = []
logger_test_loss = []
logger_test_error_rate = []
logger_test_real_true_rate = []
logger_test_fake_true_rate = []
logger_test_length_rate_avg = []
logger_test_leng_10_rate = []

# training epochs
start_time = time.time()
start = 0
if load_dir_id:
    start = load_dir_id
print('start training from iteration', start)
for i in range (start+1, max_iters+1):
    model.train()
    trajectory, weighted_adj, adj_table, condition, special_mask = loader.load_train_batch(batch_size)
    trajectory = trajectory.to(device)
    if use_adj_table:
        adj = adj_table.to(device)
    else:
        adj = weighted_adj.to(device)
    condition = condition.to(device)
    special_mask = special_mask.to(device)
    special_mask = (special_mask+special_mask_value).clamp(0,1).float()

    logits, loss = model(condition, adj, trajectory, None, special_mask)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.zero_grad()
    loss.backward()
    logger_train_loss.append(loss.item())
    optimizer.step()
    lr_sched.step()

    if i % eval_iters == 0:
        model.eval()
        result_text = f'Iteration {i:<10}|  Loss: {loss.item():<10.8f}  |'
        trajectory, weighted_adj, adj_table, condition, special_mask = loader.load_test_batch(test_size)
        trajectory = trajectory.to(device)
        if use_adj_table:
            adj = adj_table.to(device)
        else:
            adj = weighted_adj.to(device)
        condition = condition.to(device)
        special_mask = special_mask.to(device)
        special_mask_ = special_mask > 0 # record the bool mask for the special token
        special_mask = (special_mask+special_mask_value).clamp(0,1).float()

        logits, loss = model(condition, adj, trajectory, None, special_mask)
        acc = ((torch.argmax(logits, dim=-1) == trajectory).float()*special_mask_).sum()/special_mask_.sum()

        error_num = 0
        real_true_num = 0
        fake_true_num = 0
        len_rate_sum = 0
        len_10_num = 0
        for j in range(trajectory.size(0)):
            for k in range(trajectory.size(1)):
                pred = torch.argmax(logits, dim=-1)[j,k]
                if test_trajectory_error(pred, weighted_adj[j], trajectory[j,k]):
                    error_num += 1
                elif calculate_length(trajectory[j,k], weighted_adj[j]) != 0:
                    len_true = calculate_length(trajectory[j,k], weighted_adj[j])
                    len_pred = calculate_length(pred, weighted_adj[j])
                    len_rate = len_pred/len_true
                    len_rate_sum += len_rate
                    if len_true == len_pred:
                        if (torch.argmax(logits, dim=-1)[j,k] == trajectory[j,k]).any():
                            real_true_num += 1
                        else:
                            fake_true_num += 1
                    if len_rate <= 1.1:
                        len_10_num += 1

        error_rate = error_num/(trajectory.size(0)*trajectory.size(1))
        real_true_rate = real_true_num/(trajectory.size(0)*trajectory.size(1))
        fake_true_rate = fake_true_num/(trajectory.size(0)*trajectory.size(1))
        length_rate_avg = len_rate_sum/(trajectory.size(0)*trajectory.size(1)-error_num) if error_num != trajectory.size(0)*trajectory.size(1) else 0
        len_10_rate = len_10_num/(trajectory.size(0)*trajectory.size(1)-error_num) if error_num != trajectory.size(0)*trajectory.size(1) else 0
        
        result_text += f'  Test Loss: {loss.item():<10.8f}  |  Test Acc: {acc.item():<7.2%}  |  Error Rate: {error_rate:<7.2%}  |  Real True Rate: {real_true_rate:<7.2%}  |  Fake True Rate: {fake_true_rate:<7.2%}  |  Length Rate Avg: {length_rate_avg:<7.2%}  |  Length 110% rate: {len_10_rate:<7.2%}  |'
        print(result_text)
        logger_test_loss.append(loss.item())
        logger_test_error_rate.append(error_rate)
        logger_test_real_true_rate.append(real_true_rate)
        logger_test_fake_true_rate.append(fake_true_rate)
        logger_test_length_rate_avg.append(length_rate_avg)
        logger_test_leng_10_rate.append(len_10_rate)

        # save logger
        with open(f'{load_dir}logger_train_loss.pkl', 'wb') as f:
            pickle.dump(logger_train_loss, f)
        with open(f'{load_dir}logger_test_loss.pkl', 'wb') as f:
            pickle.dump(logger_test_loss, f)
        with open(f'{load_dir}logger_test_error_rate.pkl', 'wb') as f:
            pickle.dump(logger_test_error_rate, f)
        with open(f'{load_dir}logger_test_real_true_rate.pkl', 'wb') as f:
            pickle.dump(logger_test_real_true_rate, f)
        with open(f'{load_dir}logger_test_fake_true_rate.pkl', 'wb') as f:
            pickle.dump(logger_test_fake_true_rate, f)
        with open(f'{load_dir}logger_test_length_rate_avg.pkl', 'wb') as f:
            pickle.dump(logger_test_length_rate_avg, f)
        with open(f'{load_dir}logger_test_leng_10_rate.pkl', 'wb') as f:
            pickle.dump(logger_test_leng_10_rate, f)

        
    if i % save_iters == 0:
        # torch.save(model.state_dict(), f'./checkpoint/model_{i}.pth')
        torch.save(model, f'{load_dir}complete_model_{i}.pth')
        print(f'Model saved at iteration {i}')

print('Training finished in', (time.time()-start_time)//3600, 'hours', ((time.time()-start_time)%3600)//60, 'minutes', ((time.time()-start_time)%3600)%60, 'seconds')