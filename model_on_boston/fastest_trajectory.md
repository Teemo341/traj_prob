# Fastest Trajectory in once

generate a fastest trajectory in one forward propagation

the key idea is, the fastest trajectory should be guessed by the weighted graph and OD at once, rather than by auto regression

```
# B means how many weighted graph
# N means each graph how many random OD
# V means the number of edge
# L means the max length of fasted path

Inputs: condition: torch.Tensor, 
 weighted_adj: torch.Tensor, 
 y: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None, 
 special_mask: Optional[torch.Tensor] = None

        # condition: (B, N, 2) # the start point encoding and end point encoding of B*N trajectories
        # weighted_adj: (B, V, V) # the weighted adjacency matrix of B graphs
        # y: (B, N, L) # the true shortest path of B*N trajectories
        # adjmask: (B, V, V) # if use adj to control the path, not used now
        # special mask: (B, N, L) # the weight for the loss function between 0 token and predicted 0 token, None means 1, 0 means don't care about the 0 token. Note that the first 0 token is important so we give it full weight

Output: logits: torch.tensor,
 loss: torch.tensor

 # logits: (B, N, L, V) # the predictive probability of each road
 # loss: () # the CE loss between softmax(logits,dim =1) and y
```

## testing hyperparameter: {2.1}

| test_num                          | total_trajectories                         |  |
| --------------------------------- | ------------------------------------------ | - |
| how many different weighted graph | on each graph, <br />how many different OD |  |

## simulation hyperparameter

| simulation_num                                                      | total_trajectories                         | capacity_scale = 10                      | weight_quantization_scale = 3                             |
| ------------------------------------------------------------------- | ------------------------------------------ | ---------------------------------------- | --------------------------------------------------------- |
| simulate how many different graph<br />0 means simulate on training | one each graph,<br />how mant different OD | generate random weight<br />~U[1, scale] | if quantize weight, 1~ ?<br />None means no quantization |

## trainng hyperparameter: {2.3}

| max_iters = 200000 | learning_rate = 1e-3 | eval_iters = 1000 | save_iters = 10000                       | batch_size = 128                  | path_num = 1                   | block_size = 25                           | special_mask_value = 0.1                                                                      |
| ------------------ | -------------------- | ----------------- | ---------------------------------------- | --------------------------------- | ------------------------------ | ----------------------------------------- | --------------------------------------------------------------------------------------------- |
|                    |                      |                   | besides save model<br />also do lr decay | how many different weighted graph | on each graph<br />how many OD | The max length of<br /> all shortest path | the loss weight for "0 token"<br />the first 0 token has full loss<br />the other 0 token not |

| n_embd = 64 | n_head = 16 | n_layer = 16 | dropout = 0.1 | load_dir_id = 130000 | continue_train = True | vocab_size = grid_size*grid_size+1      | optimizer                                | lr_sched      |
| ----------- | ----------- | ------------ | ------------- | -------------------- | --------------------- | --------------------------------------- | ---------------------------------------- | ------------- |
|             |             |              |               | the saved model id   |                       | 0-indexing, 0 means the end placeholder | SGD(, , momentum=0.9, weight_decay=1e-4) | StepLR(,,0.1) |

# todo 20240709

将邻接矩阵改成邻接表✔️

连续值最短路径✔️

考虑evaluvation 怎么设计✔️

考虑多条最短路怎么设计loss

明确完全相同设定下，时间对比

## logger

```
max_iters = 20000
learning_rate = 1e-1
eval_iters = 100
save_iters = 10000
```
![logger](./image/fastest_trajectory/1.png)

```
max_iters = 20000
learning_rate = 1e-2
eval_iters = int(max_iters/100)
save_iters = 10000
```
![logger](./image/fastest_trajectory/2.png)

```
max_iters = 20000
learning_rate = 5e-2
eval_iters = int(max_iters/100)
save_iters = 10000
```
![logger](./image/fastest_trajectory/3.png)

```
max_iters = 50000
learning_rate = 1e-1
eval_iters = int(max_iters/100)
save_iters = 10000
```
![logger](./image/fastest_trajectory/4.png)
