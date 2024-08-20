import numpy
import pickle
from tqdm import tqdm

for i in tqdm(range(1000000)):
    with open(f'./data/data_one_by_one/{i}/adj_table_list.pkl', 'rb') as file:
        all_adj_table = pickle.load(file)
    for j in range(len(all_adj_table)):
        all_adj_table[j] = all_adj_table[j] + 1
    with open(f'./data/data_one_by_one/{i}/adj_table_list.pkl', 'wb') as file:
        pickle.dump(all_adj_table, file)
print('Done')