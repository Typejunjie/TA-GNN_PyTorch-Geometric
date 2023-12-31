import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import pickle

class session_graph(InMemoryDataset):
    def __init__(self, root, processed_file_name, phrase, transform=None, pre_transform=None, pre_filter=None):
        self.phrase = phrase
        self.processed_file_name = processed_file_name
        super(session_graph, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.processed_file_name]

    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        pre_data = pickle.load(open(self.phrase, 'rb'))

        # Process
        cache = pd.DataFrame()
        for seq, label in tqdm(zip(pre_data[0], pre_data[1]), colour='green', desc='Processing graph data', leave=False):
            cache = pd.DataFrame(seq, columns=['item_id'])
            encoded = LabelEncoder().fit_transform(cache.item_id.values)
            cache['encoded'] = encoded
            node_features = cache.loc[:, ['item_id', 'encoded']].drop_duplicates('item_id').item_id.values.reshape(-1, 1)
            x = torch.tensor(node_features, dtype=torch.long)

            source = cache.encoded.values[: -1]
            target = cache.encoded.values[1:]
            edge_index = torch.tensor([source, target], dtype=torch.long)

            y = torch.tensor([label], dtype=torch.long)

            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])