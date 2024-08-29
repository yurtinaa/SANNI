# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np
from numpy import random

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from matrixprofile.algorithms import mpdist_vector

torch.set_default_dtype(torch.double)


class ClassifierLoader(Dataset):
    def __init__(self, X, y, batch_size=32):
        # print(type(X))
        self.X = X
        self.y = y.long()

    #  print(self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class ClassifierDataset:
    arr_data: list
    snippet_list_arr: list
    batch_size: int = 32
    shuffle: bool = True
    seed: int = 2366
    size_subsequent: int = 100
    all_data: str = 'none'
    test_size: float = 0.25
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        test_y = []
        arr_index = []
        arr_snippet = []
        X = []
        y = []
        for i in np.arange(len(self.arr_data)):
            if self.all_data is 'none':
                X.append(self.arr_data[i, :])
            else:
                X.append(self.arr_data[i])
            y_buffer = []
            for dim_idx, dim_data in enumerate(self.snippet_list_arr):
                for index_snippet, snippet in enumerate(dim_data):
                    if i in snippet['neighbors']:
                        y_buffer.append(index_snippet)
                        break
            y.append(y_buffer)

        X = torch.Tensor(X)
        y = torch.Tensor(y)
        X = torch.Tensor.transpose(X, 2, 1)

        self.dataset = {}
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            shuffle=self.shuffle,
                                                            test_size=self.test_size,
                                                            random_state=self.seed)
        del X
        del y
        self.dataset["test"] = [X_test, y_test]
        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                          y_train,
                                                          shuffle=self.shuffle,
                                                          test_size=self.test_size,
                                                          random_state=self.seed)
        self.dataset["val"] = [X_val, y_val]
        self.dataset["train"] = [X_train, y_train]

    def augmentation(self, cel_frac=0.15):
        start = self.dataset["train"][1].shape
        snippets_frac = []
        out = []
        new_data_set = self.dataset["train"].copy()
        count = 0
        all_new_neighbors = {}
        all_new_label = {}
        max_count = 0
        dims = self.dataset["train"][1].shape[1]
        device = self.dataset["train"][0].device
        for idx_dim in range(dims):
            all_new_neighbors[idx_dim] = []
            all_new_label[idx_dim] = []
            size_train = self.dataset["train"][1].shape[0]
            data_label = self.dataset["train"][1][:, idx_dim].cpu()
            snippet_counts = np.unique(data_label,
                                       return_counts=True)[1]
            count_snippet = len(snippet_counts)
            frac = 1 / count_snippet
            snippet_frac = snippet_counts / size_train - frac
            # print(snippet_counts / size_train)
            snippets_frac.append(snippet_frac)
            augs_snippet = np.where(snippet_frac <= -cel_frac)
            augs_snippet[0]
            new_neighbors = []
            new_label = []
            for aug in augs_snippet[0]:
                count_aug_snippet = round((-snippet_frac[aug] - cel_frac) * size_train)
                # print(snippet_frac[aug], count_aug_snippet)
                count += count_aug_snippet
                # break
                indx_neighbors = np.where(data_label == aug)[0][:count_aug_snippet]
                dist_arr = []
                snippet = np.array(self.snippet_list_arr[idx_dim]['snippet'][aug])[:]
                wtf = self.dataset["train"][0].clone().cpu()
                label_wtf = self.dataset["train"][1].clone().cpu()
                for neiborhs in wtf[label_wtf[:, idx_dim] == aug, idx_dim, :]:
                    dist = mpdist_vector(ts=neiborhs.cpu().numpy(), ts_b=snippet[:-1], w=len(snippet) - 1)
                    dist_arr.append(np.min(dist))
                dist_arr = np.stack([dist_arr, range(len(dist_arr))]).T
                dist_arr = dist_arr[dist_arr[:, 0].argsort()]
                dist_arr = dist_arr[::-1]
                while (len(dist_arr) < count_aug_snippet):
                    dist_arr = np.vstack([dist_arr, dist_arr])

                out.append(f'min_mp_dist_{aug}_{idx_dim}:{dist_arr[count_aug_snippet, 0]}')
                out.append(f'max_mp_dist_{aug}_{idx_dim}:{dist_arr[0, 0]}')
                indx = int(dist_arr[count_aug_snippet, 0])

                neighbor = self.dataset["train"][0][indx, idx_dim].clone().cpu().numpy()
                out.append(f'min_e_{aug}_{idx_dim}:{np.linalg.norm(snippet[:-1] - neighbor)}')

                indx = int(dist_arr[0, 0])

                neighbor = self.dataset["train"][0][indx, idx_dim].clone().cpu().numpy()
                out.append(f'max_e_{aug}_{idx_dim}:{np.linalg.norm(snippet[:-1] - neighbor)}')
                for indx, (dist, indx_neighbor) in enumerate(dist_arr[:count_aug_snippet]):
                    indx_neighbor = int(indx_neighbor)
                    neighbor = self.dataset["train"][0][indx_neighbor].clone().cpu()
                    # print(np.array(dataset.snippet_list_arr[idx_dim]['snippet'][aug])[:-1],neighbor[idx_dim].numpy())
                    dist = np.linalg.norm(
                        np.array(
                            self.snippet_list_arr[idx_dim]['snippet'][aug]
                        )[:-1] - neighbor[idx_dim].numpy()
                    )
                    # print(dist)
                    dist = dist / neighbor.shape[1] * 2

                    neighbor_label = self.dataset["train"][1][indx_neighbor].clone().cpu()
                    e_arr = torch.tensor(random.uniform(-dist,
                                                        dist,
                                                        size=neighbor.shape[1] // 2)).cpu()
                    neighbor[idx_dim, random.randint(0,
                                                     neighbor.shape[1] - 1,
                                                     size=neighbor.shape[1] // 2)] += e_arr
                    new_neighbors.append(neighbor[None, idx_dim, :])
                    new_label.append(neighbor_label[idx_dim])

            if len(new_neighbors) > 0:

                all_new_neighbors[idx_dim] = torch.cat(new_neighbors, 0)
                # print('----------', all_new_neighbors[idx_dim].shape)
                all_new_label[idx_dim] = torch.tensor(new_label)
                if len(new_neighbors) > max_count:
                    max_count = len(new_neighbors)
        #  print(max_count)
        if max_count > 0:
            for indx_dim in range(dims):
                if len(all_new_neighbors[indx_dim]) < max_count:
                    count = max_count - len(all_new_neighbors[indx_dim])
                    indx = random.randint(0,
                                          self.dataset["train"][1].shape[0],
                                          size=count)

                    neiborhs = self.dataset["train"][0][indx, indx_dim]
                    neiborhs_label = self.dataset["train"][1][indx, indx_dim]
                    if len(all_new_neighbors[indx_dim]) != 0:
                        # print('----------')
                        # print(all_new_neighbors[indx_dim].shape, neiborhs.shape)
                        all_new_neighbors[indx_dim] = torch.cat([all_new_neighbors[indx_dim].to(device),
                                                                 neiborhs.to(device)], 0)

                        all_new_label[indx_dim] = torch.cat(
                            [all_new_label[indx_dim].to(device), neiborhs_label.to(device)])
                    else:
                        all_new_neighbors[indx_dim] = neiborhs
                        all_new_label[indx_dim] = neiborhs_label

                    # print(count, all_new_neighbors[indx_dim].shape)
                all_new_neighbors[indx_dim] = all_new_neighbors[indx_dim][:, None, :].to(device)
                all_new_label[indx_dim] = all_new_label[indx_dim][:, None].to(device)

            device = self.dataset["train"][0].device
            # print(np.stack(all_new_neighbors).shape)
            # for indx_dim in range(dims):
            #     print(all_new_neighbors[indx_dim].shape)
            #     print(all_new_label[indx_dim].shape)
            data = list(all_new_neighbors.values())
            all_new_neighbors = torch.cat(data, 1).to(device)
            data = list(all_new_label.values())
            all_new_label = torch.cat(data, 1).to(device)
            # print(self.dataset["train"][0].shape)
            self.dataset["train"][0] = torch.cat((self.dataset["train"][0], all_new_neighbors))
            self.dataset["train"][1] = torch.cat((self.dataset["train"][1], all_new_label))
        # print(self.dataset["train"][1].shape)

        return self.dataset["train"][1].shape[0] - start[0], \
               round((self.dataset["train"][1].shape[0] - start[0]) / start[0] * 100), \
               np.array(snippets_frac), out

    def get_loader(self, type_dataset):
        # print("батчи:", self.batch_size)

        return DataLoader(ClassifierLoader(self.dataset[type_dataset][0],
                                           self.dataset[type_dataset][1]),
                          batch_size=self.batch_size)
