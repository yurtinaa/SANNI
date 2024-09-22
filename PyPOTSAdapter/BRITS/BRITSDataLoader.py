"""
Dataset class for the imputation model BRITS.
"""
from dataclasses import dataclass, field
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Iterable, Dict, Tuple, Type

import torch
from pygrinder import fill_and_get_mask_torch
from pypots.data.utils import _parse_delta_torch
from sklearn.model_selection import train_test_split

from ...EnumConfig import EpochType
from ...Trainer.Loader.TorchLoader import TorchTensorLoader, BaseDataset
from torch.utils.data import DataLoader as TorchDataLoader


# from ...data.dataset import BaseDataset
# from ...data.utils import _parse_delta_torch

def brits_collate_fn(batch):
    data_items, labels = zip(*batch)

    data = [item.get_all_tensors() for item in data_items]
    idxs, forward_X, forward_mask, forward_delta, backward_X, backward_mask, backward_delta = map(torch.stack,
                                                                                                  zip(*data))

    batched_data_item = DataItem(
        idx=idxs,
        forward_X=forward_X,
        forward_mask=forward_mask,
        forward_delta=forward_delta,
        backward_X=backward_X,
        backward_mask=backward_mask,
        backward_delta=backward_delta
    )
    labels_tensor = torch.stack([label.clone().detach() for label in labels])

    # Возвращаем новый DataItem и батч меток
    return batched_data_item, labels_tensor


class DataItem:
    def __init__(self, idx, forward_X, forward_mask, forward_delta, backward_X, backward_mask, backward_delta):
        self.idx = torch.tensor(idx) if isinstance(idx, int) else idx.clone().detach()
        self.device = forward_X.device
        self.forward_X = forward_X
        self.forward_mask = forward_mask
        self.forward_delta = forward_delta
        self.backward_X = backward_X
        self.backward_mask = backward_mask
        self.backward_delta = backward_delta

    def get_all_tensors(self):
        return (
            self.idx,
            self.forward_X,
            self.forward_mask,
            self.forward_delta,
            self.backward_X,
            self.backward_mask,
            self.backward_delta
        )

    def __getitem__(self, key):
        if key == "forward":
            return {
                "X": self.forward_X,
                "missing_mask": self.forward_mask,
                "deltas": self.forward_delta
            }
        elif key == "backward":
            return {
                "X": self.backward_X,
                "missing_mask": self.backward_mask,
                "deltas": self.backward_delta
            }
        else:
            raise KeyError(f"Invalid key: {key}")

    def to(self, device):
        self.idx = self.idx.to(device)
        self.forward_X = self.forward_X.to(device)
        self.forward_mask = self.forward_mask.to(device)
        self.forward_delta = self.forward_delta.to(device)
        self.backward_X = self.backward_X.to(device)
        self.backward_mask = self.backward_mask.to(device)
        self.backward_delta = self.backward_delta.to(device)
        return self

    def __iter__(self):
        return iter((
            self.idx,
            self.forward_X,
            self.forward_mask,
            self.forward_delta,
            self.backward_X,
            self.backward_mask,
            self.backward_delta
        ))




@dataclass
class BRITSDataset(BaseDataset):

    def __post_init__(self):
        forward_X, forward_missing_mask = fill_and_get_mask_torch(self.X)
        forward_delta = _parse_delta_torch(forward_missing_mask)
        backward_X = torch.flip(forward_X, dims=[1])
        backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])
        backward_delta = _parse_delta_torch(backward_missing_mask)

        self.processed_data = {
            "forward": {
                "X": forward_X.to(torch.float32),
                "missing_mask": forward_missing_mask.to(torch.float32),
                "delta": forward_delta.to(torch.float32),
            },
            "backward": {
                "X": backward_X.to(torch.float32),
                "missing_mask": backward_missing_mask.to(torch.float32),
                "delta": backward_delta.to(torch.float32),
            },
        }
        # if not isinstance(self.data, str):
        #     # calculate all delta here.
        #     if self.return_X_ori:
        #         forward_missing_mask = self.missing_mask
        #         forward_X = self.X
        #     else:
        #         forward_X, forward_missing_mask = fill_and_get_mask_torch(self.X)

    def __getitem__(self, idx):
        X = DataItem(
            idx,
            self.processed_data["forward"]["X"][idx],
            self.processed_data["forward"]["missing_mask"][idx],
            self.processed_data["forward"]["delta"][idx],
            self.processed_data["backward"]["X"][idx],
            self.processed_data["backward"]["missing_mask"][idx],
            self.processed_data["backward"]["delta"][idx],
        )
        return X, self.y[idx].clone().detach()


@dataclass
class BRITSLoader(TorchTensorLoader):
    _set_dict: Dict[EpochType, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    dataset_factory: Type[BaseDataset] = BRITSDataset

    def __post_init__(self):
        # self.X = torch.tensor(self.X, dtype=torch.float32)
        # self.y = torch.tensor(self.y, dtype=torch.float32)

        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            shuffle=self.shuffle,
                                                            test_size=self.percent,
                                                            random_state=self.seed)
        # print(X_train.dtype, y_test.dtype)
        self._set_dict[EpochType.TRAIN] = X_train,y_train
        self._set_dict[EpochType.EVAL] = X_test, y_test

    # def length(self, epoch_type: EpochType) -> float:
    #     return self.__set_dict[epoch_type][0].shape[0] // self.batch_size

    def __iter__(self, epoch_type: EpochType):
        dataset = self.dataset_factory(*self._set_dict[epoch_type])
        return TorchDataLoader(dataset,

                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               collate_fn=brits_collate_fn)

    def __call__(self, epoch_type: EpochType):
        # print(self._set_dict.keys())
        dataset = self.dataset_factory(*self._set_dict[epoch_type])
        return TorchDataLoader(dataset,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle, collate_fn=brits_collate_fn)


class DatasetForBRITS(BaseDataset):
    """Dataset class for BRITS.

    Parameters
    ----------
    data :
        The dataset for model input, should be a dictionary including keys as 'X' and 'y',
        or a path string locating a data file.
        If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
        which is time-series data for input, can contain missing values, and y should be array-like of shape
        [n_samples], which is classification labels of X.
        If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
        key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

    return_y :
        Whether to return labels in function __getitem__() if they exist in the given data. If `True`, for example,
        during training of classification models, the Dataset class will return labels in __getitem__() for model input.
        Otherwise, labels won't be included in the data returned by __getitem__(). This parameter exists because we
        need the defined Dataset class for all training/validating/testing stages. For those big datasets stored in h5
        files, they already have both X and y saved. But we don't read labels from the file for validating and testing
        with function _fetch_data_from_file(), which works for all three stages. Therefore, we need this parameter for
        distinction.

    file_type :
        The type of the given file if train_set and val_set are path strings.
    """

    def __init__(
            self,
            data: Union[dict, str],
            return_X_ori: bool,
            return_y: bool,
            file_type: str = "hdf5",
    ):
        super().__init__(
            data=data,
            return_X_ori=return_X_ori,
            return_X_pred=False,
            return_y=return_y,
            file_type=file_type,
        )

        if not isinstance(self.data, str):
            # calculate all delta here.
            if self.return_X_ori:
                forward_missing_mask = self.missing_mask
                forward_X = self.X
            else:
                forward_X, forward_missing_mask = fill_and_get_mask_torch(self.X)

            forward_delta = _parse_delta_torch(forward_missing_mask)
            backward_X = torch.flip(forward_X, dims=[1])
            backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])
            backward_delta = _parse_delta_torch(backward_missing_mask)

            self.processed_data = {
                "forward": {
                    "X": forward_X.to(torch.float32),
                    "missing_mask": forward_missing_mask.to(torch.float32),
                    "delta": forward_delta.to(torch.float32),
                },
                "backward": {
                    "X": backward_X.to(torch.float32),
                    "missing_mask": backward_missing_mask.to(torch.float32),
                    "delta": backward_delta.to(torch.float32),
                },
            }

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data from self.X if it is given.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            A list contains

            index : int tensor,
                The index of the sample.

            X : tensor,
                The feature vector for model input.

            missing_mask : tensor,
                The mask indicates all missing values in X.

            delta : tensor,
                The delta matrix contains time gaps of missing values.

            label (optional) : tensor,
                The target label of the time-series sample.
        """
        sample = [
            torch.tensor(idx),
            # for forward
            self.processed_data["forward"]["X"][idx],
            self.processed_data["forward"]["missing_mask"][idx],
            self.processed_data["forward"]["delta"][idx],
            # for backward
            self.processed_data["backward"]["X"][idx],
            self.processed_data["backward"]["missing_mask"][idx],
            self.processed_data["backward"]["delta"][idx],
        ]

        if self.return_X_ori:
            sample.extend([self.X_ori[idx], self.indicating_mask[idx]])

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        """Fetch data with the lazy-loading strategy, i.e. only loading data from the file while requesting for samples.
        Here the opened file handle doesn't load the entire dataset into RAM but only load the currently accessed slice.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            The collated data sample, a list including all necessary sample info.
        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
        X, missing_mask = fill_and_get_mask_torch(X)

        forward = {
            "X": X,
            "missing_mask": missing_mask,
            "deltas": _parse_delta_torch(missing_mask),
        }

        backward = {
            "X": torch.flip(forward["X"], dims=[0]),
            "missing_mask": torch.flip(forward["missing_mask"], dims=[0]),
        }
        backward["deltas"] = _parse_delta_torch(backward["missing_mask"])

        sample = [
            torch.tensor(idx),
            # for forward
            forward["X"],
            forward["missing_mask"],
            forward["deltas"],
            # for backward
            backward["X"],
            backward["missing_mask"],
            backward["deltas"],
        ]

        if self.return_X_ori:
            X_ori = torch.from_numpy(self.file_handle["X_ori"][idx]).to(torch.float32)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            indicating_mask = X_ori_missing_mask - missing_mask
            sample.extend([X_ori, indicating_mask])

        # if the dataset has labels and is for training, then fetch it from the file
        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
