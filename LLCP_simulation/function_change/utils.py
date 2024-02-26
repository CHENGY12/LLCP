import os
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
import torch
import pickle


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


class SimulatedTrainDataset(datasets.VisionDataset):

    def __init__(self, data_path):
        super(SimulatedTrainDataset, self).__init__(data_path, transform=None, target_transform=None)
        self.structure = None
        self.data_list = self.parse_data_file(data_path)

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def parse_data_file(self, data_path):
        data_open = open(data_path, "rb")
        data = pickle.load(data_open)

        return data
        pass


def parser_lstm_data(batch_data):

    x_list = {i: [] for i in range(10)}
    y_list = []

    for frame_idx in range(16):
        for var_id in range(10):
            x_list[var_id].append(batch_data[frame_idx*10 + var_id][0])
            one_frame_batch_y = torch.unsqueeze(batch_data[frame_idx*10 + var_id][2], dim=-1)
            y_list.append(one_frame_batch_y)

    one_vedio_y = torch.vstack(y_list)
    tmp_x_list = {i: torch.vstack(x_list[i]) for i in range(10)}
    total_data = []
    for idx in range(10):
        total_data.append(torch.unsqueeze(tmp_x_list[idx], dim=0))

    total_data = torch.vstack(total_data)
    return total_data, one_vedio_y


def parser_data(batch_data):
    x_list = []
    c_list = []
    y_list = []

    for frame in batch_data:
        one_frame_batch_x = torch.unsqueeze(frame[0], dim=-1)
        one_frame_batch_c = torch.hstack(frame[1]).float()
        one_frame_batch_y = torch.unsqueeze(frame[2], dim=-1)

        x_list.append(one_frame_batch_x)
        c_list.append(one_frame_batch_c)
        y_list.append(one_frame_batch_y)

    one_vedio_x = torch.vstack(x_list)
    one_vedio_c = torch.vstack(c_list)
    one_vedio_y = torch.vstack(y_list)

    return one_vedio_x, one_vedio_c, one_vedio_y





















