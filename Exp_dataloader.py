from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio
import torchvision.transforms.functional as F


class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            data_path = path

            if os.path.exists(data_path):
                data = os.listdir(data_path)
                self.data = [{'data': data_path + '/' + data[i]} for i in
                             range(len(data))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):
        data_in = self.data[index]["data"]
        mat_in = scio.loadmat(data_in)
        
        meas = mat_in['meas1']
        meas = torch.from_numpy(meas)
        meas1 = torch.unsqueeze(meas,0)
        
        meas = mat_in['measc']
        meas = torch.from_numpy(meas)
        measc = torch.unsqueeze(meas,0)
        
        meas = mat_in['meas2']
        meas = torch.from_numpy(meas)
        meas2 = torch.unsqueeze(meas,0)
        
        

        return meas1,measc,meas2

    def __len__(self):

        return len(self.data)
