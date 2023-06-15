from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio

## Loader for training, uses one mat file per datapoint
class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            gt_path = path

            if os.path.exists(gt_path):
                gt = os.listdir(gt_path)
                self.data = [{'orig': gt_path + '/' + gt[i]} for i in range(len(gt))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

         data = self.data[index]["orig"]
         data = scio.loadmat(data)        
         meas0 = data['meas0']
         meas0 = torch.squeeze(torch.from_numpy(meas0))
         meas1 = data['meas1']
         meas1 = torch.squeeze(torch.from_numpy(meas1))         
         measc = data['measc']
         measc = torch.squeeze(torch.from_numpy(measc))

         return meas0,meas1,measc

    def __len__(self):

        return len(self.data)