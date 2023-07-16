from torch.utils.data import Dataset
import os
import torch
import scipy as sci

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
         data = torch.load(data)
         meas = data[0]        
         gt = data[1]

         return gt, meas

    def __len__(self):

        return len(self.data)
    

class ImgdatasetMat(Dataset):

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
         data = sci.loadmat(data)
         meas = data[0]        
         gt = data[1]

         return gt, meas

    def __len__(self):

        return len(self.data)    