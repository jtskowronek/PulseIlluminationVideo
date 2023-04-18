from models.my_tools import *


class re_3dcnn(nn.Module):

    def __init__(self, num_block=8,num_group=2):
        super(re_3dcnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
        )

        self.layers = nn.ModuleList()
        for i in range(num_block):
            self.layers.append(rev_3d_part1(64, num_group))

    def forward(self, y):
        y = torch.unsqueeze(y,0)
        out = self.conv1(y)

        for layer in self.layers:
            out = layer(out)

        out = self.conv2(out)

        return out[0,:]