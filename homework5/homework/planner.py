import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),    # Output: (B, 16, 48, 64)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # Output: (B, 32, 24, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # Output: (B, 64, 12, 16)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (B, 128, 6, 8)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Output: (B, 256, 3, 4)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Output: (B, 128, 6, 8)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Output: (B, 64, 12, 16)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # Output: (B, 32, 24, 32)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)   # Output: (B, 16, 48, 64)
        self.batch_norm4 = nn.BatchNorm2d(16)
        self.conv5 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)    # Output: (B, 1, 96, 128)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))  # Output heatmap of size (B, 1, 96, 128)
        return x

class Planner(torch.nn.Module):
    def __init__(self):
        super(Planner, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, img):
        encoded = self.encoder(img)
        heatmap = self.decoder(encoded)
        aim_point = spatial_argmax(heatmap.squeeze(1))  # Remove channel dimension before spatial_argmax
        return aim_point

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
