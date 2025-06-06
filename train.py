from torch.backends import cudnn
from torch import cuda

from backbone.solver import Solver
from dataloder import get_loader
from setup import setup_config, setup_argparser
from utils import read_img


def train_net(config):
    # change this dir to attack different identities
    TARGET_PATH = './assets/datasets/target/001.jpeg'

    cudnn.benchmark = True
    data_loader = get_loader(config)
    target_image = read_img(TARGET_PATH, 0.5, 0.5, config.DEVICE.device)
    solver = Solver(config, target_image, data_loader=data_loader)
    solver.train()


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    config = setup_config(args)
    print("Call with args:")
    print(config)
    print("Max memory allocated:")
    print(cuda.max_memory_allocated(0))
    train_net(config)
