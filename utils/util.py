import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def freeze_network(network):
    for p in network.parameters():
        p.requires_grad = False


def unfreeze_network(network):
    for p in network.parameters():
        p.requires_grad = True