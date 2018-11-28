__all__ = ['pascal']

import os, sys
from dataloaders.datasets import pascal
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    if args.dataset == 'pascal':
        train_set = pascal.VOCDataset(args, mode = 'train')
        val_set = pascal.VOCDataset(args, mode = 'val')

        num_class = 21
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError