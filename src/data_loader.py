import h5py
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
import numpy as np
from torch.utils.data import Subset, DataLoader
from consts import *

class ChessHiddenStateDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.total_positions = f['lc0_hidden'].shape[0]
            self.num_games = self.total_positions // POSITIONS_PER_GAME
            assert self.total_positions % POSITIONS_PER_GAME == 0
            print(f"Adatok: {self.total_positions} állás, ami {self.num_games} teljes játék.")

        self.archive = None

    def __len__(self):
        return self.num_games

    def __getitem__(self, idx):
        if self.archive is None:
            self.archive = h5py.File(self.file_path, 'r', swmr=True)
        
        start = int(idx) * POSITIONS_PER_GAME
        state = self.archive['lc0_hidden'][start : start + POSITIONS_PER_GAME]

        return torch.from_numpy(state).float() 

def make_dloaders(file_path=DATA_FILE):

    batch_size_in_games = BATCH_GAMES
    val_data_ratio = VAL_RATIO
    test_data_ratio = TEST_RATIO
    dataset = ChessHiddenStateDataset(file_path)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    val_split = int(np.floor(val_data_ratio * dataset_size)) 
    test_split = int(np.floor(test_data_ratio * dataset_size)) 

    np.random.seed(42)
    np.random.shuffle(indices)

    test_indices = indices[:test_split]
    val_indices = indices[test_split : test_split + val_split]
    train_indices = indices[test_split + val_split :]

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_in_games,
                            shuffle=True, num_workers=DLOADER_WORKERS, pin_memory=True, drop_last=True)

    val_loader   = DataLoader(val_dataset, batch_size=batch_size_in_games, 
                            shuffle=False, num_workers=DLOADER_WORKERS, pin_memory=True, drop_last=True)

    test_loader  = DataLoader(test_dataset, batch_size=batch_size_in_games, 
                            shuffle=False, num_workers=DLOADER_WORKERS, pin_memory=True, drop_last=True)

    print(f"Összes játék:  {dataset_size}")
    print(f"Tanító:  {len(train_dataset)}")
    print(f"Validációs: {len(val_dataset)}")
    print(f"Teszt:     {len(test_dataset)}")

    return train_loader, val_loader, test_loader

