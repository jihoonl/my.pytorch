from torch.utils.data import DataLoader

NUM_WORKERS = 8

def train_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=NUM_WORKERS)
