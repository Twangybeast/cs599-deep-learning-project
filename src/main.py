import argparse
from pathlib import Path
import glob
import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WANDB = True
if WANDB:
    import wandb

def read_file_games(filename):
    with open(filename, "rb") as file:
        while True:
            try:
                boards = np.load(file)
                scores = np.load(file)
                elo = np.load(file)
            except ValueError:
                break
            yield (boards, scores, elo)


def game_to_tensor(boards, scores, game_start_move, max_game_length):
    # Boards: L x 8 x 8
    # Scores: (L,)
    boards = boards[game_start_move:game_start_move + max_game_length]
    scores = scores[game_start_move:game_start_move + max_game_length]
    num_moves = boards.shape[0]

    boards_tensor = torch.zeros(max_game_length, 8, 8, dtype=torch.long)
    scores_tensor = torch.zeros(max_game_length, dtype=torch.float)

    boards_tensor[:num_moves] = torch.tensor(np.array(boards[::-1]), dtype=torch.long)
    scores_tensor[:num_moves] = torch.tensor(np.array(scores[::-1]), dtype=torch.float)
    # Latest moves at 0

    scores_tensor = scores_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, 8, 8).unsqueeze(-1)

    # Who's turn?
    move_val = float((game_start_move + len(boards) - 1) % 2)

    # Skip first plane last dimension, those are just empty squares
    plane_tensor = F.one_hot(boards_tensor, 13)[:, :, :, 1:]
    # L x 8 x 8 x 12
    plane_tensor = torch.cat([plane_tensor, scores_tensor], dim=-1)
    # L x 8 x 8 x 13
    plane_tensor = torch.movedim(plane_tensor, -1, 1)
    # L x 13 x 8 x 8
    plane_tensor = plane_tensor.reshape(-1, 8, 8)
    # (L * 13) x 8 x 8

    move_tensor = torch.tensor(move_val, dtype=torch.float).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(1, 8, 8)
    # 1 x 8 x 8
    plane_tensor = torch.cat([plane_tensor, move_tensor], dim=0)
    # (L * 13 + 1) x 8 x 8

    # mask = torch.zeros(max_game_length, dtype=torch.float)
    # mask[:num_moves] = 1

    # move_tensor = torch.remainder(torch.arange(max_game_length), 2).unsqueeze(-1).unsqueeze(-1).expand(-1, 8, 8)
    # if game_start_move % 2 == 1:
    #     move_tensor = 1 - move_tensor
    # plane_tensor = torch.cat([move_tensor.unsqueeze(-1)], dim=-1)
    # return boards_tensor, plane_tensor, scores_tensor, mask
    return plane_tensor

class GamesTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, epoch_size, max_game_length=20, min_game_length=0, num_buckets=30, bucket_width=100, sampling_power=0.3, seed=None):
        self.elo_buckets = [[] for _ in range(num_buckets)]

        for filename in data_files:
            print(f"Loading data from file [{filename}]...")
            for boards, scores, elo in read_file_games(filename):
                if len(boards) < min_game_length:
                    continue
                bucket = min(num_buckets - 1, max(0, int(float(elo) // bucket_width)))
                self.elo_buckets[bucket].append((boards, scores, elo))
        
        # 0 sampling power is evenly from each bucket, 1 is normal sampling without buckets
        self.bucket_weights = np.array([len(elo_bucket) ** sampling_power for elo_bucket in self.elo_buckets])
        self.bucket_weights /= np.sum(self.bucket_weights)

        print("Bucket sizes: ", [len(elo_bucket) for elo_bucket in self.elo_buckets])
        print("Bucket weights: ", self.bucket_weights)
        self.epoch_size = epoch_size
        self.max_game_length = max_game_length

        self.datapoints = None
        self.rng = np.random.default_rng(seed)
        self.reset()

        self.mean_elo = sum((weight * sum((elo for _, _, elo in elo_bucket)) / len(elo_bucket) for weight, elo_bucket in zip(self.bucket_weights, self.elo_buckets) if weight > 1e-6))
    
    def reset(self):
        self.datapoints = []
        for i, bucket in enumerate(self.rng.choice(len(self.elo_buckets), size=self.epoch_size, p=self.bucket_weights)):
            game_idx = self.rng.choice(len(self.elo_buckets[bucket]))
            game_length = len(self.elo_buckets[bucket][game_idx][1])
            game_start_move = 0 if game_length <= self.max_game_length else self.rng.choice(game_length - self.max_game_length + 1)
            self.datapoints.append((bucket, game_idx, game_start_move))
    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, idx):
        bucket, game_idx, game_start_move = self.datapoints[idx]
        boards, scores, elo = self.elo_buckets[bucket][game_idx]

        plane_tensor = game_to_tensor(boards, scores, game_start_move, self.max_game_length)
        return plane_tensor, torch.tensor(elo, dtype=torch.float)

class ChessNet(nn.Module):
    def __init__(self, filter_size, num_layers, max_seq_length, hidden_filter_size=32, hidden_size=128):
        super(ChessNet, self).__init__()
        input_filter_size = max_seq_length * 13 + 1
        self.net = nn.Sequential(ChessConvModule(input_filter_size, filter_size), 
                                *[ChessResModule(filter_size) for i in range(num_layers)])
        self.head = ChessEloHead(filter_size, hidden_filter_size=hidden_filter_size, hidden_size=hidden_size)
    
    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x

class ChessConvModule(nn.Module):
    def __init__(self, input_filter_size, output_filter_size):
        super(ChessConvModule, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(input_filter_size, output_filter_size, 3, padding='same')
        self.bn = nn.BatchNorm2d(output_filter_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChessResModule(nn.Module):
    def __init__(self, filter_size, input_size=None):
        super(ChessResModule, self).__init__()
        if input_size is None:
            input_size = filter_size
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_size, filter_size, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(filter_size)
        self.conv2 = nn.Conv2d(filter_size, filter_size, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(filter_size)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += x
        y = self.relu(y)
        return y

class ChessEloHead(nn.Module):
    def __init__(self, in_filter_size, hidden_filter_size = 32, hidden_size = 128):
        super(ChessEloHead, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_filter_size, hidden_filter_size, 3, padding='same')
        self.bn = nn.BatchNorm2d(hidden_filter_size)
        self.dense1 = nn.Linear(hidden_filter_size * 8 * 8, hidden_size)
        self.dense2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(len(x), -1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x.squeeze(-1)

class ChessCNN(nn.Module):
    def __init__(self, filter_size, num_layers):
        super(ChessCNN, self).__init__()
        self.net = nn.Sequential(*[ChessResModule(filter_size) for i in range(num_layers)])
    
    def forward(self, x):
        return self.net(x)

def loss_func(pred, actual):
    return nn.MSELoss()(pred / 300, actual / 300)

# def loss_func(pred, actual, mask):
#     mask = mask.unsqueeze(-1)
#     actual = actual.unsqueeze(-1).unsqueeze(-1).expand(-1, pred.size()[1], -1)
#     return nn.MSELoss(reduction='sum')(pred * mask / 300, actual * mask / 300) / torch.sum(mask)


def train(model, device, optimizer, train_loader, epoch, log_interval):
    model.train()
    losses = []
    maes = []
    for batch_idx, (plane_tensor, elo_tensor) in enumerate(tqdm(train_loader)):
        plane_tensor = plane_tensor.to(device)
        elo_tensor = elo_tensor.to(device)

        optimizer.zero_grad()
        output = model(plane_tensor)
        loss = loss_func(output, elo_tensor)
        losses.append(loss.item())
        maes.append(torch.abs(output - elo_tensor).mean().item())
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(plane_tensor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses), np.mean(maes)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    absolute_error = 0
    naive_mae = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (plane_tensor, elo_tensor) in enumerate(test_loader):
            plane_tensor = plane_tensor.to(device)
            elo_tensor = elo_tensor.to(device)

            batch_size = len(plane_tensor)
            total += batch_size
            output = model(plane_tensor)

            test_loss += loss_func(output, elo_tensor).item() * batch_size

            absolute_error += torch.sum(torch.abs(output - elo_tensor)).item()
            naive_mae += torch.sum(torch.abs(elo_tensor - test_loader.dataset.mean_elo)).item()

    test_loss /= total
    absolute_error /= total
    naive_mae /= total

    print('\nTest set: Average loss: {:.4f}, MAE: {:.4f}, Naive MAE {:.4f}\n'.format(
        test_loss, absolute_error, naive_mae))
    return test_loss, absolute_error

        
def main():
    USE_CUDA = True
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 0.0005
    PRINT_INTERVAL = 100
    EPOCHS = 50
    # BASE_PATH = "....."
    MODEL_PATH = None
    FILTER_SIZE = 512
    NUM_LAYERS = 30
    MAX_GAME_LENGTH = 12
    HIDDEN_FILTER_SIZE = 32
    FINAL_HIDDEN_SIZE = 128
    MODEL_PREFIX = "v5_"
    if WANDB:
        config = {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "filter_size": FILTER_SIZE,
            "num_layers": NUM_LAYERS,
            "max_game_length": MAX_GAME_LENGTH,
            "hidden_filter_size": HIDDEN_FILTER_SIZE,
            "final_hidden_size": FINAL_HIDDEN_SIZE,
            "version": "v5",
            "model_prefix": MODEL_PREFIX
        }
        wandb.init(project="deep-learning-final-project", entity="uw-d0", config=config)

    TRAIN_FILENAMES = ["worker01.npy", "worker02.npy", "worker03.npy", "worker04.npy", "worker05.npy"]
    # TRAIN_FILENAMES = ["worker01.npy"]
    TEST_FILENAMES = ["worker07.npy"]
    
    
    data_train = GamesTrainDataset([os.path.join(BASE_PATH, filename) for filename in TRAIN_FILENAMES],
                    epoch_size=51200, max_game_length=MAX_GAME_LENGTH, min_game_length=5)
    # data_test = data_train
    data_test = GamesTrainDataset([os.path.join(BASE_PATH, filename) for filename in TEST_FILENAMES], 
                    epoch_size=12800, max_game_length=MAX_GAME_LENGTH, min_game_length=MAX_GAME_LENGTH, seed=4909)
    # data_eval = GamesTestDataset(["data/worker07.npy"], 20, max_datapoints=10000)

    checkpoint_root = os.path.join(BASE_PATH, "checkpoints", MODEL_PREFIX)
    Path(checkpoint_root).mkdir(exist_ok=True)

    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                               shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                              shuffle=False, **kwargs)

    # model = ChessNet(FILTER_SIZE, FILTER_SIZE, ChessCNN(FILTER_SIZE, 10)).to(device)
    model = ChessNet(FILTER_SIZE, NUM_LAYERS, MAX_GAME_LENGTH, hidden_filter_size=HIDDEN_FILTER_SIZE, hidden_size=FINAL_HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_losses = []
    test_losses = []
    test_maes = []
    start_epoch = 1
    best_test_mae = None
    best_epoch = None

    if MODEL_PATH is not None:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_test_mae = checkpoint["best_test_mae"]
        best_epoch = checkpoint["best_epoch"]

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"Starting epoch {epoch}")
        data_train.reset()
        train_loss, train_mae = train(model, device, optimizer, train_loader, epoch, PRINT_INTERVAL)
        test_loss, test_mae = test(model, device, test_loader)

        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))
        test_maes.append((epoch, test_mae))

        if best_test_mae is None or test_mae < best_test_mae:
            best_test_mae = test_mae
            best_epoch = epoch
        
        if WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "test_loss": test_loss,
                "test_mae": test_mae
                })
        # pt_util.write_log(LOG_PATH, (train_losses, test_losses, test_accuracies))
        # model.save_best_model(test_accuracy, DATA_PATH + 'checkpoints/%03d.pt' % epoch)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "test_loss": test_loss,
                "test_mae": test_mae,
                "best_test_mae": best_test_mae,
                "best_epoch": best_epoch
            }, os.path.join(checkpoint_root, f"{epoch}.pt"))
        for checkpoint_file in glob.glob(os.path.join(checkpoint_root, "*.pt")):
            basename = os.path.basename(checkpoint_file)
            if basename == f"{epoch}.pt" or basename == f"{best_epoch}.pt":
                break
            os.remove(checkpoint_file)
# main()
if __name__ == '__main__':
    main()
