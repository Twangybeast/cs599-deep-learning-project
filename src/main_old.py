import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    boards = boards[game_start_move:game_start_move + max_game_length]
    scores = scores[game_start_move:game_start_move + max_game_length]
    num_moves = boards.shape[0]

    boards_tensor = torch.zeros(max_game_length, 8, 8, dtype=torch.long)
    scores_tensor = torch.zeros(max_game_length, dtype=torch.float)

    boards_tensor[:num_moves] = torch.tensor(boards, dtype=torch.long)
    scores_tensor[:num_moves] = torch.tensor(scores, dtype=torch.float)

    mask = torch.zeros(max_game_length, dtype=torch.float)
    mask[:num_moves] = 1

    move_tensor = torch.remainder(torch.arange(max_game_length), 2).unsqueeze(-1).unsqueeze(-1).expand(-1, 8, 8)
    if game_start_move % 2 == 1:
        move_tensor = 1 - move_tensor
    plane_tensor = torch.cat([move_tensor.unsqueeze(-1)], dim=-1)
    return boards_tensor, plane_tensor, scores_tensor, mask

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

        boards_tensor, plane_tensor, scores_tensor, mask = game_to_tensor(boards, scores, game_start_move, self.max_game_length)
        # Returns:
        # Board: L x 8 x 8, long tensor
        # Plane: L x 8 x 8 x 1, float tensor
        # Eval: (L,), float tensor
        # ELO: (,), float tensor
        # Mask: (L,), float tensor
        return boards_tensor, plane_tensor, scores_tensor, torch.tensor(elo, dtype=torch.float), mask

class ChessNet(nn.Module):
    def __init__(self, in_filter_size, out_filter_size, cnn):
        super(ChessNet, self).__init__()
        self.in_filter_size = in_filter_size
        self.out_filter_size = out_filter_size

        self.embedding = nn.Embedding(13, self.in_filter_size - 1)
        self.cnn = cnn
        self.rnn = nn.LSTM(self.out_filter_size + 1, self.out_filter_size + 1, batch_first=True)
        self.head = nn.Linear(self.out_filter_size + 1, 1)
    
    def forward(self, boards_tensor, plane_tensor, scores_tensor):
        # B x L x 8 x 8
        x = self.embedding(boards_tensor)
        # B x L x 8 x 8 x (IN_FILTER_SIZE - 1)

        x = torch.cat([x, plane_tensor], dim=-1)
        # B x L x 8 x 8 x IN_FILTER_SIZE
        x = torch.movedim(x, -1, 2)
        # B x L x IN_FILTER_SIZE x 8 x 8
        original_size = x.size()
        x = self.cnn(x.view(-1, self.in_filter_size, 8, 8))
        # (B * L) x OUT_FILTER_SIZE x 8 x 8
        x = x.view(*original_size)
        # B x L x OUT_FILTER_SIZE x 8 x 8
        x = x.mean(dim=(-1, -2))
        # B x L x OUT_FILTER_SIZE

        x = torch.cat([x, scores_tensor.unsqueeze(-1)], dim=-1)
        # B x L x (OUT_FILTER_SIZE + 1)
        x, hidden = self.rnn(x)

        # B x L x (OUT_FILTER_SIZE + 1)
        x = self.head(x)
        # B x L x 1
        return x * 300 + 1600

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

class ChessCNN(nn.Module):
    def __init__(self, filter_size, num_layers):
        super(ChessCNN, self).__init__()
        self.net = nn.Sequential(*[ChessResModule(filter_size) for i in range(num_layers)])
    
    def forward(self, x):
        return self.net(x)

def loss_func(pred, actual, mask):
    mask = mask.unsqueeze(-1)
    actual = actual.unsqueeze(-1).unsqueeze(-1).expand(-1, pred.size()[1], -1)
    return nn.MSELoss(reduction='sum')(pred * mask / 300, actual * mask / 300) / torch.sum(mask)


def train(model, device, optimizer, train_loader, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (boards_tensor, plane_tensor, scores_tensor, elo_tensor, mask) in enumerate(tqdm(train_loader)):
        boards_tensor = boards_tensor.to(device)
        plane_tensor = plane_tensor.to(device)
        scores_tensor = scores_tensor.to(device)
        elo_tensor = elo_tensor.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        output = model(boards_tensor, plane_tensor, scores_tensor)
        loss = loss_func(output, elo_tensor, mask)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(boards_tensor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    absolute_error = 0
    naive_mae = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (boards_tensor, plane_tensor, scores_tensor, elo_tensor, mask) in enumerate(test_loader):
            boards_tensor = boards_tensor.to(device)
            plane_tensor = plane_tensor.to(device)
            scores_tensor = scores_tensor.to(device)
            elo_tensor = elo_tensor.to(device)
            mask = mask.to(device)

            batch_size = len(boards_tensor)
            total += batch_size
            output = model(boards_tensor, plane_tensor, scores_tensor)

            test_loss += loss_func(output, elo_tensor, mask).item() * batch_size

            # last_idxs = torch.round(mask.sum(dim=-1)).int() - 1
            # pred_elos = torch.tensor([output[i, j] for i, j in enumerate(last_idxs)]).to(device)
            pred_elos = output[:, -1, 0]
            absolute_error += torch.sum(torch.abs(pred_elos - elo_tensor)).item()
            naive_mae += torch.sum(torch.abs(elo_tensor - test_loader.dataset.mean_elo)).item()

    test_loss /= total
    absolute_error /= total
    naive_mae /= total

    print('\nTest set: Average loss: {:.4f}, MAE: {:.4f}, Naive MAE{:.4f}\n'.format(
        test_loss, absolute_error, naive_mae))
    return test_loss, absolute_error

        
def main():
    USE_CUDA = True
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 0.0005
    PRINT_INTERVAL = 100
    EPOCHS = 20
    # BASE_PATH = "....."
    MODEL_PATH = None
    FILTER_SIZE = 64

    TRAIN_FILENAMES = ["worker01.npy", "worker02.npy", "worker03.npy", "worker04.npy", "worker05.npy"]
    # TRAIN_FILENAMES = ["worker01.npy"]
    TEST_FILENAMES = ["worker07.npy"]
    
    data_train = GamesTrainDataset([os.path.join(BASE_PATH, filename) for filename in TRAIN_FILENAMES],
                    epoch_size=51200, max_game_length=20, min_game_length=5)
    # data_test = data_train
    data_test = GamesTrainDataset([os.path.join(BASE_PATH, filename) for filename in TEST_FILENAMES], epoch_size=12800, max_game_length=20, min_game_length=20, seed=4909)
    # data_eval = GamesTestDataset(["data/worker07.npy"], 20, max_datapoints=10000)

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

    model = ChessNet(FILTER_SIZE, FILTER_SIZE, ChessCNN(FILTER_SIZE, 10)).to(device)
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
        train_loss = train(model, device, optimizer, train_loader, epoch, PRINT_INTERVAL)
        test_loss, test_mae = test(model, device, test_loader)

        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))
        test_maes.append((epoch, test_mae))

        if best_test_mae is None or test_mae < best_test_mae:
            best_test_mae = test_mae
            best_epoch = epoch
        # pt_util.write_log(LOG_PATH, (train_losses, test_losses, test_accuracies))
        # model.save_best_model(test_accuracy, DATA_PATH + 'checkpoints/%03d.pt' % epoch)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_mae": test_mae,
                "best_test_mae": best_test_mae,
                "best_epoch": best_epoch
            }, os.path.join(BASE_PATH, f"checkpoints/v2_{epoch}.pt"))
main()
# if __name__ == '__main__':
#     main()
