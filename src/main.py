import argparse
import glob

import numpy as np
# import torch
# import torch.nn as nn

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

    mask = torch.zeros(self.max_game_length, dtype=torch.float)
    mask[:num_moves] = 1
    return boards_tensor, scores_tensor, mask

class GamesTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, epoch_size, max_game_length=20, min_game_length=0, num_buckets=30, bucket_width=100, sampling_power=0.5):
        self.elo_buckets = [[] for _ in range(num_buckets)]

        for filename in data_files:
            for boards, scores, elo in read_file_games(filename):
                if len(boards) < min_game_length:
                    continue
                bucket = min(num_buckets - 1, max(0, float(elo) // bucket_width))
                self.elo_buckets[bucket].append((boards, scores, elo))
        
        # 0 sampling power is evenly from each bucket, 1 is normal sampling without buckets
        self.bucket_weights = np.array([len(elo_bucket) ** sampling_power for elo_bucket in self.elo_buckets])
        self.bucket_weights /= np.sum(self.bucket_weights)
        self.epoch_size = epoch_size
        self.max_game_length = max_game_length

        self.datapoints = None
        self.reset()
    
    def reset(self):
        self.datapoints = []
        for i, bucket in enumerate(np.random.choice(len(self.elo_buckets), size=self.epoch_size, p=self.bucket_weights))
            game_idx = np.random.choice(len(self.elo_buckets[bucket]))
            game_length = len(self.elo_buckets[bucket][game_idx][1])
            game_start_move = 0 if game_length <= self.max_game_length else np.random.choice(game_length - self.max_game_length + 1)
            self.datapoints.append((bucket, game_idx, game_start_move))
    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(idx):
        bucket, game_idx, game_start_move = self.datapoints[idx]
        boards, scores, elo = self.elo_buckets[bucket][game_idx]

        boards_tensor, scores_tensor, mask = game_to_tensor(boards, scores, game_start_move, self.max_game_length)
        # Returns:
        # Board: L x 8 x 8, long tensor
        # Eval: (L,), float tensor
        # ELO: (,), float tensor
        # Mask: (L,), float tensor
        return boards_tensor, scores_tensor, torch.tensor(elo, dtype=torch.float), mask  


class GamesTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, max_game_length, min_game_length=None):
        if min_game_length is None:
            min_game_length = max_game_length
        self.games = []
        for filename in data_files:
            for boards, scores, elo in read_file_games(filename):
                if len(boards) < min_game_length:
                    continue
                bucket = min(num_buckets - 1, max(0, float(elo) // bucket_width))
                self.games.append((boards, scores, elo))
        self.max_game_length = max_game_length

        rng = np.random.default_rng(1235)
        self.game_starts = []
        for boards, _, _ in self.games:
            game_length = len(boards)
            game_start_move = 0 if game_length <= self.max_game_length else rng.choice(game_length - self.max_game_length + 1)
            self.game_starts.append(game_start_move)
        
    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, idx):
        boards, scores, elo = self.games[idx]
        boards_tensor, scores_tensor, mask = game_to_tensor(boards, scores, game_start_move, self.max_game_length)
        return boards_tensor, scores_tensor, torch.tensor(elo, dtype=torch.float), mask  

class ChessNet(nn.Module):
    def __init__(self, in_filter_size, cnn):
        super(ChessNet, self).__init__()
        self.in_filter_size = in_filter_size
        self.out_filter_size = out_filter_size

        self.embedding = nn.Embedding(13, self.in_filter_size, )
        self.cnn = cnn
        self.rnn = nn.LSTM(self.out_filter_size + 1, self.out_filter_size + 1, batch_first=True)
        self.head = nn.Linear(self.out_filter_size + 1, 1)
    
    def forward(self, boards_tensor, scores_tensor):
        # B x L x 8 x 8
        x = self.embedding(boards_tensor)
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

        
        x = torch.cat([x, scores_tensor.view(-1, 1, 1).expand(-1, x.size()[1], -1)])
        # B x L x (OUT_FILTER_SIZE + 1)
        x, hidden = self.rnn(x)

        # B x L x (OUT_FILTER_SIZE + 1)
        x = self.head(x)
        # B x L x 1
        return x * 300 + 1600

class ChessResModule(nn.Module):
    def __init__(self, filter_size):
        super(ChessResModule, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(filter_size, filter_size, 3, padding='same')
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
        self.net = nn.Sequential([ChessResModule(filter_size) for i in range(num_layers)])
    
    def forward(self, x):
        return self.net(x)

def loss_func(pred, actual, mask):
    mask = mask.unsqueeze(-1)
    actual = actual.unsqueeze(-1)
    return nn.MSELoss(reduction='sum')(pred * mask / 300, actual * mask / 300) / torch.sum(mask)


def train(model, device, optimizer, train_loader, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (boards_tensor, scores_tensor, elo_tensor, mask) in enumerate(tqdm.tqdm(train_loader)):
        boards_tensor = boards_tensor.to(device)
        scores_tensor = scores_tensor.to(device)
        elo_tensor = elo_tensor.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        output = model(boards_tensor, scores_tensor)
        loss = loss_func(output, elo_tensor, mask)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    absolute_error = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (boards_tensor, scores_tensor, elo_tensor, mask) in enumerate(test_loader):
            boards_tensor = boards_tensor.to(device)
            scores_tensor = scores_tensor.to(device)
            elo_tensor = elo_tensor.to(device)
            mask = mask.to(device)

            batch_size = len(boards_tensor)
            total += batch_size
            output = model(boards_tensor, scores_tensor)

            test_loss += loss_func(output, elo_tensor, mask).item() * batch_size

            # last_idxs = torch.round(mask.sum(dim=-1)).int() - 1
            # pred_elos = torch.tensor([output[i, j] for i, j in enumerate(last_idxs)]).to(device)
            pred_elos = output[:, -1, 0]
            absolute_error = torch.sum(torch.abs(pred_elos - elo_tensor)).item()




            pred = output.max(-1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct

    test_loss /= total
    absolute_error /= total

    print('\nTest set: Average loss: {:.4f}, MAE: {:.4f}\n'.format(
        test_loss, absolute_error))
    return test_loss, absolute_error

        
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("files", nargs='+')
    # args = parser.parse_args()


    # for filename in glob.glob("data/*.npy"):
    #     for game in read_file_games(filename):
    #         pass

if __name__ == '__main__':
    main()
