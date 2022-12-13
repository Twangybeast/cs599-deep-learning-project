## Abstract
Chess is perhaps the most popular studied game in the field of artificial intelligence. The current state-of-the-art chess engines use variants of minimax combined with neural networks to evaluate positions. Although much work has been done in evaluating chess positions, i.e. who is winning and by how much, much less work has been done in evaluating the strength of the players from the chess position. In our work, we take a chess position and the last 6 moves played and estimate the average strength of the two players, measured by Lichess’s Elo system.
We use publicly available games that real people played on Lichess from 2018, and input game states into our convolutional neural network model. We used held out data to test our model based on the mean absolute error of the guessed and actual average Elo rating. We include a demo with our model predicting the Elo rating from a chess PGN string.


## Video Summary
https://youtu.be/o_t1yP3T0YU

## Demo
To use our model, all you need is a PGN string of a chess game. The PGN format is very common, and we’re pretty sure any online chess website has a way to export games as PGN strings. Also, you can create a PGN using various online board editors. We have a demo in the link below, which also includes additional details on how to make a PGN.

https://colab.research.google.com/drive/1bFW1oKJjLClVR2tP_cs4-8G_e-Zdvsze#scrollTo=bivAqCA3wOnZ 


## Problem statement
Chess is an incredibly famous game, even outside specialized communities. Most people have doubtlessly watched or played a game of chess. When stumbling upon a pair of people playing chess and observing the game for a few moves, it can be quite hard to estimate how good the players are, especially if one doesn’t know much about chess. Even when using a computer engine to evaluate who’s winning, large swings in evaluation scores could be due to inhumanly complicated tactics and sacrifices, and simply the fact that a player is winning doesn’t indicate how good the total strength of the players is. 

Our goal is to take the last few moves of a chess game, which may be in progress, and use a model to predict the total skill of the players, assuming the players are approximately equally skilled.

## Related Work
We were inspired by existing work in applying deep learning to chess, especially AlphaZero and Leela Chess Zero, which use reinforcement learning and Monte Carlo tree search with a deep convolutional model to learn to play chess from scratch. We were also motivated by the fact that the current best chess engine in the world, Stockfish 15, uses a neural network to evaluate positions as well. 

We were also inspired by human attempts at guessing the Elo rating of players from chess games, such as by GothamChess, though our model doesn’t have the advantage of looking at the entirety of a game, looking at the top 3 engine moves, and domain knowledge of how to play chess.

We used a dataset of Lichess games https://database.lichess.org/, which includes billions of chess games in the PGN format, though we did not use all of it. This dataset includes information such as the Elo ratings of the players, the moves they played, the time control setting and how long they took per move, and various other meta-information about the game. Our training data was from January 2018 to May 2018, and test data was from July 2018. 

## Methodology
### Data
We used a dataset from Lichess as described previously. In chess, time controls dictate how much time each player has to think, and faster time controls means the players have to play faster. For training data, we use approximately 100,000 chess games. For test data, we use about 20,000 chess games. To keep the games consistent, we only examine games with a time control of “300+0”, which is equivalent to each player having 5 minutes for the entire game. 

Additionally, our model uses the engine evaluation scores for each move. It’s computationally infeasible for us to self compute those, e.g. if we ran the engine for 0.2 seconds for each board state, average 80 board states per game, then that would take almost 20 days. So, we leverage existing evaluation scores from our dataset. Unfortunately, only about 6% of the dataset has these evaluation scores. Combined with our existing time control limitation, we’re only able to use about 1% of the games. Using a Python PGN parser, we’re only able to parse 4 games per second, and this was a major limitation in how much data we could use.

The dataset used is quite imbalanced in terms of Elo ratings. Naturally, Elo ratings are normally distributed. Unfortunately, this imbalance means that extreme Elo ratings, such as very good players or very bad players, are underrepresented. To solve this issue, we used a rebalancing trick to upsample extreme data points. We bucket each datapoint into 30 buckets that are 100 points wide, e.g. the 15th bucket consists of games with Elo ratings of 1400-1499. Then, we sample a bucket randomly proportional to the size of the bucket raised to the power of 0.3. Then we pick a datapoint uniformly at random from that bucket. Intuitively, if we sample proportional to the size of the bucket, this is equivalent to picking a random datapoint. If we sample a bucket uniformly, it’ll mean we almost have every Elo score uniformly. Since data is sparse at the extremities, we can’t do the latter. Using a power of 0.3 was empirically a good middle ground between these two ideas, which flattens the normal distribution of the ratings. Also, for each datapoint sampled, we randomly pick a point in the middle of the game for the model to look at, since the model cannot see the entire game.

During training, we resample all the datapoints using this mechanism at the beginning of each epoch. For test data, we randomly pick them once, and don’t resample later.

Note that we use Lichess Elo ratings for our project, which are not equivalent to other Elo ratings. Usually, subtracting a correction constant allows for an approximate estimate of how the Elo rating translates.

### Game Representation
We represent a chess board state sparsely. For each of the 12 chess pieces (6 pieces per side, two colors), we use a “one-hot” encoding. In other words, for each piece, we have an 8x8 tensor of ones and zeros, where the ones correspond to the location of the piece (e.g. white bishop). Note that since there may be multiple of a given piece, e.g. 8 white pawns, the tensor may have multiple ones. Since there are 12 pieces, this is a 12x8x8 tensor. We also include the Stockfish evaluation of the game, representing which side is winning, in terms of pawns. If one side has a forced mate, we represent that as a 10 pawns advantage. We create another 8x8 tensor filled with this engine evaluation score. This creates a 13x8x8 tensor for each board state. To include the history of prior board states, we concatenate the representations of each board state. So, if we want the model to use the last 4 board states, we have a (4*13)x8x8 tensor. Lastly, we concatenate an 8x8 tensor of all zeros or all ones depending on whose move it is (white or black).

Note that our tensor representation of the last few moves does not fully describe the game. There are additional edge rules in chess, such as threefold repetition and ability to castle, that we don’t encode. With the exception of very adverse cases, the state of these rules can usually be guessed from our representation. For example, if the king and rook are on their starting square, they can probably castle. Intuitively, these edge cases likely don’t matter.

Methodology - Model
Our neural network architecture is similar to that of AlphaZero and Leela Chess Zero, though we made major modifications. We based our architecture off resources from [Leela Chess](https://lczero.org/dev/backend/nn/) and [AlphaZero](https://adspassets.blob.core.windows.net/website/content/alpha_go_zero_cheat_sheet.png) from those respective links.

All convolutions are 3x3, stride 1, and padding to ensure the output is the same as the input. All convolutions are immediately followed by a batch normalization layer, and with the exception of the last layer, output the same number of channels. 

As input, we take a (L*13+1)x8x8 tensor as described above. We first do a convolution and a ReLU. Then, we stack some number of residual blocks on top of that. Each residual block consists of a convolution, ReLU, convolution, residual from the start of the block, and a ReLU. Finally, we do a last convolution layer, ReLU, fully connected layer, ReLU, and fully connected layer. 

The primary differences between our architecture and those from AlphaZero and Leela Zero is that our hyperparameters for model size are different, and our input tensor contains some different data. Additionally, the final head which we use to predict the Elo score is also different. 

Initially, we used a completely different architecture that would encode each board state as a vector, then use an LSTM on the entire board history. However, this model was empirically weaker while requiring significantly more compute/parameters to run, so we abandoned this approach.

### Training
We implemented the model from scratch using Pytorch. We train with epochs of size 51,200 datapoints, and we trained our final model for about 50 epochs. We train off the mean squared error of the Elo predictions as our loss. To normalize the prediction scores, we first divide by 300 before computing the loss. For our hyperparameter search, we used a pseudo-random human-in-the-loop strategy, though due to compute limitations, we didn’t search very exhaustively.

## Evaluation
For evaluation, we used the mean absolute error (MAE) of the predicted Elo ratings on the test data. In English, this translates to: “how many Elo points off from the actual Elo was the model?” The final train MAE of the model was 219 and the final test MAE was 243. 

Charts and data from our final training run is [here](https://wandb.ai/uw-d0/deep-learning-final-project/reports/Final-Model-Run--VmlldzozMTI3MzU5?accessToken=0e7cgb7v9xzydixsfjoqmw8e2hqw96a92jxx6nlklgdnetzqn046pu2ibzo8elc9).

We also tried inputting some custom games into our model and it seemed to predict the scores reasonably alright! We played a game and the model predicted a rating of 1328, which seemed reasonable. 

```
[Event "Casual Blitz game"]
[Site "https://lichess.org/KetofGqB"]
[Date "2022.12.13"]
[White "Anonymous"]
[Black "Anonymous"]
[Result "0-1"]
[UTCDate "2022.12.13"]
[UTCTime "01:08:43"]
[WhiteElo "?"]
[BlackElo "?"]
[Variant "Standard"]
[TimeControl "300+3"]
[ECO "B06"]
[Opening "Modern Defense"]
[Termination "Normal"]
[Annotator "lichess.org"]

1. e4 g6 { B06 Modern Defense } 2. d3 Bg7 3. Nf3 d6 4. Bg5 Nf6 5. Nc3 O-O 6. Be2 Nc6 7. O-O Bd7 8. Nd5 Nxd5 9. exd5 Nb4 10. c4 b6 11. Rb1 Na6 12. b4 c6 13. b5 cxb5 14. cxb5 Nc7 15. a4 Nxd5 16. d4 Nc3 17. Qb3 Nxb1 18. Bd3 Nd2 19. Nxd2 Bxd4 20. Re1 Bf6 21. Bxf6 exf6 22. Nc4 Be6 23. Qc3 Bxc4 24. Qxc4 d5 25. Qh4 Kg7 26. g3 Re8 27. Rd1 Rc8 28. f3 Rc3 29. Be4 d4 30. Bc6 Ree3 31. Qxd4 Qxd4 32. Rxd4 Rxf3 33. Bxf3 Rxf3 34. Kg2 Ra3 35. h4 h5 
```

## Results
Overall, it seems our model does pretty well. The MAE on test data was 243 which is significantly better than random guessing, which is closer to 400. Also, chess professionals GothamChess and Hikaru had an MAE of 233 and 443 respectively, so our model does comparable to or better than chess professionals.

One major limitation of our model was that we didn’t have enough datapoints at the tail of the distribution, which makes our model much worse at predicting really bad players or really good players. Unfortunately, this is partly due to our compute limits, since the parsing speed of PGNs was quite slow, we couldn’t parse enough datapoints to get the extreme cases since the dataset isn’t nicely grouped by Elo ratings. So even though the data technically existed, we didn’t have enough compute to actually get it all in the format required for the model.

Another limitation of the model was that it seems like it didn’t fully converge on the training data. We didn’t train it for longer due to the compute limits of using colab. However empirically, the test MAE was going down much slower, so training for even longer would likely have negligible benefits.
