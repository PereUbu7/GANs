# Playing around with GANs
## trying to make them model arrival distributions conditioned on facts

### Situation to model:
We have measured two vehicles moving from A (x = 0) to B (x = 1).
One was moving with constant speed. The other moved with constant speed but made a short stop in the middle. The data is prepared to match run-time that's left (y) to arrive at B (x = 1) for each x. Thus, we get the following data:

| x | y1 | y2 |
|---|----|----|
| 0 | 10 | 13 |
|0.1| 9  | 12 |
|0.2| 8  | 11 |
|0.3| 7  | 10 |
|0.4| 6  | 9  |
|0.5| 5  | 8  |
|0.6| 4  | 4  |
|0.7| 3  | 3  |
|0.8| 2  | 2  |
|0.9| 1  | 1  |
| 1 | 0  | 0  |

We can see that vehicle 2 made a stop at x = 0.5

### Try 1 (GAN_test1.py)
Using the standard loss (minimizing Jensen-Shannon divergence) for the discriminator and a min/max loss (minimizing D's rights and maximizing D's wrongs) for the generator