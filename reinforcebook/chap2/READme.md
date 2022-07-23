#reinforce book

This is Chap 2-example 1.

It's in a range of 5X7.

In every condition, you can go "up", "down", "right", and "left", and all the probability is the same.

There are two requirement:

1.When your next step is to go out the range of 5X7, then your condition won't change.

2.When you are at the place (2,7), whatever next action you choose, you will go to the place (2,1) and get the reward=40.

And the code is try to find the State-Value function.

Theta=0.01, and when delta < theta ,the while loop end.

S_prime is a 4X2 list, and it's used to get the condition after the action is taken.

R is a 4X1 list, and it's uedto get the reward after the action is taken.

In the while loop, we use two for loop to go throught the whole condition in the range of 5X7.

Use a for loop to get all the condition and reward after all the action has been taken and we'll get V(i,j).

Compare V(i,j) and v to delta, and choose the maximum one.

If the delta is larger then theta, go throught the while loop again, and if not, end while loop.
