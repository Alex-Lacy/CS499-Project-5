Our Neural Network algorithm can be found here: [NNet Optimization](https://github.com/Alex-Lacy/CS499-Project-4/blob/master/Project4.py)

This time, our algorithm and driver code are packaged into the same file.  Additionally, we rely on libraries instead of our own from-scratch NN algorithm.

Our code is tested to only work on single sets of binary classification data.  For our results, we used [Stanford's Spam Data](https://web.stanford.edu/~hastie/ElemStatLearn/data.html) for our data. 

In order to use our code effecitvely, please be sure you have the data file in the same directory as the driver function, and be sure to import it in the code.

Our code automatically finds the optimal number of epochs.  After running it to find out what it is, you will need to adjust the code slightly to run it just using that number of epochs.

You can also freely adjust the number of hidden units you wish to use.  As-is, our code is designed to run with 10, 100, and 1000 hidden units.  If you are just running one set, then you will need to comment out the code pertaining to the others.
