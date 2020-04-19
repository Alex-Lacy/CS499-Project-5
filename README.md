Our Neural Network algorithm can be found here: [NNet Param Optimization](https://github.com/Alex-Lacy/CS499-Project-5/blob/master/Project5.py)

This time, our algorithm and driver code are packaged into the same file.  Additionally, we rely on libraries instead of our own from-scratch NN algorithm.

Additionally, this time our code tests different parameters for optimization purposes.

Our code is tested to only work on single sets of binary classification data.  For our results, we used [Stanford's Spam Data](https://web.stanford.edu/~hastie/ElemStatLearn/data.html) for our data. 

In order to use our code effecitvely, please be sure you have the data file in the same directory as the driver function, and be sure to import it in the code.

Our code automatically finds the optimal number of hidden layers and hidden units.  This time, our code only needs run once.  It will automatically find the best of each of these parameters, train on them, then give the accuracy result for the best of each.

NOTE: there is a segment of our code that needs un-commented the first time you run it in order to work properly.

To adjust our code for your own reasons might be a little involed.  If you wish to optimize for different units, most of the code will remain the same, but you will need to perform the setup differently.  If you wish to add more/different numbers for the number of hidden layers or units tested, you will have to add sections for those.  That should be relatively simple, however, because you can just copy existing segments.

Good luck!
