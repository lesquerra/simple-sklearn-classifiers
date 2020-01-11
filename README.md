# simple-sklearn-classifiers
This repository contains a <b>simple collection of classifiers applied using the sklearn package</b>

## Data Preparation

Data is split into a <b>training</b> (60%) and a <b>testing</b> (40%) dataset using stratified sampling and cross validation with k=5. These settings can be changed through the <code>model_selection.train_test_split()</code> function.

## Classifiers

There are 7 algorithms implemented testing multiple configurations using grid search:

- <b>SVM with Linear Kernel</b>. Configurations: C = [0.1, 0.5, 1, 5, 10, 50, 100].
- <b>SVM with Polynomial Kernel</b>. Configurations: C = [0.1, 1, 3] and gamma = [0.1, 0.5].
- <b>SVM with RBF Kernel</b>. Configurations: C = [0.1, 0.5, 1, 5, 10, 50, 100] and gamma = [0.1, 0.5, 1, 3, 6, 10].
- <b>Logistic Regression</b>. Configurations: C = [0.1, 0.5, 1, 5, 10, 50, 100].
- <b>k-Nearest Neighbors</b>. Configurations: n_neighbors = [1, 2, 3, ..., 50] and leaf_size = [5, 10, 15, ..., 60].
- <b>Decision Tree</b>. Configurations: max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].
- <b>Random Forest</b>. Configurations: max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].

Any configuration can be edited using the corresponding <code>parameters</code> variable. All <b>scores</b> are stacked and written into an output file.

## Running the algorithms

All algorithms can be applied to any dataset provided through command line. A single input dataset Xy is expected as an input with the response variable in the last column. To run the classifiers use the following command:

<code> $ python3 main.py Xy_train.csv output_filename.csv </code>
