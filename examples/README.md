# dynamic-learning-rivers/examples

This directory contains the following:
1. `vis_superlearner_tutorial.ipynb`: Jupyter notebook that uses an already trained SuperLearner model to make predictions.
2. `60357`: a directory containing all the input files and output for a single SuperLearner model training.  The model itself is stored in `60357`/model_dir/SuperLearner.pkl`. The notebook is preconfigured to load this SuperLearner model. The training and testing data sets for this model are also in this directory.
3. `WH_RA_GL_global_predict_25_inputs.csv`: a list of all the features (i.e. input variables) at many different sites for making predictions.
4. `WH_RA_GL_global_predict_ixy.csv`: a list fo the ID, longitude, and latitude corresponding to each site (these data are NOT inputs to the ML model).

Then, there are two additional files that were helpful in
setting the order of how sites were added to the ML model training:
5. `sl_pca_training.csv`: the same as the main train/test input file except with `pca.dist` column added as computed in pca.py. This functionality was commented out in PCA.py since we don't need this with every iteration.
6. `generate_modex_train_test.py`: Python script that uses the `pca.dist` metric in `sl_pca_training.csv` to generate a series of files used in training several iterative cycles of the ModEx loop. The successive files (<r|o>_<1-5>.csv), are retained here. Each file is manually inserted as `../input_data/whondrs_25_inputs_train.csv` with each iteration of the loop adding more and more data.  The order of the files goes by their numbers (1-5) with `r_` prefixed files indicating a randomized addition of data points while `o_` indicates points are added from high to low pca.dist.
