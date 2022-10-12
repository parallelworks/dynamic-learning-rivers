# dynamic-learning-rivers/examples

This directory contains the following:
1. `vis_superlearner_tutorial.ipynb`: Jupyter notebook that uses an already trained SuperLearner model to make predictions.
2. `60357`: a directory containing all the input files and output for a single SuperLearner model training.  The model itself is stored in `60357`/model_dir/SuperLearner.pkl`. The notebook is preconfigured to load this SuperLearner model. The training and testing data sets for this model are also in this directory.
3. `WH_RA_GL_global_predict_25_inputs.csv`: a list of all the features (i.e. input variables) at many different sites for making predictions.
4. `WH_RA_GL_global_predict_ixy.csv`: a list fo the ID, longitude, and latitude corresponding to each site (these data are NOT inputs to the ML model).
