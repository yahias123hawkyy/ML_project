import pandas as pd
import explore
import preprocess
import eda
import model_training
import feature_engineering
import tune_model
import model_interpret



################################################################## Pre-Explore ##########################################################################

# Load the PRE dataset
# preDataset = pd.read_csv('featured_data.csv')
# explore.exploreTheDataSet(preDataset)

################################################################# Preprocessing ########################################################################
# 
# preprocess.preprocessTheDataSet(preDataset)

################################################################## EDA INTENSIVE ########################################################################
cleandeDataSet = pd.read_csv('cleaned_data.csv')

eda.exploreInDepthDataSet(cleandeDataSet)

############################################################ Feature Engineering ########################################################################

# feature_engineering.featureEngineeringDataSet()

############################################################ Model Training #############################################################################

# model_training.trainTheModel()

# ############################################################ Model Tuning ###############################################################################

# best_random_forest,X_train,X_test= tune_model.tuneTheModel()

# ############################################################## Model Interpret ##########################################################################

# model_interpret.interpretTheModel(best_random_forest,X_train,X_test)