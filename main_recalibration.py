"""
Main script to run the recalibration experiments
"""
# Author: Alessandro Brusaferri
# License: Apache-2.0 license

# Modified by: Group 15 Financial Engineering

import os
import pandas as pd
import matplotlib.pyplot as plt
os.environ["TF_USE_LEGACY_KERAS"]="1"
from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs
from tools.prediction_quantiles_tools import plot_quantiles
from tools.conformal_prediction import compute_cp
from tools.adaptive_conformal_inference import compute_aci
from compute_losses import *
import random
#--------------------------------------------------------------------------------------------------------------------
def build_zero_delta_cov(test_predictions,quantiles, task_name):
    """
        Adjust the quantiles from point prediction to have zero delta coverage
    :return: the test_predictions with zero delta coverage
    """
    n = len(quantiles)
    m = len(test_predictions.loc[:, 0.5])
    for i,q in enumerate(quantiles[:n//2]):
        # Define the confidence level
        cl = quantiles[-i-1] - quantiles[i]
        # Adjust quantiles
        test_predictions.loc[:,quantiles[-i-1]] = test_predictions.loc[:,0.5] + 1e-15*cl
        test_predictions.loc[:,quantiles[i]] = test_predictions.loc[:, 0.5] - 1e-15*cl
        test_predictions.iloc[:int(cl*m)].loc[:,quantiles[-i-1]]= test_predictions.iloc[:int(cl*m)].loc[:,0.5] + 100*cl
        test_predictions.iloc[:int(cl*m)].loc[:,quantiles[i]]= test_predictions.iloc[:int(cl*m)].loc[:,0.5] - 100*cl

    return test_predictions.loc[:, [task_name]+quantiles]
#-----------------------------------------------------------------------------------------------------------------------
# Set seed for reproducibility
SEED = 420
random.seed(SEED)
np.random.seed(SEED)

#-----------------------------------------------------------------------------------------------------------------------
# Set PEPF task to execute
PF_task_name = 'EM_price'
# Set Model setup to execute
exper_setup = 'point-RDR'

#-----------------------------------------------------------------------------------------------------------------------
# Set run configs
run_id = 'recalib_opt_grid_1_2'
# Load hyperparams from file (select: load_tuned or optuna_tuner)
hyper_mode = 'load_tuned'
# Plot train history flag
plot_train_history = False
plot_weights = False
ensemble_mode = False # Ensemble mode for ACI with two models
#---------------------------------------------------------------------------------------------------------------------
# Load experiments configuration from json file
configs = load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)

# Load dataset
dir_path = os.getcwd()
ds = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', configs['data_config'].dataset_name))
ds.set_index(ds.columns[0], inplace=True)
ds["FUTU__hour_cos"]= np.cos(ds["Hour"]*2*np.pi/24) # in FUTU as is not constant in the day after
ds["FUTU__hour_sin"]= np.sin(ds["Hour"]*2*np.pi/24) # in FUTU as is not constant in the day after

#-----------------------------------------------------------------------------------------------------------------------
# Instantiate recalibratione engine
PrTSF_eng = PrTsfRecalibEngine(dataset=ds,
                               data_configs=configs['data_config'],
                               model_configs=configs['model_config'])

# Get model hyperparameters (previously saved or by tuning)
model_hyperparams = PrTSF_eng.get_model_hyperparams(method=hyper_mode, optuna_m=configs['model_config']['optuna_m'])

# Exec recalib loop over the test_set samples, using the tuned hyperparams
test_predictions = PrTSF_eng.run_recalibration(model_hyperparams=model_hyperparams,
                                               plot_history=plot_train_history,
                                               plot_weights=plot_weights)
if ensemble_mode:
    # Select the model to ensemble
    exper_setup = 'point-DNN'
    run_id = 'recalib_opt_grid_1_2'
    configs = load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)
    PrTSF_eng = PrTsfRecalibEngine(dataset=ds,
                                   data_configs=configs['data_config'],
                                   model_configs=configs['model_config'])

    # Get model hyperparameters (previously saved or by tuning)
    model_hyperparams = PrTSF_eng.get_model_hyperparams(method=hyper_mode, optuna_m=configs['model_config']['optuna_m'])

    # Exec recalib loop over the test_set samples, using the tuned hyperparams
    test_predictions1 = PrTSF_eng.run_recalibration(model_hyperparams=model_hyperparams,
                                                   plot_history=plot_train_history,
                                                   plot_weights=plot_weights)
    # Embed the two models
    test_predictions = (test_predictions+test_predictions1)/2


#--------------------------------------------------------------------------------------------------------------------
# Conformal prediction settings
exec_CP = False
exec_ACI = True
zero_delta_cov = False
# set the size of the calibration set sufficiently large to cover the target alpha (tails)
cp_settings={'target_alpha':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]}
num_cali_samples = 122
#cp_settings={'target_alpha':[0.10]}
#num_cali_samples = 31

if exec_CP:
    print('------------------------------------------------------------------------------')
    print(" CP tecnique selected for quantile prediction ")
    print('------------------------------------------------------------------------------')
    if exper_setup[:5]=='point':
        # build the settings to build PF from point using CP
        cp_settings['pred_horiz']=configs['data_config'].pred_horiz
        cp_settings['task_name']=configs['data_config'].task_name
        cp_settings['num_cali_samples']=num_cali_samples
        # exec conformal prediction
        test_predictions = compute_cp(test_predictions,cp_settings)
    else:
        print('conformal prediction implemented on point predictions')

elif exec_ACI:
    print('------------------------------------------------------------------------------')
    print(" ACI tecnique selected for quantile prediction ")
    print('------------------------------------------------------------------------------')
    if exper_setup[:5]=='point':
        # build the settings to build PF from point using CP
        cp_settings['pred_horiz']=configs['data_config'].pred_horiz
        cp_settings['task_name']=configs['data_config'].task_name
        cp_settings['num_cali_samples']=num_cali_samples
        # exec conformal prediction
        test_predictions = compute_aci(test_predictions,cp_settings,0.022)
    else:
        print('adaptive conformal inference implemented on point predictions')
elif zero_delta_cov:
    print('------------------------------------------------------------------------------')
    print(" Zero delta cov tecnique selected for quantile prediction ")
    print('------------------------------------------------------------------------------')
    test_predictions = build_zero_delta_cov(test_predictions,PrTSF_eng.__build_target_quantiles__(cp_settings['target_alpha']), PF_task_name)

#--------------------------------------------------------------------------------------------------------------------
# Plot test predictions
plot_quantiles(test_predictions, target=PF_task_name)
#--------------------------------------------------------------------------------------------------------------------
# Compute scores
quantiles_levels =  PrTSF_eng.__build_target_quantiles__(cp_settings['target_alpha'])
pred_steps = configs['model_config']['pred_horiz']


print(" Mean Absolute Error: ", np.mean(np.abs(test_predictions.loc[:,0.5] -test_predictions.loc[:,PF_task_name])))
print(" MAPE: ", np.mean(np.abs((test_predictions.loc[:,0.5] -test_predictions.loc[:,PF_task_name])/(test_predictions.loc[:,PF_task_name]))))
print(" RMSE: ", np.sqrt(np.mean((test_predictions.loc[:,0.5] -test_predictions.loc[:,PF_task_name])**2)))

# Compute pinball scores
pinball_scores = compute_pinball_scores(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)
# Compute winkler scores
winkler_scores = compute_winkler_scores(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)

# Compute delta_coverage
delta_coverage = compute_delta_coverage(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)

# Compute CRPS
CRPS = compute_CRPS(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)

print(" Average Pinball Score: ", np.mean(np.mean(pinball_scores)))
print(" Average Winkler Score: ", np.mean(np.mean(winkler_scores)))
print(" CRPS: ", CRPS)
print(" Delta Coverage: ", delta_coverage)


plt.plot(np.mean(pinball_scores, axis=1), marker='o')
plt.suptitle('Average Pinball score ' +exper_setup)
plt.xlabel('Hour')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(16, 4))

# Plot del punteggio medio Winkler
axs[0].plot(np.mean(winkler_scores, axis=1), marker='o')
axs[0].set_title('Average Winkler score')
axs[0].set_xlabel('Hour')

# Plot del punteggio Winkler al 99%
axs[1].plot(winkler_scores[:, 0], marker='s')
axs[1].set_title('Winkler score 99%')
axs[1].set_xlabel('Hour')

# Plot del punteggio Winkler al 95%
axs[2].plot(winkler_scores[:, 4], marker='^')
axs[2].set_title('Winkler score 90%')
axs[2].set_xlabel('Hour')

fig.suptitle('Winkler score ' + exper_setup)
fig.tight_layout()
fig.show()

#--------------------------------------------------------------------------------------------------------------------
print('Done!')