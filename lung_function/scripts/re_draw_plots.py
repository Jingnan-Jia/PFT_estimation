


from lung_function.modules.compute_metrics import icc, metrics





for mode in ['valid', 'test']:
    

    parent_dir = '/home/jjia/data/lung_function/lung_function/scripts/results/experiments/2750'
    label_fpath = parent_dir + f'/{mode}_label.csv'
    pred_fpath = parent_dir + f'/{mode}_pred.csv'
    
 
    # add r
    r_p_value = metrics(pred_fpath, label_fpath, ignore_1st_column=True)
    r_p_value_ensemble = {'ensemble_' + k:v  for k, v in r_p_value.items()}  # update keys




