import pandas as pd
import os
  
# checking if the directory demo_folder 
# exist or not.
if not os.path.exists("/home/ekabanga/Plot_CB/RDM_saved_models"):

    # if the demo_folder directory is not present 
    # then create it.
    os.makedirs("/home/ekabanga/Plot_CB/RDM_saved_models")

def save_model(model, history, model_name, data_category):

    if not os.path.exists(f"/home/ekabanga/Plot_CB/RDM_saved_models/{model_name}_zzz_{data_category}_don"):

        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(f"/home/ekabanga/Plot_CB/RDM_saved_models/{model_name}_zzz_{data_category}_don")
    
    #  Saving the model weight
    json_model = model.to_json()

    with open(f'RDM_saved_models/{model_name}_zzz_{data_category}_don/'+model_name+'.json', 'w') as json_file:
        json_file.write(json_model)

    model.save(f'RDM_saved_models/{model_name}_zzz_{data_category}_don/'+model_name+'.h5')

    #  Convert history to pandas Dataframe
    hist_df = pd.DataFrame(history.history)

    #  Save to csv:
    hist_csv_file = f'RDM_saved_models/{model_name}_zzz_{data_category}_don/'+model_name+'_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
