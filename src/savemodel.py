import pandas as pd
import os
  
# checking if the directory demo_folder 
# exist or not.
if not os.path.exists("/home/ekabanga/Plot_CB/saved_models"):

    # if the demo_folder directory is not present 
    # then create it.
    os.makedirs("/home/ekabanga/Plot_CB/saved_models")

def save_model(model, history, model_name):

    #  Saving the model weight
    json_model = model.to_json()

    with open('saved_models/'+model_name+'.json', 'w') as json_file:
        json_file.write(json_model)

    model.save('saved_models/'+model_name+'.h5')

    #  Convert history to pandas Dataframe
    hist_df = pd.DataFrame(history.history)

    #  Save to csv:
    hist_csv_file = 'saved_models/'+model_name+'_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)