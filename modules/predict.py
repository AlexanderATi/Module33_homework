import os
import pandas as pd
import dill
import datetime
import json


path = os.environ.get('PROJECT_PATH', '.')
def predict():
    models_list = os.listdir(f'{path}/data/models/')
    with open(f'{path}/data/models/{models_list[-1]}', 'rb') as file:
        model = dill.load(file)

    df_pred = pd.DataFrame(columns=['car_id', 'price_predict'])
    files_list = os.listdir(f'{path}/data/test/')

    for filename in files_list:
        with open(f'{path}/data/test/{filename}') as file:
            form = json.load(file)
        data = pd.DataFrame.from_dict([form])
        prediction = model.predict(data)
        dict_pred = {'car_id': data.id, 'price_predict': prediction}
        df = pd.DataFrame(dict_pred)
        df_pred = pd.concat([df_pred, df], axis=0)

    df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv',
                   index=False)
    pass


if __name__ == '__main__':
    predict()
