from utils.model import perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import numpy as np
import pandas as pd
import logging
import os

logging_str = '[%(asctime)s: %(levelname)s: %(module)s] %(message)s'
log_dir = 'Logs'
os.makedirs(log_dir, exist_ok= True)
logging.basicConfig(filename= os.path.join(log_dir, 'Running_logs.log'),level=logging.INFO, format=logging_str, filemode='a')

def main(data, ETA, Epochs, modelpath, plotname):
       df = pd.DataFrame(data)
       logging.info(f'This is actual dataframe: {df}')
       X, y = prepare_data(df)
       model = perceptron(eta=ETA, epochs=Epochs)
       model.fit(X, y)

       _ = model.totalloss()

       save_model(model, filename=modelpath)
       save_plot(df, plotname, model)

if __name__ == '__main__':
       AND = {'x1': [0, 0, 1, 1],
       'x2': [0, 1, 0, 1],
       'y': [0, 0, 0, 1]}
       ETA = 0.3
       Epochs = 10
       try:
              logging.info('>>>>>> Starting Training >>>>>>>')
              main(data=AND, ETA = ETA, Epochs = Epochs, modelpath= 'and.model', plotname='and.png')
              logging.info('>>>>>> Training of model is finishes >>>>>>>')
       except Exception as e:
              logging.exception(e)
              raise e




