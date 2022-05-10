from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd
import logging 
import os

logging.basicConfig(
    filename=os.path.join('logs','runing_logs.log'),
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode='a'
    )

def main(data, modelName, plotName, eta, epochs):

    df_OR = pd.DataFrame(data)

    X, y = prepare_data(df_OR)

    logging.info(f"This is the raw dataset: {df_OR}")


    model_or = Perceptron(eta=ETA, epochs=EPOCHS)
    model_or.fit(X, y)
    _ = model_or.total_loss()
    #3 Saving themodel
    model_or.save(filename=modelName, model_dir="model_or")

    ## SAving the plot
    save_plot(df_OR, model_or, filename=plotName)

if __name__=="__main__":
    gate='OR'
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,1,1,1]
    }
    ETA = 0.1 # 0 and 1
    EPOCHS = 10
    try:
        logging.info(f">>>>>>>Starting Training for {gate}>>>>>>>")
        main(data=OR, modelName='or.model', plotName='or.png', eta=ETA, epochs= EPOCHS)
        logging.info(">>>>>>Done Training>>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
