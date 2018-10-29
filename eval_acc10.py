import pandas as pd
import keras
from keras.models import load_model
import numpy as np
from tqdm import tqdm

def generate_predictions(df_path, model_path, batch_size=800, testing=False, n_channels=4, test_limit=5):
    df = pd.read_csv(df_path)
    model = load_model(model_path)

    predictions = {}

    for lig_id, grp in tqdm(df.groupby('lig_id')):
        grp.sort_values('pro_id', inplace=True)
        grp.reset_index(inplace=True)

        dims = (24,24,24)
        X = np.empty((len(grp), *dims, n_channels))
        for row in grp.itertuples():
            X[row[0],] = np.load(row.dest)

        probs = model.predict(X, batch_size=batch_size)
        probs = probs.flatten()

        predictions[lig_id] = probs

        if testing:
            if test_limit == 0:
                break
            test_limit-=1

    return predictions

def _test():
    # for testing the prediction generator

    df_path = './data/csv/test_acc10_300.csv'
    model_path = './models/try_epochs_16.h5'

    predictions = generate_predictions(df_path, model_path, 200, testing=True, n_channels=2)

    score = 0
    for lig_id, probs in predictions.items():
        relative_lig_id = lig_id - 2701
        largest_first = list(reversed(np.argsort(probs).tolist()))
        top10 = largest_first[:10]
        if relative_lig_id in top10:
            score += 1
    
    print(score/len(predictions))

if __name__ == "__main__":
    # _test()
    df_path = './data/csv/eval_acc10_2.csv'
    model_path = './models/.h5'