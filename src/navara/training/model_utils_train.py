from navara.preprocessing import get_df
from navara.training.models import k_means
import joblib


def train(datapath, model_version):
    X = get_df(datapath)

    clst = k_means.pipeline(number_clusters=7)
    fitted_model = clst.fit(X)

    with open(f'trained_models/model_{model_version}.joblib', 'wb') as file:
        joblib.dump(fitted_model, file)