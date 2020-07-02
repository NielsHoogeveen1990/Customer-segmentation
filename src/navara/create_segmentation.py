import joblib
from navara.preprocessing import get_df, get_original_df


def merge_clusters(df, clusters_array):
    """
    This function assigns clusters to a dataframe.
    :param df: dataframe
    :param clusters_array: numpy array of clusters
    :return: dataframe
    """
    return df.assign(
        cluster = lambda d: clusters_array
    )


def create_segmentations(input_path, output_path):
    """
    This function loads the data from the preprocessing datapipeline and
    loads the trained model, in order to create clusters and assign those clusters to the original dataframe.
    :param input_path: data path of the input data
    :param output_path: output path for the clustered data
    :return: CSV file with a dataframe
    """
    model = joblib.load(open('trained_models/model_1.joblib', 'rb'))

    X = get_df(input_path)
    df = get_original_df(input_path)

    y_clusters = model.predict(X)

    df = merge_clusters(df, y_clusters)

    df.to_csv(f'{output_path}/segmentations.csv', encoding='utf-8', index=False)

