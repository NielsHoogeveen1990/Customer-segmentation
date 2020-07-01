from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from category_encoders import HashingEncoder

from navara.transformers.dtype_selector import DTypeSelector
from navara.transformers.correlation_filter import CorrFilterHighTotalCorrelation


def pipeline(number_clusters=8):

    numerical_pipeline = make_pipeline(
        DTypeSelector('number'),
        CorrFilterHighTotalCorrelation(),
        KNNImputer(n_neighbors=5),
        StandardScaler()
    )

    object_pipeline = make_pipeline(
        DTypeSelector('object'),
        SimpleImputer(strategy='most_frequent'),
        HashingEncoder(n_components=48)
    )

    return make_pipeline(
        make_union(
            numerical_pipeline,
            object_pipeline,
        ),
        KMeans(n_clusters=number_clusters,
               init='k-means++',
               n_init=10,
               max_iter=300,
               random_state=0)
    )