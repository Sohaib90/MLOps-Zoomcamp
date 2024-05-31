from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)

    records_pu = df[['PULocationID']].to_dict(orient='records')
    records_do = df[['DOLocationID']].to_dict(orient='records')

    vectorizer_pu = DictVectorizer(sparse=True)
    vectorizer_do = DictVectorizer(sparse=True)
    
    X_pu = vectorizer_pu.fit_transform(records_pu)
    X_do = vectorizer_do.fit_transform(records_do)
        
    X = hstack([X_pu, X_do])
    y = df

    model = LinearRegression()
    model.fit(X, df.duration.values)  

    return [vectorizer_do, vectorizer_pu, model]


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
