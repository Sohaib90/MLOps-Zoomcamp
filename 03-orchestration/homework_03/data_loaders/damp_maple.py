from mage_ai.io.file import FileIO
import pandas as pd 

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_file(*args, **kwargs):
    """
    Template for loading data from filesystem.
    Load data from 1 file or multiple file directories.

    For multiple directories, use the following:
        FileIO().load(file_directories=['dir_1', 'dir_2'])

    Docs: https://docs.mage.ai/design/data-loading#fileio
    """
    filepath = '/home/src/mage_data/homework_03/yellow_tripdata_2023-03.parquet'

    return pd.read_parquet(filepath)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    print("Number of records in the nyc_yellow_trip: {}".format(len(output)))
    assert len(output.columns) 