from mage_ai.io.file import FileIO
from pandas import DataFrame
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_file(df: DataFrame, **kwargs) -> None:
    """
    Template for exporting data to filesystem.

    Docs: https://docs.mage.ai/design/data-loading#fileio
    """
    filepath = 'export_data/test_titanic.csv'
    if not os.path.exists('export_data'):
        os.mkdir('export_data')

    FileIO().export(df, filepath)
