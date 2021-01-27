from pandas import DataFrame


def assert_no_nan(df: DataFrame):
    assert df.isna().any().any() == False, Exception("NaN present in dataframe")


def assert_valid_datetime_index(df: DataFrame):
    assert df.index.freqstr is not None, Exception("Dataframe index cannot be None")
