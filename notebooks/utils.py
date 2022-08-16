import altair as alt

# A function that generates a histogram of filtered data.
def filtered_hist(field, label, filter):
    """Creates a layered chart of histograms.
    The first layer (light gray) contains the histogram of the full data, and the
    second contains the histogram of the filtered data.
    Args:
      field: the field for which to generate the histogram.
      label: String label of the histogram.
      filter: an alt.Selection object to be used to filter the data.
    """
    base = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X(field, bin=alt.Bin(maxbins=10), title=label),
            y="count()",
        )
        .properties(
            width=300,
        )
    )
    return alt.layer(
        base.transform_filter(filter),
        base.encode(color=alt.value("lightgray"), opacity=alt.value(0.7)),
    ).resolve_scale(y="independent")


def split_dataframe(df, holdout_fraction=0.1):
  """Splits a DataFrame into training and test sets.
  Args:
    df: a dataframe.
    holdout_fraction: fraction of dataframe rows to use in the test set.
  Returns:
    train: dataframe for training
    test: dataframe for testing
  """
  test = df.sample(frac=holdout_fraction, replace=False)
  train = df[~df.index.isin(test.index)]
  return train, test