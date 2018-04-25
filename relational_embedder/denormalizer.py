import pandas as pd


def join_ab_on_key(a: pd.DataFrame, b: pd.DataFrame, a_key: str, b_key: str, suffix_str=None):
    # First make sure to remove empty/nan values from join columns
    a_valid_index = (a[a_key].dropna()).index
    b_valid_index = (b[b_key].dropna()).index
    a = a.iloc[a_valid_index]
    b = b.iloc[b_valid_index]

    # Normalize join columns
    a[a_key] = a[a_key].apply(lambda x: str(x).lower())
    b[b_key] = b[b_key].apply(lambda x: str(x).lower())

    joined = pd.merge(a, b, how='inner', left_on=a_key, right_on=b_key, sort=False, suffixes=('', suffix_str))

    return joined


if __name__ == "__main__":
    print("Denormalizer")
