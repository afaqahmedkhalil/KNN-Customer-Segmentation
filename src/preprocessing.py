import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """Encode Gender + drop CustomerID."""
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df.drop('CustomerID', axis=1, inplace=True)
    return df


def create_segments(df):
    """Create segmentation labels based on Spending Score."""
    df = df.copy()
    df['Segment'] = pd.cut(
        df['Spending Score (1-100)'],
        bins=[0, 33, 66, 100],
        labels=['Low', 'Medium', 'High']
    )
    return df


def split_xy(df):
    """Split features and labels."""
    X = df.drop('Segment', axis=1)
    y = df['Segment']
    return X, y


def scale_features(X):
    """Scale all numeric features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
