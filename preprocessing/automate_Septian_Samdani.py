import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[load] Shape: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[dedup] Removed {before - len(df)} duplicates")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    print(f"[missing] Removed {before - len(df)} rows with nulls")
    return df


def remove_outliers_iqr(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    before = len(df)
    cols = [c for c in df.columns if c not in exclude_cols]
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    print(f"[outlier] Removed {before - len(df)} outlier rows")
    return df


def scale_features(df: pd.DataFrame, target_col: str):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_outputs(X_train, X_test, y_train, y_test, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['Engine_Condition'] = y_train.values
    test_df = X_test.copy()
    test_df['Engine_Condition'] = y_test.values

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"[save] Files saved to {output_dir}")


def run_preprocessing(input_path: str, output_dir: str, target_col: str = 'Engine_Condition'):
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = remove_outliers_iqr(df, exclude_cols=[target_col])
    X_scaled, y = scale_features(df, target_col)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    save_outputs(X_train, X_test, y_train, y_test, output_dir)
    print("[done] Preprocessing complete.")


if __name__ == '__main__':
    run_preprocessing(
        input_path='engine_fault_raw/engine_fault_detection.csv',
        output_dir='preprocessing/engine_fault_preprocessing'
    )