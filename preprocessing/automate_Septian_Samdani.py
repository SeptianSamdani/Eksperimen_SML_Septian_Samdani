import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[load] Shape: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # FIX: reset index
    print(f"[dedup] Removed {before - len(df)} duplicates")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna().reset_index(drop=True)  # FIX: reset index
    print(f"[missing] Removed {before - len(df)} rows with nulls")
    return df


def remove_outliers_iqr(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """
    FIX: IQR hanya pada class mayoritas (Normal=0),
    agar data minority (Minor Fault, Critical Fault) tidak ikut terbuang.
    """
    before = len(df)
    cols = [c for c in df.columns if c not in exclude_cols]
    target_col = exclude_cols[0]

    # Hitung batas IQR dari class Normal saja
    majority = df[df[target_col] == 0]
    bounds = {}
    for col in cols:
        Q1 = majority[col].quantile(0.25)
        Q3 = majority[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Filter hanya pada class Normal
    mask_normal = df[target_col] == 0
    for col in cols:
        low, high = bounds[col]
        mask_normal = mask_normal & (df[col] >= low) & (df[col] <= high)

    # Gabungkan: normal yang sudah bersih + semua minority tetap utuh
    df_clean = pd.concat([
        df[mask_normal],
        df[df[target_col] != 0]
    ]).reset_index(drop=True)

    print(f"[outlier] Removed {before - len(df_clean)} outlier rows (majority class only)")
    return df_clean

def scale_features(df: pd.DataFrame, target_col: str):
    X = df.drop(target_col, axis=1)
    y = df[target_col].reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y, scaler  # FIX: return scaler


def apply_smote(X_train, y_train, random_state=42):
    """NEW: Handle class imbalance dengan SMOTE."""
    print(f"[smote] Before: {dict(y_train.value_counts().sort_index())}")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[smote] After : {dict(pd.Series(y_res).value_counts().sort_index())}")
    return X_res, y_res


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_outputs(X_train, X_test, y_train, y_test, scaler, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(X_train, columns=X_train.columns if hasattr(X_train, 'columns') else None)
    train_df['Engine_Condition'] = y_train.values if hasattr(y_train, 'values') else y_train

    test_df = pd.DataFrame(X_test, columns=X_test.columns if hasattr(X_test, 'columns') else None)
    test_df['Engine_Condition'] = y_test.values

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    # FIX: Simpan scaler untuk inference
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"[save] Files saved to '{output_dir}' (train.csv, test.csv, scaler.pkl)")


def run_preprocessing(input_path: str, output_dir: str, target_col: str = 'Engine_Condition'):
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = remove_outliers_iqr(df, exclude_cols=[target_col])
    X_scaled, y, scaler = scale_features(df, target_col)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    X_train, y_train = apply_smote(X_train, y_train)  # NEW: SMOTE setelah split
    save_outputs(X_train, X_test, y_train, y_test, scaler, output_dir)
    print("[done] Preprocessing complete.")


if __name__ == '__main__':
    run_preprocessing(
        input_path='engine_fault_raw/engine_fault_detection_dataset.csv',
        output_dir='preprocessing/engine_fault_preprocessing'
    )