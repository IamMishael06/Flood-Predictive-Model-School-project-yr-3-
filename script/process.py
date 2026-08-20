import pandas as pd
import numpy as np

def preprocessing(df):
    if df.isna().any(axis=None):
        # Fill NaN values with 0
        df["QGL"] = df["QGL"].fillna(0)
        df["P-JUS-CKGL"] = df["P-JUS-CKGL"].fillna(0)
    
        # Drop NaN rows from P-MON-CKP and T-JUS-CKP
        df = df.dropna(subset=["P-MON-CKP", "T-JUS-CKP"])
        
        if df["T-TPT"].isna().any(axis=None):
            df["T-TPT"] = df["T-TPT"].ffill() 

        df = df.dropna(ignore_index=True)

    else:
        print(f"No Missing data")
        return df

    # Fill missing values of P-JUS-CKGL & QGL with	0
    df["gas_lift_active"] = df["QGL"].notna().astype(int)

    # Remove T-JUS-CKGL 
    df = df.drop(columns=["T-JUS-CKGL","Unnamed: 0"])

    # Convert to bar
    bar_equivalent = 0.00001   # standard unit
    pressure_cols = ['P-PDG', 'P-TPT', 'P-MON-CKP', 'P-JUS-CKGL']
    df[pressure_cols] = df[pressure_cols] * bar_equivalent

    # Rearrange columns so class can be the last column to the right
    df = df[["timestamp", "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "P-JUS-CKGL", "QGL", "gas_lift_active", "class"]]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.drop(columns=["month"])

    # Extract datetime information and drop column in preparation for fitting model
    df = df.drop(columns="timestamp")

    for col in ['P-PDG','P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'QGL']:
        df[f"{col}_mean"] = df[col].rolling(window=10, min_periods=1).mean()  # Mean column 10minutes period
        df[f"{col}_mean"] = df[col].rolling(window=120, min_periods=1).mean().bfill()  # Mean column 2hr period
        df[f"{col}_var"] = df[col].rolling(window=10, min_periods=1).var().fillna(0)  # Varience column
        df[f"{col}_std"] = df[col].rolling(window=10, min_periods=1).std().fillna(0)  # Standard derivation 10 minute period
        df[f"{col}_std"] = df[col].rolling(window=120, min_periods=1).std().bfill()  # Standard derivation 2 hr period
        df[f"{col}_range"] = df[col].rolling(window=10, min_periods=1).apply(lambda x : max(x) - min(x))  # Range column

    for col in ['P-PDG','P-TPT', 'P-MON-CKP', 'P-JUS-CKGL', 'QGL']:
        df[f"{col}_ewm"] = df[col].ewm(span=10).mean()  # EWM for pressure columns

    
    df["P-PDG_P-TPT_ratio"] = (df["P-PDG"] / (df["P-TPT"] + 1)).clip(0,1)
    
    df["T_P_divergence"] = df["T-TPT"] / (df["P-TPT"] + 1)
    
    df['choke_pressure_diff'] = df['P-MON-CKP'] - df['P-TPT']
    
    df['P_TPT_diff'] = df['P-TPT'].diff().bfill()

    df['P_TPT_sign_changes'] = df['P_TPT_diff'].rolling(10).apply(
        lambda x: ((x[:-1] * x[1:]) < 0).sum()
    ).bfill()
    
    df['productivity_index'] = df['P-TPT'] / (df['QGL'] + 1)
    
    df['P_MON_CKP_slope_60'] = df['P-MON-CKP'].rolling(60, min_periods=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    ).bfill()
    
    df['hydrate_risk'] = df['T-JUS-CKP'] - 20  # distance from hydrate threshold

    # 60  minute rolling window.
    df['P_TPT_std_60'] = df['P-TPT'].rolling(60, min_periods=1).std().bfill()
    df['T_TPT_std_60'] = df['T-TPT'].rolling(60, min_periods=1).std().bfill()
    
    # Creating stable Temp and Pressure column
    df['all_stable'] = (
        (df['P-TPT'].between(80, 140)) &
        (df['T-TPT'].between(50, 130)) &
        (df['P-MON-CKP'].between(10, 50))
    ).astype(int)
    
    # Stability streak finding the stabe
    df['stability_streak'] = df['all_stable'].groupby( 
        (df['all_stable'] != df['all_stable'].shift()).cumsum()
    ).cumcount()
            
    
    if df.isna().any(axis=None):
        print(f"Missing values still detected")
        return df
    else:
        print(f"Dataframe preprocessed successfully")
        return df
