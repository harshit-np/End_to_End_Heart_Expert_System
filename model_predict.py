import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')


features = ['age', 'sex', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca',
            'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal',
            'cp_typical angina', 'fbs', 'restecg_lv hypertrophy',
            'restecg_normal', 'restecg_st-t abnormality', 'exang',
            'slope_downsloping', 'slope_flat', 'slope_upsloping',
            'thal_fixed_defect', 'thal_normal', 'thal_reversable_defect']


def preprocessing(df_tmp, feat_cols):
    df_tmp['sex'] = (df_tmp['sex'] == 'Male') * 1
    df_tmp['exang'] = df_tmp['exang'].apply(lambda x: 1 if x == 'True' else 0)
    df_tmp['fbs'] = df_tmp['fbs'].apply(lambda x: 1 if x == 'True' else 0)

    for col in ['cp', 'restecg', 'slope', 'thal']:
        col_val = df_tmp[col].iloc[0]
        df_tmp = df_tmp.rename(columns={col: col_val})
        df_tmp[col_val] = 1

    for col in feat_cols:
        if col not in df_tmp.columns:
            df_tmp[col] = 0

    scaler = joblib.load('C:\\Users\\bansaln\\PycharmProjects\\HeartDisease\\scaler_hd.pkl')
    df_tmp_std = scaler.transform(df_tmp[feat_cols])

    return df_tmp_std


def make_predictions(valid_data):
    valid_df = preprocessing(pd.DataFrame(valid_data, index=[0]), features)
    print(valid_df)
    model = joblib.load("C:\\Users\\bansaln\\PycharmProjects\\HeartDisease\\logreg_hd.pkl")

    print(f"Heart Disease Prediction : {model.predict(valid_df)[0]}")
    print(f"Heart Disease Prediction Probability : {int(model.predict_proba(valid_df)[0][1]*100)}%")

    predict = model.predict(valid_df)[0]
    predict_proba = int(model.predict_proba(valid_df)[0][1] * 100)
    if (predict==1):
        predict_str='YES'
    else:
        predict_str='NO'
    return predict_str, predict_proba


