import pickle
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd

def RMSE(y_test, y_pred):
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    return np.sqrt(mse)

test_num = 5
performance = []
for num in range(test_num):
    from sklearn.model_selection import train_test_split

    data = pickle.load(open('./random_split_0-2048-3-true.pkl', "rb"))
    X = data[0]
    y = data[1]
    # 将数据集划分为训练集+验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=num)
    # 将训练集+验证集进一步划分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=num)
    # Vanilla hyp§erparams
    model = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=12,
        min_child_weight=6,
        colsample_bytree=0.6,
        subsample=0.8,
        random_state=42,
        early_stopping_rounds=10,
    )

    # Train the model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Inference
    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
    y_pred[y_pred < 0.0] = 0.0

    # Get the metrics
    y_pred_train = model.predict(X_train, ntree_limit=model.best_ntree_limit)
    train_r2 = r2_score(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = RMSE(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = RMSE(y_test, y_pred)
    performance.append([train_r2, train_rmse, train_mae, test_r2, test_rmse, test_mae])

perform_df = pd.DataFrame(performance, columns=["train_r2", "train_rmse", "train_mae", "test_r2", "test_rmse", "test_mae"])
print(perform_df)
perform_df.to_excel("./Suzuki_XGBoost.xlsx")