import numpy as np # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

def get_stock_data(csv_file):
    """
    Đọc dữ liệu cổ phiếu từ file CSV
    """
    df = pd.read_csv(csv_file)
    # Chuẩn hóa tên cột: viết hoa chữ cái đầu để tương thích với code
    df.columns = [col.capitalize() for col in df.columns]
    # Ưu tiên nhận diện cột ngày tháng là 'Date', nếu không có thì thử 'Time'
    date_col = None
    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'Time' in df.columns:
        date_col = 'Time'
    else:
        print(f"File {csv_file} không có cột ngày tháng ('Date' hoặc 'Time'). Hãy kiểm tra lại định dạng file!")
        return None
    # Nếu dữ liệu là yyyy-mm-dd thì không cần dayfirst
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    return df

def prepare_data(df, sequence_length=10):
    """
    Chuẩn bị dữ liệu cho mô hình với các đặc trưng mới
    """
    df = df.copy()
    
    # Các đặc trưng kỹ thuật
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Thêm các đặc trưng kỹ thuật mới
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df['Bollinger_Bands'] = calculate_bollinger_bands(df['Close'])
    
    # Đặc trưng thời gian
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Quarter'] = df.index.quarter
    
    # Xóa các dòng có giá trị NaN
    df = df.dropna()
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                'MA5', 'MA20', 'MA50', 'Volume_Change', 'RSI', 'MACD', 
                'Bollinger_Bands']
    df[features] = scaler.fit_transform(df[features])
    
    # Tạo sequences cho RNN
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[features].iloc[i:(i + sequence_length)].values)
        y.append(df['Close'].iloc[i + sequence_length])
    
    return np.array(X), np.array(y), scaler

def calculate_rsi(prices, period=14):
    """Tính chỉ số RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    """Tính chỉ số MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def calculate_bollinger_bands(prices, window=20):
    """Tính Bollinger Bands"""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    return (prices - ma) / (2 * std)

def create_mlp_model(input_shape):
    """
    Tạo mô hình MLP cải tiến
    """
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu', 
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu',
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu',
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_rnn_model(input_shape):
    """
    Tạo mô hình RNN cải tiến: Kết hợp LSTM và GRU, tăng số units, thêm BatchNormalization, giảm learning rate, điều chỉnh Dropout.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
        BatchNormalization(),
        Dropout(0.4),
        GRU(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
        BatchNormalization(),
        Dropout(0.3),
        GRU(32, kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Huấn luyện và đánh giá các mô hình với tối ưu hóa
    """
    # Chia validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Reshape dữ liệu
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    # MLP
    mlp_model = create_mlp_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    mlp_history = mlp_model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # RNN
    rnn_model = create_rnn_model((X_train.shape[1], X_train.shape[2]))
    rnn_history = rnn_model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Decision Tree với GridSearch
    dt_params = {
        'max_depth': [2, 3, 4, 5, 6, 7],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8],
        'max_features': [None, 'sqrt']
    }
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_grid = GridSearchCV(dt_model, dt_params, cv=5, scoring='neg_mean_squared_error')
    dt_grid.fit(X_train_reshaped, y_train)
    dt_model = dt_grid.best_estimator_
    
    # Lasso với GridSearch
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0]}
    lasso_model = Lasso(max_iter=10000)
    lasso_grid = GridSearchCV(lasso_model, lasso_params, cv=5, scoring='neg_mean_squared_error')
    lasso_grid.fit(X_train_reshaped, y_train)
    lasso_model = lasso_grid.best_estimator_
    
    # Ridge với GridSearch
    ridge_params = {'alpha': [1, 10, 50, 100, 200, 500, 1000, 2000]}
    ridge_model = Ridge()
    ridge_grid = GridSearchCV(ridge_model, ridge_params, cv=5, scoring='neg_mean_squared_error')
    ridge_grid.fit(X_train_reshaped, y_train)
    ridge_model = ridge_grid.best_estimator_
    
    # kNN với GridSearch
    knn_params = {'n_neighbors': [5, 7, 10, 15, 20, 30, 50]}
    knn_model = KNeighborsRegressor()
    knn_grid = GridSearchCV(knn_model, knn_params, cv=5, scoring='neg_mean_squared_error')
    knn_grid.fit(X_train_reshaped, y_train)
    knn_model = knn_grid.best_estimator_
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_reshaped, y_train)
    
    # Đánh giá các mô hình
    models = {
        'MLP': (mlp_model, X_test, X_train),
        'RNN': (rnn_model, X_test, X_train),
        'Decision Tree': (dt_model, X_test_reshaped, X_train_reshaped),
        'Lasso': (lasso_model, X_test_reshaped, X_train_reshaped),
        'Ridge': (ridge_model, X_test_reshaped, X_train_reshaped),
        'kNN': (knn_model, X_test_reshaped, X_train_reshaped),
        'Linear Regression': (lr_model, X_test_reshaped, X_train_reshaped)
    }
    
    # In thông tin về tham số tối ưu của Decision Tree
    print("\nThông tin về Decision Tree:")
    print(f"Tham số tối ưu: {dt_grid.best_params_}")
    print(f"Điểm cross-validation tốt nhất: {-dt_grid.best_score_:.4f}")
    
    results = {}
    for name, (model, X_test_data, X_train_data) in models.items():
        # Dự đoán
        if name in ['MLP', 'RNN']:
            y_pred_train = model.predict(X_train_data).reshape(-1)
            y_pred_test = model.predict(X_test_data).reshape(-1)
        else:
            y_pred_train = model.predict(X_train_data)
            y_pred_test = model.predict(X_test_data)
        
        # Đảm bảo kích thước
        y_pred_test = y_pred_test[:len(y_test)]
        y_pred_train = y_pred_train[:len(y_train)]
        
        # Tính metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'MSE_train': mse_train,
            'MSE_test': mse_test,
            'R2_train': r2_train,
            'R2_test': r2_test,
            'Overfitting_Score': mse_test / mse_train,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train
        }
    
    return results, mlp_history, rnn_history, X_train_reshaped, y_train, dt_model, lasso_model, ridge_model, knn_model, lr_model

def plot_results(results, mlp_history, rnn_history, y_test, X_train_reshaped, y_train, trained_models=None):
    """
    Vẽ đồ thị kết quả chi tiết
    """
    # 1. So sánh MSE giữa train và test
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    mse_train = [results[model]['MSE_train'] for model in models]
    mse_test = [results[model]['MSE_test'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, mse_train, width, label='Train MSE')
    plt.bar(x + width/2, mse_test, width, label='Test MSE')
    plt.title('So sánh MSE giữa tập Train và Test')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.yscale('log')
    plt.ylabel('MSE (Mean Squared Error)')
    plt.tight_layout()
    # Hiển thị giá trị trên cột với định dạng thập phân đầy đủ
    for i, v in enumerate(mse_train):
        plt.text(x[i] - width/2, v, f'{v:.6f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(mse_test):
        plt.text(x[i] + width/2, v, f'{v:.6f}', ha='center', va='bottom', fontsize=8)
    plt.savefig('model_comparison.png')
    plt.close()

    # 1.1. Vẽ sơ đồ so sánh R^2 giữa train và test
    r2_train = [results[model]['R2_train'] for model in models]
    r2_test = [results[model]['R2_test'] for model in models]
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, r2_train, width, label='Train R²')
    plt.bar(x + width/2, r2_test, width, label='Test R²')
    plt.title('So sánh R² giữa tập Train và Test')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.ylabel('R² (Hệ số xác định)')
    plt.tight_layout()
    for i, v in enumerate(r2_train):
        plt.text(x[i] - width/2, v, f'{v:.6f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(r2_test):
        plt.text(x[i] + width/2, v, f'{v:.6f}', ha='center', va='bottom', fontsize=8)
    plt.savefig('model_comparison_r2.png')
    plt.close()

    # 1.2. Vẽ sơ đồ so sánh chỉ số overfitting (MSE_test / MSE_train)
    overfitting_scores = [results[model]['Overfitting_Score'] for model in models]
    plt.figure(figsize=(12, 6))
    plt.bar(x, overfitting_scores, width=0.5, color='orange', label='Overfitting Score')
    plt.title('So sánh chỉ số Overfitting (MSE_test / MSE_train)')
    plt.xticks(x, models, rotation=45)
    plt.ylabel('Chỉ số Overfitting')
    plt.tight_layout()
    for i, v in enumerate(overfitting_scores):
        plt.text(x[i], v, f'{v:.6f}', ha='center', va='bottom', fontsize=8)
    plt.savefig('model_comparison_overfitting.png')
    plt.close()
    
    # 2. Vẽ quá trình huấn luyện của MLP
    plt.figure(figsize=(10, 6))
    plt.plot(mlp_history.history['loss'], label='Training Loss')
    plt.plot(mlp_history.history['val_loss'], label='Validation Loss')
    plt.title('MLP Training Process')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('mlp_training.png')
    plt.close()
    
    # 3. Vẽ quá trình huấn luyện của RNN
    plt.figure(figsize=(10, 6))
    plt.plot(rnn_history.history['loss'], label='Training Loss')
    plt.plot(rnn_history.history['val_loss'], label='Validation Loss')
    plt.title('RNN Training Process')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('rnn_training.png')
    plt.close()

    # 3.1. Vẽ learning curve cho các thuật toán truyền thống
    traditional_models = ['Decision Tree', 'Lasso', 'Ridge', 'kNN', 'Linear Regression']
    for model_name in traditional_models:
        if model_name not in results:
            continue
        # Sử dụng lại mô hình đã huấn luyện nếu truyền vào
        model = None
        if trained_models and model_name in trained_models:
            model = trained_models[model_name]
        if model is None:
            continue
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train_reshaped, y_train, cv=5, scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
        )
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Train MSE')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation MSE')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2, color='r')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2, color='g')
        plt.title(f'Learning Curve - {model_name}')
        plt.xlabel('Số lượng mẫu huấn luyện')
        plt.ylabel('MSE')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{model_name}_learning_curve.png')
        plt.close()
    
    # 4. Vẽ dự đoán so với giá thật cho mỗi mô hình
    for model_name in models:
        plt.figure(figsize=(12, 6))
        plt.plot(results[model_name]['y_pred_test'], label='Dự đoán')
        plt.plot(y_test, label='Giá thật')
        plt.title(f'So sánh dự đoán và giá thật - {model_name}')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model_name}_predictions.png')
        plt.close()
    
    # In báo cáo chi tiết
    print("\nBáo cáo chi tiết các mô hình:")
    print("-" * 50)
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"MSE Train: {metrics['MSE_train']:.4f}")
        print(f"MSE Test: {metrics['MSE_test']:.4f}")
        print(f"R2 Train: {metrics['R2_train']:.4f}")
        print(f"R2 Test: {metrics['R2_test']:.4f}")
        print(f"Chỉ số Overfitting (MSE_test/MSE_train): {metrics['Overfitting_Score']:.4f}")
        if metrics['Overfitting_Score'] > 1.5:
            print("Cảnh báo: Có dấu hiệu overfitting!")
        print("-" * 30)

    # Kết luận tổng quan về mô hình tốt nhất
    # Tiêu chí: MSE Test thấp nhất và R2 Test cao nhất
    best_mse_model = min(results.items(), key=lambda x: x[1]['MSE_test'])[0]
    best_r2_model = max(results.items(), key=lambda x: x[1]['R2_test'])[0]
    print("\nKẾT LUẬN TỔNG QUAN:")
    print("=" * 50)
    print(f"Mô hình có MSE Test thấp nhất: {best_mse_model} (MSE Test = {results[best_mse_model]['MSE_test']:.6f})")
    print(f"Mô hình có R² Test cao nhất: {best_r2_model} (R² Test = {results[best_r2_model]['R2_test']:.6f})")
    if best_mse_model == best_r2_model:
        print(f"=> {best_mse_model} là mô hình tổng thể tốt nhất trên tập kiểm tra!")
    else:
        print(f"=> {best_mse_model} tối ưu về MSE, {best_r2_model} tối ưu về R². Hãy cân nhắc mục tiêu sử dụng để chọn mô hình phù hợp.")
    print("=" * 50)

    # Bảng xếp hạng các thuật toán
    print("\nBẢNG XẾP HẠNG CÁC THUẬT TOÁN:")
    print("-" * 50)
    # Ranking theo MSE Test (tăng dần)
    print("Xếp hạng theo MSE Test (từ thấp đến cao):")
    mse_sorted = sorted(results.items(), key=lambda x: x[1]['MSE_test'])
    for idx, (model, metrics) in enumerate(mse_sorted, 1):
        print(f"{idx}. {model}: MSE Test = {metrics['MSE_test']:.6f}")
    print()
    # Ranking theo R² Test (giảm dần)
    print("Xếp hạng theo R² Test (từ cao đến thấp):")
    r2_sorted = sorted(results.items(), key=lambda x: x[1]['R2_test'], reverse=True)
    for idx, (model, metrics) in enumerate(r2_sorted, 1):
        print(f"{idx}. {model}: R² Test = {metrics['R2_test']:.6f}")
    print("-" * 50)

def main():
    # Đọc dữ liệu từ file CSV
    csv_file = 'VNM.csv'
    df = get_stock_data(csv_file)
    # Kiểm tra dữ liệu
    if df is None or df.empty:
        print(f"Không đọc được dữ liệu từ file {csv_file}. Hãy kiểm tra lại file!")
        return
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Index của DataFrame không phải là DatetimeIndex. Không thể xử lý dữ liệu!")
        return
    # Chuẩn bị dữ liệu
    X, y, scaler = prepare_data(df)
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Huấn luyện và đánh giá mô hình
    results, mlp_history, rnn_history, X_train_reshaped, y_train, dt_model, lasso_model, ridge_model, knn_model, lr_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    # Lấy lại các mô hình đã huấn luyện để truyền vào plot_results
    trained_models = {'Decision Tree': dt_model, 'Lasso': lasso_model, 'Ridge': ridge_model, 'kNN': knn_model, 'Linear Regression': lr_model}
    # Vẽ đồ thị và in kết quả
    plot_results(results, mlp_history, rnn_history, y_test, X_train_reshaped, y_train, trained_models)

if __name__ == "__main__":
    main()