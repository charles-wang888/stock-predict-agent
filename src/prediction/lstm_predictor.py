"""
LSTM股票预测模型
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# TensorFlow是可选的
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow未安装，LSTM功能将不可用。可以使用简单预测模型。")


class LSTMPredictor:
    """基于LSTM的股票价格预测器"""
    
    def __init__(self, lookback_days: int = 60, prediction_days: int = 5, 
                 lstm_units: int = 50, dropout_rate: float = 0.2):
        """
        初始化LSTM预测器
        
        Args:
            lookback_days: 使用过去多少天的数据进行预测
            prediction_days: 预测未来多少天
            lstm_units: LSTM层单元数
            dropout_rate: Dropout比率
        """
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.feature_columns = None
        self.training_history = None
    
    def prepare_data(self, data: pd.DataFrame, use_indicators: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        Args:
            data: 股票数据DataFrame（应包含技术指标）
            use_indicators: 是否使用技术指标
        
        Returns:
            X: 特征数据 (samples, timesteps, features)
            y: 目标数据 (samples, prediction_days)
            feature_names: 特征名称列表
        """
        # 选择特征列
        base_features = ['close', 'open', 'high', 'low', 'volume']
        indicator_features = ['ma5', 'ma10', 'ma20', 'rsi', 'macd', 'macd_signal', 
                             'kdj_k', 'kdj_d', 'volume_ratio', 'volatility']
        
        if use_indicators:
            available_features = [f for f in base_features + indicator_features if f in data.columns]
        else:
            available_features = [f for f in base_features if f in data.columns]
        
        self.feature_columns = available_features
        
        # 提取特征数据
        feature_data = data[available_features].copy()
        
        # 填充缺失值
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
        
        # 标准化
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # 创建时间序列数据
        X, y = [], []
        for i in range(self.lookback_days, len(scaled_data) - self.prediction_days + 1):
            X.append(scaled_data[i - self.lookback_days:i])
            y.append(scaled_data[i:i + self.prediction_days, 0])  # 只预测收盘价
        
        return np.array(X), np.array(y), available_features
    
    def build_model(self, input_shape: Tuple[int, int]):
        """
        构建LSTM模型
        
        Args:
            input_shape: (timesteps, features)
        
        Returns:
            Keras模型
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法构建LSTM模型")
        
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25),
            Dense(self.prediction_days)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2, 
              epochs: int = 50, batch_size: int = 32, verbose: int = 0) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data: 训练数据
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 输出详细程度
        
        Returns:
            训练历史信息
        """
        # 准备数据
        X, y, feature_names = self.prepare_data(data)
        
        if len(X) == 0:
            raise ValueError("数据不足，无法训练模型")
        
        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 构建模型
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # 早停机制
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # 训练
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        # 评估
        train_pred = self.model.predict(X_train, verbose=0)
        val_pred = self.model.predict(X_val, verbose=0)
        
        # 反标准化
        train_pred_original = self._inverse_transform_predictions(train_pred, data)
        val_pred_original = self._inverse_transform_predictions(val_pred, data)
        y_train_original = self._inverse_transform_predictions(y_train, data)
        y_val_original = self._inverse_transform_predictions(y_val, data)
        
        train_rmse = np.sqrt(mean_squared_error(y_train_original, train_pred_original))
        val_rmse = np.sqrt(mean_squared_error(y_val_original, val_pred_original))
        train_mae = mean_absolute_error(y_train_original, train_pred_original)
        val_mae = mean_absolute_error(y_val_original, val_pred_original)
        
        return {
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse),
            'train_mae': float(train_mae),
            'val_mae': float(val_mae),
            'feature_names': feature_names,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
    
    def _inverse_transform_predictions(self, predictions: np.ndarray, original_data: pd.DataFrame) -> np.ndarray:
        """将预测结果反标准化"""
        # 创建一个临时数组用于反标准化
        temp_array = np.zeros((predictions.shape[0], predictions.shape[1], len(self.feature_columns)))
        temp_array[:, :, 0] = predictions  # 收盘价在第一个位置
        predictions_original = self.scaler.inverse_transform(temp_array.reshape(-1, len(self.feature_columns)))[:, 0]
        return predictions_original.reshape(predictions.shape)
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        预测未来价格
        
        Args:
            data: 历史数据（应包含技术指标）
        
        Returns:
            预测结果字典
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 准备数据
        X, _, _ = self.prepare_data(data.tail(self.lookback_days + self.prediction_days))
        
        if len(X) == 0:
            # 如果数据不足，返回简单预测
            return self._simple_fallback_prediction(data)
        
        # 使用最后一段数据进行预测
        X_last = X[-1:].reshape(1, self.lookback_days, len(self.feature_columns))
        
        # 预测
        predictions_scaled = self.model.predict(X_last, verbose=0)
        
        # 反标准化
        predictions = self._inverse_transform_predictions(predictions_scaled, data)
        predictions = predictions[0]  # 取第一个样本
        
        # 获取当前价格
        current_price = float(data['close'].iloc[-1])
        
        # 计算预测涨跌幅
        predicted_return = (predictions[-1] - current_price) / current_price
        
        # 计算置信度（基于训练误差）
        if self.training_history:
            val_loss = min(self.training_history.history['val_loss'])
            # 将损失转换为置信度（损失越小，置信度越高）
            confidence = max(0.5, min(0.95, 1 - min(val_loss / (current_price * 0.1), 0.5)))
        else:
            confidence = 0.7
        
        # 风险评估
        volatility = data['volatility'].iloc[-1] if 'volatility' in data.columns else 0.02
        if abs(predicted_return) < 0.02 and volatility < 0.03:
            risk_level = "低"
        elif abs(predicted_return) < 0.05 and volatility < 0.05:
            risk_level = "中"
        else:
            risk_level = "高"
        
        return {
            'current_price': current_price,
            'predictions': [float(p) for p in predictions],
            'predicted_return': float(predicted_return),
            'predicted_return_pct': float(predicted_return * 100),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'trend': '上涨' if predicted_return > 0 else '下跌',
            'prediction_days': self.prediction_days,
            'prediction_dates': [
                (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(1, self.prediction_days + 1)
            ],
            'model_type': 'LSTM'
        }
    
    def _simple_fallback_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """简单的回退预测（当数据不足时）"""
        current_price = float(data['close'].iloc[-1])
        trend = float(data['close'].pct_change().tail(5).mean())
        predictions = [current_price * (1 + trend * i) for i in range(1, self.prediction_days + 1)]
        
        return {
            'current_price': current_price,
            'predictions': predictions,
            'predicted_return': trend * self.prediction_days,
            'predicted_return_pct': trend * self.prediction_days * 100,
            'confidence': 0.6,
            'risk_level': '中',
            'trend': '上涨' if trend > 0 else '下跌',
            'prediction_days': self.prediction_days,
            'prediction_dates': [
                (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(1, self.prediction_days + 1)
            ],
            'model_type': 'Simple'
        }
    
    def evaluate_accuracy(self, data: pd.DataFrame, test_days: int = 15) -> Dict[str, Any]:
        """
        使用最近N天的数据评估模型准确性
        
        Args:
            data: 完整历史数据
            test_days: 用于测试的天数（最近N天）
        
        Returns:
            评估结果
        """
        if self.model is None:
            return {'error': '模型未训练'}
        
        if len(data) < self.lookback_days + test_days:
            return {'error': f'数据不足，需要至少{self.lookback_days + test_days}天数据'}
        
        # 使用最近test_days天的数据作为测试集
        test_data = data.tail(self.lookback_days + test_days)
        
        # 准备测试数据
        X_test, y_test, _ = self.prepare_data(test_data)
        
        if len(X_test) == 0:
            return {'error': '测试数据准备失败'}
        
        # 预测
        predictions_scaled = self.model.predict(X_test, verbose=0)
        predictions = self._inverse_transform_predictions(predictions_scaled, test_data)
        
        # 反标准化实际值
        y_test_original = self._inverse_transform_predictions(y_test, test_data)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
        mae = mean_absolute_error(y_test_original, predictions)
        
        # 计算方向准确率（预测涨跌方向是否正确）
        actual_directions = np.sign(np.diff(y_test_original, axis=1))
        pred_directions = np.sign(np.diff(predictions, axis=1))
        direction_accuracy = np.mean(actual_directions == pred_directions)
        
        # 计算价格误差百分比
        price_errors = np.abs((y_test_original - predictions) / y_test_original) * 100
        mean_error_pct = np.mean(price_errors)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mean_error_pct': float(mean_error_pct),
            'direction_accuracy': float(direction_accuracy),
            'test_samples': len(X_test),
            'test_days': test_days
        }

