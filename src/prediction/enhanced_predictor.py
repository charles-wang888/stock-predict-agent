"""
增强的股票预测器 - 整合LSTM和传统方法
"""
import pandas as pd
from typing import Dict, Any, Optional
from .stock_predictor import StockPredictor
from .lstm_predictor import LSTMPredictor
from datetime import datetime, timedelta


class EnhancedStockPredictor:
    """增强的股票预测器，支持多种预测模型"""
    
    def __init__(self, lookback_days: int = 60, prediction_days: int = 5, 
                 use_lstm: bool = True, auto_train: bool = True):
        """
        初始化增强预测器
        
        Args:
            lookback_days: 使用过去多少天的数据
            prediction_days: 预测未来多少天
            use_lstm: 是否使用LSTM模型
            auto_train: 是否自动训练LSTM模型
        """
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.use_lstm = use_lstm
        self.auto_train = auto_train
        
        # 初始化预测器
        self.simple_predictor = StockPredictor(lookback_days, prediction_days)
        
        # 尝试初始化LSTM预测器（如果可用）
        self.lstm_predictor = None
        self.lstm_trained = False
        if use_lstm:
            try:
                from .lstm_predictor import TENSORFLOW_AVAILABLE
                if TENSORFLOW_AVAILABLE:
                    self.lstm_predictor = LSTMPredictor(lookback_days, prediction_days)
                else:
                    print("提示: TensorFlow未安装，将使用简单预测模型")
            except Exception as e:
                print(f"初始化LSTM预测器失败: {e}，将使用简单预测模型")
    
    def predict(self, stock_data: pd.DataFrame, force_retrain: bool = False) -> Dict[str, Any]:
        """
        预测股票价格（自动选择最佳模型）
        
        Args:
            stock_data: 历史股票数据（应包含技术指标）
            force_retrain: 是否强制重新训练LSTM模型
        
        Returns:
            预测结果
        """
        if stock_data is None or stock_data.empty:
            return self.simple_predictor._generate_mock_prediction()
        
        # 确保数据足够
        if len(stock_data) < self.lookback_days:
            return self.simple_predictor._generate_mock_prediction()
        
        # 尝试使用LSTM模型
        if self.use_lstm and self.lstm_predictor is not None:
            try:
                # 检查是否需要训练
                if not self.lstm_trained or force_retrain:
                    if self.auto_train and len(stock_data) >= self.lookback_days + 30:
                        # 训练模型（使用除最后15天外的数据）
                        train_data = stock_data.iloc[:-15] if len(stock_data) > 60 else stock_data
                        self.lstm_predictor.train(train_data, epochs=30, verbose=0)
                        self.lstm_trained = True
                
                # 使用LSTM预测
                if self.lstm_trained:
                    prediction = self.lstm_predictor.predict(stock_data)
                    
                    # 评估准确性（使用最近15天数据）
                    try:
                        if len(stock_data) >= self.lookback_days + 15:
                            accuracy = self.lstm_predictor.evaluate_accuracy(stock_data, test_days=15)
                            if accuracy and 'error' not in accuracy:
                                prediction['accuracy_metrics'] = accuracy
                                prediction['model_confidence'] = accuracy.get('direction_accuracy', 0.7)
                    except Exception as e:
                        print(f"评估准确性时出错: {e}")
                    
                    return prediction
            except Exception as e:
                print(f"LSTM预测失败，使用简单模型: {e}")
        
        # 回退到简单模型
        return self.simple_predictor.predict(stock_data)
    
    def evaluate_model_accuracy(self, stock_data: pd.DataFrame, test_days: int = 15) -> Dict[str, Any]:
        """
        评估模型准确性（使用最近N天的数据）
        
        Args:
            stock_data: 完整历史数据
            test_days: 用于测试的天数
        
        Returns:
            评估结果
        """
        if not self.lstm_predictor or not self.lstm_trained:
            return {'error': 'LSTM模型未训练'}
        
        return self.lstm_predictor.evaluate_accuracy(stock_data, test_days)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'use_lstm': self.use_lstm,
            'lstm_trained': self.lstm_trained,
            'lookback_days': self.lookback_days,
            'prediction_days': self.prediction_days
        }
        
        if self.lstm_predictor and self.lstm_trained:
            info['lstm_units'] = self.lstm_predictor.lstm_units
            if self.lstm_predictor.training_history:
                info['training_loss'] = min(self.lstm_predictor.training_history.history['val_loss'])
        
        return info

