"""
技术指标计算模块 - 集成Talib量化交易库
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

# 尝试导入Talib，如果失败则使用自定义实现
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("提示: TA-Lib未安装，将使用自定义技术指标计算")


class TechnicalIndicators:
    """技术指标计算类"""
    
    @staticmethod
    def calculate_ma(data: pd.DataFrame, period: int = 5) -> pd.Series:
        """计算移动平均线（MA）"""
        return data['close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int = 12) -> pd.Series:
        """计算指数移动平均线（EMA）"""
        return data['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """计算MACD指标（优先使用Talib）"""
        if TALIB_AVAILABLE:
            try:
                macd, signal_line, histogram = talib.MACD(
                    data['close'].values,
                    fastperiod=fast,
                    slowperiod=slow,
                    signalperiod=signal
                )
                return {
                    'macd': pd.Series(macd, index=data.index),
                    'signal': pd.Series(signal_line, index=data.index),
                    'histogram': pd.Series(histogram, index=data.index)
                }
            except:
                pass
        
        # 如果Talib不可用或失败，使用自定义实现
        ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算相对强弱指标（RSI，优先使用Talib）"""
        if TALIB_AVAILABLE:
            try:
                rsi = talib.RSI(data['close'].values, timeperiod=period)
                return pd.Series(rsi, index=data.index)
            except:
                pass
        
        # 如果Talib不可用或失败，使用自定义实现
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """计算布林带（Bollinger Bands）"""
        ma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': ma,
            'lower': lower_band
        }
    
    @staticmethod
    def calculate_kdj(data: pd.DataFrame, period: int = 9) -> Dict[str, pd.Series]:
        """计算KDJ指标"""
        low_min = data['low'].rolling(window=period).min()
        high_max = data['high'].rolling(window=period).max()
        rsv = (data['close'] - low_min) / (high_max - low_min) * 100
        
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return {
            'k': k,
            'd': d,
            'j': j
        }
    
    @staticmethod
    def calculate_volume_ratio(data: pd.DataFrame, period: int = 5) -> pd.Series:
        """计算量比"""
        avg_volume = data['volume'].rolling(window=period).mean()
        volume_ratio = data['volume'] / avg_volume
        return volume_ratio
    
    @staticmethod
    def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """添加所有技术指标到数据框（集成Talib）"""
        df = data.copy()
        
        # 确保数据按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # 移动平均线（使用Talib如果可用）
        if TALIB_AVAILABLE:
            try:
                df['ma5'] = talib.SMA(df['close'].values, timeperiod=5)
                df['ma10'] = talib.SMA(df['close'].values, timeperiod=10)
                df['ma20'] = talib.SMA(df['close'].values, timeperiod=20)
                df['ma60'] = talib.SMA(df['close'].values, timeperiod=60)
            except:
                df['ma5'] = TechnicalIndicators.calculate_ma(df, 5)
                df['ma10'] = TechnicalIndicators.calculate_ma(df, 10)
                df['ma20'] = TechnicalIndicators.calculate_ma(df, 20)
                df['ma60'] = TechnicalIndicators.calculate_ma(df, 60)
        else:
            df['ma5'] = TechnicalIndicators.calculate_ma(df, 5)
            df['ma10'] = TechnicalIndicators.calculate_ma(df, 10)
            df['ma20'] = TechnicalIndicators.calculate_ma(df, 20)
            df['ma60'] = TechnicalIndicators.calculate_ma(df, 60)
        
        # EMA（使用Talib如果可用）
        if TALIB_AVAILABLE:
            try:
                df['ema12'] = talib.EMA(df['close'].values, timeperiod=12)
                df['ema26'] = talib.EMA(df['close'].values, timeperiod=26)
            except:
                df['ema12'] = TechnicalIndicators.calculate_ema(df, 12)
                df['ema26'] = TechnicalIndicators.calculate_ema(df, 26)
        else:
            df['ema12'] = TechnicalIndicators.calculate_ema(df, 12)
            df['ema26'] = TechnicalIndicators.calculate_ema(df, 26)
        
        # MACD（优先使用Talib）
        macd_data = TechnicalIndicators.calculate_macd(df)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['histogram']
        
        # RSI（优先使用Talib）
        df['rsi'] = TechnicalIndicators.calculate_rsi(df)
        
        # 布林带（使用Talib如果可用）
        if TALIB_AVAILABLE:
            try:
                upper, middle, lower = talib.BBANDS(
                    df['close'].values,
                    timeperiod=20,
                    nbdevup=2,
                    nbdevdn=2,
                    matype=0
                )
                df['bb_upper'] = pd.Series(upper, index=df.index)
                df['bb_middle'] = pd.Series(middle, index=df.index)
                df['bb_lower'] = pd.Series(lower, index=df.index)
            except:
                bb_data = TechnicalIndicators.calculate_bollinger_bands(df)
                df['bb_upper'] = bb_data['upper']
                df['bb_middle'] = bb_data['middle']
                df['bb_lower'] = bb_data['lower']
        else:
            bb_data = TechnicalIndicators.calculate_bollinger_bands(df)
            df['bb_upper'] = bb_data['upper']
            df['bb_middle'] = bb_data['middle']
            df['bb_lower'] = bb_data['lower']
        
        # KDJ（使用Talib的STOCH）
        if TALIB_AVAILABLE:
            try:
                slowk, slowd = talib.STOCH(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    fastk_period=9,
                    slowk_period=3,
                    slowd_period=3
                )
                df['kdj_k'] = pd.Series(slowk, index=df.index)
                df['kdj_d'] = pd.Series(slowd, index=df.index)
                df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
            except:
                kdj_data = TechnicalIndicators.calculate_kdj(df)
                df['kdj_k'] = kdj_data['k']
                df['kdj_d'] = kdj_data['d']
                df['kdj_j'] = kdj_data['j']
        else:
            kdj_data = TechnicalIndicators.calculate_kdj(df)
            df['kdj_k'] = kdj_data['k']
            df['kdj_d'] = kdj_data['d']
            df['kdj_j'] = kdj_data['j']
        
        # 量比
        df['volume_ratio'] = TechnicalIndicators.calculate_volume_ratio(df)
        
        # 价格变化率
        df['price_change'] = df['close'].pct_change()
        
        # 波动率
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # 其他Talib指标（如果可用）
        if TALIB_AVAILABLE:
            try:
                # ATR（平均真实波幅）
                df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                
                # CCI（商品通道指数）
                df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                
                # OBV（能量潮）
                df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            except:
                pass
        
        return df

