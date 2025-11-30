"""
模型可解释性模块
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: SHAP未安装，部分可解释性功能将不可用")


class ModelExplainer:
    """模型可解释性分析器"""
    
    def __init__(self, model=None, feature_names: List[str] = None):
        """
        初始化解释器
        
        Args:
            model: 训练好的模型（可选）
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names or []
        self.shap_explainer = None
        self.shap_values = None
    
    def explain_prediction(self, prediction_data: pd.DataFrame, 
                          model=None, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        解释预测结果
        
        Args:
            prediction_data: 用于预测的数据
            model: 预测模型（如果未在初始化时提供）
            feature_names: 特征名称（如果未在初始化时提供）
        
        Returns:
            解释结果
        """
        model = model or self.model
        feature_names = feature_names or self.feature_names
        
        if model is None:
            return {'error': '模型未提供'}
        
        explanations = {
            'feature_importance': self._calculate_feature_importance(prediction_data, model),
            'prediction_factors': self._analyze_prediction_factors(prediction_data),
            'risk_factors': self._identify_risk_factors(prediction_data),
            'trend_analysis': self._analyze_trend(prediction_data)
        }
        
        # 如果SHAP可用，添加SHAP解释
        if SHAP_AVAILABLE and hasattr(model, 'predict'):
            try:
                shap_explanation = self._shap_explain(model, prediction_data, feature_names)
                explanations['shap_values'] = shap_explanation
            except Exception as e:
                print(f"SHAP解释失败: {e}")
        
        return explanations
    
    def _calculate_feature_importance(self, data: pd.DataFrame, model) -> Dict[str, float]:
        """计算特征重要性（基于相关性）"""
        if 'close' not in data.columns or len(data) < 2:
            return {}
        
        # 计算与目标变量的相关性
        target = data['close'].pct_change().dropna()
        
        importance = {}
        for col in data.columns:
            if col in ['close', 'date']:
                continue
            try:
                if data[col].dtype in [np.float64, np.int64]:
                    corr = abs(data[col].corr(target))
                    if not np.isnan(corr):
                        importance[col] = float(corr)
            except:
                pass
        
        # 归一化
        if importance:
            max_val = max(importance.values())
            if max_val > 0:
                importance = {k: v / max_val for k, v in importance.items()}
        
        # 排序
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return importance
    
    def _analyze_prediction_factors(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析影响预测的因素"""
        factors = {}
        
        # 价格趋势
        if 'close' in data.columns and len(data) >= 5:
            recent_prices = data['close'].tail(5)
            price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            factors['price_trend'] = {
                'value': float(price_trend),
                'description': '近期价格趋势',
                'impact': 'positive' if price_trend > 0 else 'negative'
            }
        
        # 成交量
        if 'volume' in data.columns and 'volume_ratio' in data.columns:
            volume_ratio = data['volume_ratio'].iloc[-1] if not pd.isna(data['volume_ratio'].iloc[-1]) else 1.0
            factors['volume'] = {
                'value': float(volume_ratio),
                'description': '成交量比率',
                'impact': 'positive' if volume_ratio > 1.2 else 'neutral' if volume_ratio > 0.8 else 'negative'
            }
        
        # RSI
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1] if not pd.isna(data['rsi'].iloc[-1]) else 50.0
            factors['rsi'] = {
                'value': float(rsi),
                'description': '相对强弱指标',
                'impact': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
            }
        
        # MACD
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd = data['macd'].iloc[-1] if not pd.isna(data['macd'].iloc[-1]) else 0.0
            signal = data['macd_signal'].iloc[-1] if not pd.isna(data['macd_signal'].iloc[-1]) else 0.0
            factors['macd'] = {
                'value': float(macd - signal),
                'description': 'MACD信号',
                'impact': 'positive' if macd > signal else 'negative'
            }
        
        return factors
    
    def _identify_risk_factors(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """识别风险因素"""
        risks = []
        
        # 高波动率
        if 'volatility' in data.columns:
            vol = data['volatility'].iloc[-1] if not pd.isna(data['volatility'].iloc[-1]) else 0.0
            if vol > 0.05:
                risks.append({
                    'type': 'high_volatility',
                    'level': 'high',
                    'description': f'波动率较高 ({vol*100:.2f}%)，价格波动较大'
                })
        
        # RSI超买超卖
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1] if not pd.isna(data['rsi'].iloc[-1]) else 50.0
            if rsi > 80:
                risks.append({
                    'type': 'overbought',
                    'level': 'medium',
                    'description': f'RSI指标显示超买 ({rsi:.1f})，可能面临回调'
                })
            elif rsi < 20:
                risks.append({
                    'type': 'oversold',
                    'level': 'low',
                    'description': f'RSI指标显示超卖 ({rsi:.1f})，可能存在反弹机会'
                })
        
        # 成交量异常
        if 'volume_ratio' in data.columns:
            vr = data['volume_ratio'].iloc[-1] if not pd.isna(data['volume_ratio'].iloc[-1]) else 1.0
            if vr < 0.5:
                risks.append({
                    'type': 'low_volume',
                    'level': 'medium',
                    'description': f'成交量较低 (比率: {vr:.2f})，流动性可能不足'
                })
        
        return risks
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析趋势"""
        if 'close' not in data.columns or len(data) < 10:
            return {}
        
        prices = data['close'].tail(10)
        
        # 短期趋势
        short_trend = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
        
        # 长期趋势
        long_trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        
        # 趋势强度
        trend_strength = abs(short_trend) / (abs(long_trend) + 0.001)
        
        return {
            'short_term': float(short_trend),
            'long_term': float(long_trend),
            'strength': float(trend_strength),
            'direction': 'up' if short_trend > 0 else 'down',
            'description': self._generate_trend_description(short_trend, long_trend, trend_strength)
        }
    
    def _generate_trend_description(self, short_trend: float, long_trend: float, strength: float) -> str:
        """生成趋势描述"""
        if short_trend > 0.02 and long_trend > 0:
            return "短期和长期均呈上涨趋势，趋势较强"
        elif short_trend > 0.02 and long_trend < 0:
            return "短期上涨但长期下跌，可能存在反弹"
        elif short_trend < -0.02 and long_trend < 0:
            return "短期和长期均呈下跌趋势，需谨慎"
        elif short_trend < -0.02 and long_trend > 0:
            return "短期下跌但长期上涨，可能是调整"
        else:
            return "趋势不明显，处于震荡状态"
    
    def _shap_explain(self, model, data: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """使用SHAP解释模型"""
        if not SHAP_AVAILABLE:
            return {}
        
        try:
            # 准备数据
            feature_data = data[feature_names].fillna(0).values
            
            # 创建SHAP解释器
            if len(feature_data) > 100:
                # 使用样本数据
                sample_data = feature_data[:100]
            else:
                sample_data = feature_data
            
            # 使用TreeExplainer或KernelExplainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(sample_data)
            else:
                explainer = shap.KernelExplainer(model.predict, sample_data[:10])
                shap_values = explainer.shap_values(sample_data)
            
            # 计算特征重要性
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # 取第一个输出
            
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            return {
                'feature_importance': dict(zip(feature_names, feature_importance.tolist())),
                'shap_values_available': True
            }
        except Exception as e:
            print(f"SHAP解释出错: {e}")
            return {'error': str(e)}
    
    def generate_natural_language_explanation(self, prediction: Dict[str, Any], 
                                            explanations: Dict[str, Any]) -> str:
        """生成自然语言解释"""
        parts = []
        
        # 预测结果
        trend = prediction.get('trend', '未知')
        confidence = prediction.get('confidence', 0) * 100
        return_pct = prediction.get('predicted_return_pct', 0)
        
        parts.append(f"根据模型分析，该股票预测{trend}，预期收益率{return_pct:.2f}%，置信度{confidence:.1f}%。")
        
        # 关键因素
        factors = explanations.get('prediction_factors', {})
        if factors:
            parts.append("\n主要影响因素：")
            for name, factor in list(factors.items())[:3]:
                desc = factor.get('description', '')
                impact = factor.get('impact', '')
                parts.append(f"  - {desc}：{impact}")
        
        # 风险因素
        risks = explanations.get('risk_factors', [])
        if risks:
            parts.append("\n风险提示：")
            for risk in risks[:3]:
                parts.append(f"  - {risk.get('description', '')}")
        
        # 趋势分析
        trend_analysis = explanations.get('trend_analysis', {})
        if trend_analysis.get('description'):
            parts.append(f"\n趋势分析：{trend_analysis['description']}")
        
        return "\n".join(parts)


