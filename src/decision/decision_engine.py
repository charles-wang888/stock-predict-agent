"""
决策引擎模块
"""
from typing import Dict, Any
from ..rule_engine.rule_engine import RuleEngine
from config.settings import DEFAULT_ACCOUNT


class DecisionEngine:
    """决策引擎：整合预测结果和业务规则"""
    
    def __init__(self, rule_engine: RuleEngine, account: Dict[str, Any] = None):
        """
        初始化决策引擎
        
        Args:
            rule_engine: 规则引擎实例
            account: 账户信息
        """
        self.rule_engine = rule_engine
        self.account = account or DEFAULT_ACCOUNT.copy()
    
    def make_decision(self, stock_code: str, prediction: Dict[str, Any], 
                     current_price: float, trade_amount: float = None) -> Dict[str, Any]:
        """
        做出交易决策
        
        Args:
            stock_code: 股票代码
            prediction: 预测结果
            current_price: 当前价格
            trade_amount: 交易金额（如果None，自动计算）
        
        Returns:
            决策结果
        """
        # 计算交易金额（如果没有指定，使用预测置信度来决定）
        if trade_amount is None:
            # 根据预测置信度和预期收益率决定交易金额
            confidence = prediction.get('confidence', 0.5)
            predicted_return = abs(prediction.get('predicted_return', 0))
            
            # 保守策略：最多使用可用资金的30%，根据置信度调整
            max_trade_amount = self.account['available_cash'] * 0.3
            trade_amount = max_trade_amount * confidence * min(predicted_return * 20, 1.0)
            trade_amount = min(trade_amount, self.account['available_cash'])
        
        # 计算持仓信息
        position = self.account['positions'].get(stock_code, {'shares': 0, 'cost': 0})
        position_value = position['shares'] * current_price
        position_ratio = position_value / self.account['total_assets'] if self.account['total_assets'] > 0 else 0
        
        # 计算新交易后的持仓
        new_shares = trade_amount / current_price
        new_position_value = (position['shares'] + new_shares) * current_price
        new_position_ratio = new_position_value / self.account['total_assets'] if self.account['total_assets'] > 0 else 0
        
        # 准备规则评估上下文
        from datetime import datetime
        current_hour = datetime.now().hour
        
        context = {
            'stock_code': stock_code,
            'current_price': current_price,
            'trade_amount': trade_amount,
            'position_ratio': new_position_ratio,  # 交易后的持仓比例
            'available_cash': self.account['available_cash'] - trade_amount,  # 交易后的可用资金
            'total_assets': self.account['total_assets'],
            'predicted_return': prediction.get('predicted_return', 0),
            'confidence': prediction.get('confidence', 0.5),
            'risk_level': prediction.get('risk_level', '中'),
            'current_hour': current_hour  # 添加当前小时，用于时间窗口规则
        }
        
        # 如果有持仓，计算盈亏
        if position['shares'] > 0:
            profit = (current_price - position['cost']) * position['shares']
            profit_ratio = (current_price - position['cost']) / position['cost'] if position['cost'] > 0 else 0
            context['loss_ratio'] = profit_ratio  # 如果是负数就是亏损
            context['total_loss_ratio'] = profit / self.account['total_assets']  # 总资产亏损比例
        
        # 评估规则
        rule_result = self.rule_engine.evaluate_all(context)
        
        # 生成决策建议
        decision = self._generate_decision(
            stock_code=stock_code,
            prediction=prediction,
            rule_result=rule_result,
            trade_amount=trade_amount,
            current_price=current_price,
            context=context
        )
        
        return decision
    
    def _generate_decision(self, stock_code: str, prediction: Dict[str, Any],
                          rule_result: Dict[str, Any], trade_amount: float,
                          current_price: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终决策"""
        
        # 根据预测结果决定基本操作
        predicted_return = prediction.get('predicted_return', 0)
        confidence = prediction.get('confidence', 0.5)
        
        # 基础决策
        if rule_result['is_allowed']:
            if predicted_return > 0.02 and confidence > 0.6:
                action = '买入'
                suggestion = f"预测上涨概率{confidence*100:.1f}%，预期收益率{predicted_return*100:.2f}%，建议买入"
            elif predicted_return < -0.02:
                action = '卖出'
                suggestion = f"预测可能下跌，建议卖出或观望"
            else:
                action = '持有'
                suggestion = f"预测趋势不明显，建议持有或观望"
        else:
            action = '禁止'
            suggestion = rule_result['reason']
        
        # 计算建议数量（必须是100的倍数，符合A股交易规则）
        if action == '买入' and current_price > 0:
            raw_shares = trade_amount / current_price
            shares = (int(raw_shares) // 100) * 100  # 向下取整到100的倍数
            if shares < 100:
                shares = 0  # 如果不足100股，建议不买入
        else:
            shares = 0
        
        # 组装决策结果
        decision = {
            'stock_code': stock_code,
            'action': action,
            'suggestion': suggestion,
            'current_price': current_price,
            'suggested_shares': shares,
            'suggested_amount': trade_amount,
            'prediction': prediction,
            'rule_evaluation': rule_result,
            'reasoning': self._generate_reasoning(prediction, rule_result)
        }
        
        return decision
    
    def _generate_reasoning(self, prediction: Dict[str, Any], 
                           rule_result: Dict[str, Any]) -> str:
        """生成决策理由（可解释性）"""
        reasoning_parts = []
        
        # 预测部分
        reasoning_parts.append(
            f"预测分析：预测{prediction.get('trend', '未知')}，"
            f"预期收益率{prediction.get('predicted_return_pct', 0):.2f}%，"
            f"置信度{prediction.get('confidence', 0)*100:.1f}%，"
            f"风险等级：{prediction.get('risk_level', '中')}"
        )
        
        # 规则评估部分
        if rule_result['triggered_rules']:
            reasoning_parts.append(
                f"规则评估：触发了{rule_result['rule_count']}条规则"
            )
            for rule in rule_result['triggered_rules'][:3]:  # 只显示前3条
                reasoning_parts.append(f"  - {rule['name']}：{rule.get('description', '')}")
        
        if rule_result['warnings']:
            reasoning_parts.append("警告信息：")
            for warning in rule_result['warnings']:
                reasoning_parts.append(f"  - {warning['message']}")
        
        if rule_result['optimizations']:
            reasoning_parts.append("优化建议：")
            for opt in rule_result['optimizations']:
                reasoning_parts.append(f"  - {opt['message']}")
        
        return "\n".join(reasoning_parts)

