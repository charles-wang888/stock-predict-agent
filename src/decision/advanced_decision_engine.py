"""
高级决策引擎 - 体现复杂业务逻辑的实现
支持多层次决策、规则依赖、动态参数调整等复杂业务逻辑
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from ..rule_engine.rule_engine import RuleEngine
from config.settings import DEFAULT_ACCOUNT


class AdvancedDecisionEngine:
    """
    高级决策引擎 - 体现复杂业务逻辑
    
    核心特性：
    1. 多阶段决策流程（预检查 -> 风险评估 -> 规则评估 -> 策略选择 -> 最终决策）
    2. 规则依赖关系（规则链和规则组）
    3. 动态参数调整（根据市场状态、历史表现）
    4. 综合风险评估（多维度风险计算）
    5. 交易策略链（买入 -> 持仓管理 -> 卖出）
    """
    
    def __init__(self, rule_engine: RuleEngine, account: Dict[str, Any] = None):
        """
        初始化高级决策引擎
        
        Args:
            rule_engine: 规则引擎实例
            account: 账户信息
        """
        self.rule_engine = rule_engine
        self.account = account or DEFAULT_ACCOUNT.copy()
        self.decision_history = []  # 决策历史记录
        self.rule_adjustments = {}  # 规则参数动态调整记录
    
    def make_decision(self, stock_code: str, prediction: Dict[str, Any], 
                     current_price: float, stock_data: pd.DataFrame = None,
                     trade_amount: float = None) -> Dict[str, Any]:
        """
        做出交易决策（多阶段复杂流程）
        
        决策流程：
        1. 预检查阶段：基础条件检查
        2. 风险评估阶段：多维度风险评估
        3. 规则评估阶段：规则依赖链评估
        4. 策略选择阶段：根据市场状态选择策略
        5. 资金计算阶段：动态资金分配
        6. 最终决策阶段：综合所有因素生成决策
        
        Args:
            stock_code: 股票代码
            prediction: 预测结果
            current_price: 当前价格
            stock_data: 股票历史数据（用于风险评估）
            trade_amount: 交易金额（如果None，自动计算）
        
        Returns:
            决策结果（包含详细的决策过程和推理）
        """
        # 记录决策开始
        decision_context = {
            'stock_code': stock_code,
            'current_price': current_price,
            'timestamp': datetime.now(),
            'stages': []  # 记录各阶段执行情况
        }
        
        # ========== 阶段1：预检查阶段 ==========
        precheck_result = self._precheck_stage(stock_code, current_price, prediction)
        decision_context['stages'].append({
            'stage': '预检查',
            'result': precheck_result
        })
        
        if not precheck_result['passed']:
            return self._build_final_decision(
                stock_code=stock_code,
                action='禁止',
                current_price=current_price,
                reason=precheck_result['reason'],
                context=decision_context,
                stage_details=decision_context['stages']
            )
        
        # ========== 阶段2：综合风险评估阶段 ==========
        risk_assessment = self._comprehensive_risk_assessment(
            stock_code, current_price, prediction, stock_data
        )
        decision_context['stages'].append({
            'stage': '风险评估',
            'result': risk_assessment
        })
        decision_context['risk_score'] = risk_assessment['total_risk_score']
        decision_context['risk_factors'] = risk_assessment['risk_factors']
        
        # ========== 阶段3：动态规则参数调整阶段 ==========
        adjusted_context = self._adjust_rule_parameters(
            stock_code, current_price, prediction, risk_assessment
        )
        decision_context['stages'].append({
            'stage': '规则参数调整',
            'result': adjusted_context
        })
        
        # ========== 阶段4：规则依赖链评估阶段 ==========
        rule_evaluation = self._evaluate_rule_chain(
            stock_code, current_price, prediction, adjusted_context
        )
        decision_context['stages'].append({
            'stage': '规则链评估',
            'result': rule_evaluation
        })
        
        if not rule_evaluation['is_allowed']:
            return self._build_final_decision(
                stock_code=stock_code,
                action='禁止',
                current_price=current_price,
                reason=rule_evaluation['reason'],
                context=decision_context,
                stage_details=decision_context['stages'],
                risk_assessment=risk_assessment
            )
        
        # ========== 阶段5：交易策略选择阶段 ==========
        strategy = self._select_trading_strategy(
            stock_code, current_price, prediction, risk_assessment, rule_evaluation
        )
        decision_context['stages'].append({
            'stage': '策略选择',
            'result': strategy
        })
        
        # ========== 阶段6：动态资金计算阶段 ==========
        if trade_amount is None:
            trade_amount = self._calculate_dynamic_trade_amount(
                stock_code, current_price, prediction, risk_assessment, 
                strategy, adjusted_context
            )
        
        # ========== 阶段7：最终决策生成阶段 ==========
        final_decision = self._generate_final_decision(
            stock_code=stock_code,
            prediction=prediction,
            rule_result=rule_evaluation,
            risk_assessment=risk_assessment,
            strategy=strategy,
            trade_amount=trade_amount,
            current_price=current_price,
            context=adjusted_context
        )
        
        decision_context['stages'].append({
            'stage': '最终决策',
            'result': final_decision
        })
        
        # 保存决策历史
        decision_record = {
            'stock_code': stock_code,
            'timestamp': datetime.now(),
            'action': final_decision['action'],
            'risk_score': risk_assessment['total_risk_score'],
            'strategy': strategy['name']
        }
        self.decision_history.append(decision_record)
        
        # 构建完整决策结果
        final_decision['decision_process'] = decision_context['stages']
        final_decision['risk_assessment'] = risk_assessment
        final_decision['strategy'] = strategy
        final_decision['rule_adjustments'] = adjusted_context.get('adjustments', {})
        
        return final_decision
    
    def _precheck_stage(self, stock_code: str, current_price: float, 
                       prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        预检查阶段：基础条件检查
        
        复杂逻辑：
        - 价格合理性检查（是否为0或异常值）
        - 预测结果有效性检查
        - 账户状态基础检查
        """
        # 检查价格
        if current_price <= 0:
            return {
                'passed': False,
                'reason': '当前价格无效（<=0）',
                'checks': {'price_valid': False}
            }
        
        # 检查预测结果
        if not prediction or 'confidence' not in prediction:
            return {
                'passed': False,
                'reason': '预测结果无效',
                'checks': {'prediction_valid': False}
            }
        
        # 检查账户资金
        if self.account['available_cash'] < 100:
            return {
                'passed': False,
                'reason': '账户可用资金不足100元',
                'checks': {'sufficient_cash': False}
            }
        
        return {
            'passed': True,
            'reason': '预检查通过',
            'checks': {
                'price_valid': True,
                'prediction_valid': True,
                'sufficient_cash': True
            }
        }
    
    def _comprehensive_risk_assessment(self, stock_code: str, current_price: float,
                                      prediction: Dict[str, Any], 
                                      stock_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        综合风险评估阶段：多维度风险计算
        
        复杂业务逻辑：
        1. 价格波动风险（基于历史波动率）
        2. 预测置信度风险（置信度越低风险越高）
        3. 持仓集中度风险（单只股票持仓比例）
        4. 市场状态风险（大盘环境）
        5. 流动性风险（成交量）
        6. 综合风险评分（加权计算）
        """
        risk_factors = []
        risk_weights = {
            'volatility_risk': 0.25,
            'confidence_risk': 0.20,
            'concentration_risk': 0.20,
            'market_risk': 0.15,
            'liquidity_risk': 0.10,
            'account_risk': 0.10
        }
        
        total_risk_score = 0.0
        
        # 1. 价格波动风险（0-1，越高风险越大）
        volatility_risk = 0.5  # 默认中等风险
        if stock_data is not None and len(stock_data) > 0:
            if 'close' in stock_data.columns:
                prices = stock_data['close'].tail(20)
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    volatility = returns.std()
                    # 波动率转换为风险评分（0-1）
                    volatility_risk = min(abs(volatility) * 10, 1.0)
        
        risk_factors.append({
            'factor': '价格波动风险',
            'score': volatility_risk,
            'description': f'基于历史波动率评估，当前波动率风险评分：{volatility_risk:.2f}'
        })
        total_risk_score += volatility_risk * risk_weights['volatility_risk']
        
        # 2. 预测置信度风险
        confidence = prediction.get('confidence', 0.5)
        confidence_risk = 1.0 - confidence  # 置信度越低风险越高
        risk_factors.append({
            'factor': '预测置信度风险',
            'score': confidence_risk,
            'description': f'预测置信度为{confidence:.2%}，置信度风险：{confidence_risk:.2f}'
        })
        total_risk_score += confidence_risk * risk_weights['confidence_risk']
        
        # 3. 持仓集中度风险
        position = self.account['positions'].get(stock_code, {'shares': 0, 'cost': 0})
        position_value = position['shares'] * current_price
        position_ratio = position_value / self.account['total_assets'] if self.account['total_assets'] > 0 else 0
        # 持仓比例越高，集中度风险越大（非线性）
        concentration_risk = min((position_ratio / 0.3) ** 2, 1.0) if position_ratio > 0 else 0
        risk_factors.append({
            'factor': '持仓集中度风险',
            'score': concentration_risk,
            'description': f'当前持仓比例{position_ratio:.2%}，集中度风险：{concentration_risk:.2f}'
        })
        total_risk_score += concentration_risk * risk_weights['concentration_risk']
        
        # 4. 市场状态风险（基于预测的风险等级）
        risk_level = prediction.get('risk_level', '中')
        market_risk_map = {'低': 0.2, '中': 0.5, '高': 0.8}
        market_risk = market_risk_map.get(risk_level, 0.5)
        risk_factors.append({
            'factor': '市场状态风险',
            'score': market_risk,
            'description': f'预测风险等级为{risk_level}，市场风险：{market_risk:.2f}'
        })
        total_risk_score += market_risk * risk_weights['market_risk']
        
        # 5. 流动性风险（基于成交量）
        liquidity_risk = 0.3  # 默认低风险
        if stock_data is not None and len(stock_data) > 0:
            if 'volume' in stock_data.columns:
                recent_volumes = stock_data['volume'].tail(5)
                avg_volume = recent_volumes.mean()
                if avg_volume > 0:
                    # 成交量过小，流动性风险高
                    liquidity_risk = min(1.0 / (avg_volume / 1000000 + 1), 1.0)
        
        risk_factors.append({
            'factor': '流动性风险',
            'score': liquidity_risk,
            'description': f'基于成交量评估，流动性风险：{liquidity_risk:.2f}'
        })
        total_risk_score += liquidity_risk * risk_weights['liquidity_risk']
        
        # 6. 账户风险（总亏损比例）
        total_profit = sum(
            (pos.get('current_price', 0) - pos.get('cost', 0)) * pos.get('shares', 0)
            for pos in self.account['positions'].values()
        )
        account_loss_ratio = abs(min(total_profit / self.account['total_assets'], 0)) if self.account['total_assets'] > 0 else 0
        account_risk = min(account_loss_ratio * 5, 1.0)  # 亏损比例转换为风险
        risk_factors.append({
            'factor': '账户风险',
            'score': account_risk,
            'description': f'账户总亏损比例{account_loss_ratio:.2%}，账户风险：{account_risk:.2f}'
        })
        total_risk_score += account_risk * risk_weights['account_risk']
        
        # 确定综合风险等级
        if total_risk_score < 0.3:
            risk_level_str = '低'
        elif total_risk_score < 0.6:
            risk_level_str = '中'
        else:
            risk_level_str = '高'
        
        return {
            'total_risk_score': total_risk_score,
            'risk_level': risk_level_str,
            'risk_factors': risk_factors,
            'risk_weights': risk_weights,
            'description': f'综合风险评分：{total_risk_score:.2f}（{risk_level_str}风险）'
        }
    
    def _adjust_rule_parameters(self, stock_code: str, current_price: float,
                               prediction: Dict[str, Any],
                               risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        动态规则参数调整阶段
        
        复杂业务逻辑：
        - 根据风险等级动态调整规则阈值
        - 根据市场波动率调整止损止盈比例
        - 根据账户表现调整资金分配比例
        """
        context = {
            'stock_code': stock_code,
            'current_price': current_price,
            'trade_amount': 0,
            'position_ratio': 0,
            'available_cash': self.account['available_cash'],
            'total_assets': self.account['total_assets'],
            'predicted_return': prediction.get('predicted_return', 0),
            'confidence': prediction.get('confidence', 0.5),
            'risk_level': prediction.get('risk_level', '中'),
            'current_hour': datetime.now().hour,
            'adjustments': {}  # 记录调整的参数
        }
        
        risk_score = risk_assessment['total_risk_score']
        
        # 根据风险动态调整持仓限制
        # 高风险时降低单股持仓上限，低风险时可以适当提高
        base_position_limit = 0.3  # 基础限制30%
        if risk_score > 0.7:
            adjusted_limit = base_position_limit * 0.6  # 高风险时降至18%
            context['adjustments']['position_limit'] = {
                'original': base_position_limit,
                'adjusted': adjusted_limit,
                'reason': '高风险环境，降低单股持仓上限'
            }
            context['position_limit'] = adjusted_limit
        elif risk_score < 0.3:
            adjusted_limit = base_position_limit * 1.2  # 低风险时可提高到36%
            context['adjustments']['position_limit'] = {
                'original': base_position_limit,
                'adjusted': adjusted_limit,
                'reason': '低风险环境，适当提高单股持仓上限'
            }
            context['position_limit'] = adjusted_limit
        else:
            context['position_limit'] = base_position_limit
        
        # 根据风险调整止损止盈比例
        base_stop_loss = -0.05  # 基础止损-5%
        base_take_profit = 0.10  # 基础止盈10%
        
        if risk_score > 0.7:
            # 高风险时收紧止损
            adjusted_stop_loss = base_stop_loss * 0.8  # -4%
            adjusted_take_profit = base_take_profit * 1.2  # 12%
            context['adjustments']['stop_loss'] = {
                'original': base_stop_loss,
                'adjusted': adjusted_stop_loss,
                'reason': '高风险环境，收紧止损'
            }
            context['adjustments']['take_profit'] = {
                'original': base_take_profit,
                'adjusted': adjusted_take_profit,
                'reason': '高风险环境，提高止盈目标'
            }
        else:
            context['stop_loss'] = base_stop_loss
            context['take_profit'] = base_take_profit
        
        # 计算持仓信息
        position = self.account['positions'].get(stock_code, {'shares': 0, 'cost': 0})
        position_value = position['shares'] * current_price
        context['position_ratio'] = position_value / self.account['total_assets'] if self.account['total_assets'] > 0 else 0
        
        # 如果有持仓，计算盈亏比例（用于止损/止盈规则）
        if position['shares'] > 0 and position['cost'] > 0:
            profit_ratio = (current_price - position['cost']) / position['cost']
            context['loss_ratio'] = profit_ratio  # 正数表示盈利，负数表示亏损
            # 计算总资产亏损比例
            profit = (current_price - position['cost']) * position['shares']
            context['total_loss_ratio'] = profit / self.account['total_assets'] if self.account['total_assets'] > 0 else 0
        else:
            # 没有持仓时，设置为0
            context['loss_ratio'] = 0
            context['total_loss_ratio'] = 0
        
        return context
    
    def _evaluate_rule_chain(self, stock_code: str, current_price: float,
                            prediction: Dict[str, Any],
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """
        规则依赖链评估阶段
        
        复杂业务逻辑：
        1. 规则分组（风控规则组 -> 优化规则组）
        2. 规则依赖关系（某些规则必须在前置规则通过后才执行）
        3. 规则链执行（按依赖顺序执行）
        """
        # 补充上下文信息
        context['predicted_return'] = prediction.get('predicted_return', 0)
        context['confidence'] = prediction.get('confidence', 0.5)
        context['risk_level'] = prediction.get('risk_level', '中')
        
        # 确保 loss_ratio 和 total_loss_ratio 存在（如果之前没有计算）
        if 'loss_ratio' not in context:
            position = self.account['positions'].get(stock_code, {'shares': 0, 'cost': 0})
            if position['shares'] > 0 and position['cost'] > 0:
                profit_ratio = (current_price - position['cost']) / position['cost']
                context['loss_ratio'] = profit_ratio
                profit = (current_price - position['cost']) * position['shares']
                context['total_loss_ratio'] = profit / self.account['total_assets'] if self.account['total_assets'] > 0 else 0
            else:
                context['loss_ratio'] = 0
                context['total_loss_ratio'] = 0
        
        # 规则分组执行
        # 阶段1：执行风控规则组（优先级最高）
        context['rule_group'] = 'risk_control'
        risk_control_result = self.rule_engine.evaluate_all(context)
        
        # 如果风控规则禁止，直接返回
        if not risk_control_result['is_allowed']:
            return {
                'is_allowed': False,
                'reason': f"风控规则禁止：{risk_control_result['reason']}",
                'rule_group': 'risk_control',
                'triggered_rules': risk_control_result['triggered_rules'],
                'warnings': risk_control_result['warnings'],
                'optimizations': []
            }
        
        # 阶段2：执行优化规则组（风控通过后才执行）
        context['rule_group'] = 'optimization'
        optimization_result = self.rule_engine.evaluate_all(context)
        
        # 合并结果
        return {
            'is_allowed': True,
            'reason': '规则评估通过',
            'rule_groups': {
                'risk_control': risk_control_result,
                'optimization': optimization_result
            },
            'triggered_rules': risk_control_result['triggered_rules'] + optimization_result['triggered_rules'],
            'warnings': risk_control_result['warnings'] + optimization_result['warnings'],
            'optimizations': optimization_result['optimizations'],
            'rule_count': risk_control_result['rule_count'] + optimization_result['rule_count']
        }
    
    def _select_trading_strategy(self, stock_code: str, current_price: float,
                                prediction: Dict[str, Any],
                                risk_assessment: Dict[str, Any],
                                rule_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        交易策略选择阶段
        
        复杂业务逻辑：
        - 根据风险等级、预测置信度、预期收益选择不同策略
        - 策略包括：激进策略、稳健策略、保守策略
        - 每种策略有不同的资金分配和风险控制参数
        """
        risk_score = risk_assessment['total_risk_score']
        confidence = prediction.get('confidence', 0.5)
        predicted_return = prediction.get('predicted_return', 0)
        
        # 策略评分系统
        strategy_scores = {
            'aggressive': 0,  # 激进策略
            'moderate': 0,    # 稳健策略
            'conservative': 0  # 保守策略
        }
        
        # 激进策略评分（高风险高收益）
        if risk_score > 0.6 and predicted_return > 0.05 and confidence > 0.7:
            strategy_scores['aggressive'] = (predicted_return * 2 + confidence - risk_score) * 0.5
        
        # 稳健策略评分（平衡风险收益）
        if 0.3 <= risk_score <= 0.7 and predicted_return > 0.02:
            strategy_scores['moderate'] = (predicted_return + confidence * 0.5 - abs(risk_score - 0.5)) * 0.8
        
        # 保守策略评分（低风险稳定）
        if risk_score < 0.4 and confidence > 0.6:
            strategy_scores['conservative'] = (confidence + (1 - risk_score) * 0.5) * 0.6
        
        # 选择得分最高的策略
        selected_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        strategy_name = selected_strategy[0]
        strategy_score = selected_strategy[1]
        
        # 策略参数配置
        strategies = {
            'aggressive': {
                'name': '激进策略',
                'description': '高风险高收益，适合高置信度强信号',
                'max_position_ratio': 0.35,
                'max_trade_ratio': 0.35,
                'stop_loss_ratio': -0.08,
                'take_profit_ratio': 0.15
            },
            'moderate': {
                'name': '稳健策略',
                'description': '平衡风险收益，适合中等市场环境',
                'max_position_ratio': 0.25,
                'max_trade_ratio': 0.25,
                'stop_loss_ratio': -0.05,
                'take_profit_ratio': 0.10
            },
            'conservative': {
                'name': '保守策略',
                'description': '低风险稳定收益，适合不确定性较高的市场',
                'max_position_ratio': 0.15,
                'max_trade_ratio': 0.15,
                'stop_loss_ratio': -0.03,
                'take_profit_ratio': 0.08
            }
        }
        
        strategy = strategies.get(strategy_name, strategies['moderate'])
        strategy['selected_score'] = strategy_score
        strategy['selection_reason'] = f'风险评分{risk_score:.2f}，置信度{confidence:.2f}，预期收益{predicted_return:.2%}'
        
        return strategy
    
    def _calculate_dynamic_trade_amount(self, stock_code: str, current_price: float,
                                       prediction: Dict[str, Any],
                                       risk_assessment: Dict[str, Any],
                                       strategy: Dict[str, Any],
                                       context: Dict[str, Any]) -> float:
        """
        动态资金计算阶段
        
        复杂业务逻辑：
        - 根据策略类型确定基础资金比例
        - 根据风险等级调整资金比例
        - 根据预测置信度和预期收益调整
        - 考虑已有持仓情况
        """
        available_cash = self.account['available_cash']
        total_assets = self.account['total_assets']
        
        # 基础资金比例（来自策略）
        base_ratio = strategy.get('max_trade_ratio', 0.25)
        
        # 风险调整系数
        risk_score = risk_assessment['total_risk_score']
        risk_adjustment = 1.0 - (risk_score * 0.3)  # 风险越高，资金越少
        
        # 置信度调整系数
        confidence = prediction.get('confidence', 0.5)
        confidence_adjustment = 0.7 + (confidence * 0.3)  # 置信度越高，资金越多
        
        # 预期收益调整系数
        predicted_return = abs(prediction.get('predicted_return', 0))
        return_adjustment = min(0.8 + (predicted_return * 4), 1.2)  # 收益越高，资金越多
        
        # 综合计算
        trade_ratio = base_ratio * risk_adjustment * confidence_adjustment * return_adjustment
        trade_ratio = min(trade_ratio, strategy.get('max_trade_ratio', 0.25))
        
        trade_amount = total_assets * trade_ratio
        trade_amount = min(trade_amount, available_cash)
        trade_amount = max(trade_amount, 0)
        
        return trade_amount
    
    def _generate_final_decision(self, stock_code: str, prediction: Dict[str, Any],
                                rule_result: Dict[str, Any],
                                risk_assessment: Dict[str, Any],
                                strategy: Dict[str, Any],
                                trade_amount: float,
                                current_price: float,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终决策"""
        
        # 根据预测结果决定基本操作
        predicted_return = prediction.get('predicted_return', 0)
        confidence = prediction.get('confidence', 0.5)
        
        # 基础决策
        if rule_result['is_allowed']:
            if predicted_return > 0.02 and confidence > 0.6:
                action = '买入'
                suggestion = f"采用{strategy['name']}，预测上涨概率{confidence:.1%}，预期收益率{predicted_return:.2%}，建议买入"
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
        
        # 生成决策理由
        reasoning_parts = [
            f"【决策流程】经过多阶段评估：",
            f"1. 预检查：通过基础条件验证",
            f"2. 风险评估：综合风险评分{risk_assessment['total_risk_score']:.2f}（{risk_assessment['risk_level']}风险）",
            f"3. 规则评估：触发{rule_result['rule_count']}条规则，规则链评估通过",
            f"4. 策略选择：选择{strategy['name']}（{strategy['description']}）",
            f"5. 资金计算：建议投资金额￥{trade_amount:.2f}",
            f"【最终建议】{suggestion}"
        ]
        
        return {
            'stock_code': stock_code,
            'action': action,
            'suggestion': suggestion,
            'current_price': current_price,
            'suggested_shares': shares,
            'suggested_amount': trade_amount,
            'prediction': prediction,
            'rule_evaluation': rule_result,
            'reasoning': "\n".join(reasoning_parts)
        }
    
    def _build_final_decision(self, stock_code: str, action: str, current_price: float,
                             reason: str, context: Dict[str, Any],
                             stage_details: List[Dict[str, Any]] = None,
                             risk_assessment: Dict[str, Any] = None) -> Dict[str, Any]:
        """构建最终决策（被禁止的情况）"""
        return {
            'stock_code': stock_code,
            'action': action,
            'suggestion': reason,
            'current_price': current_price,
            'suggested_shares': 0,
            'suggested_amount': 0,
            'reasoning': f"【决策流程】\n" + "\n".join([
                f"{i+1}. {stage['stage']}：{stage['result'].get('reason', '通过')}"
                for i, stage in enumerate(stage_details or [])
            ]) + f"\n【最终决定】{reason}",
            'decision_process': stage_details or [],
            'risk_assessment': risk_assessment
        }
    
    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取决策历史"""
        return self.decision_history[-limit:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取决策表现统计"""
        if not self.decision_history:
            return {}
        
        total_decisions = len(self.decision_history)
        actions = [d['action'] for d in self.decision_history]
        
        return {
            'total_decisions': total_decisions,
            'buy_count': actions.count('买入'),
            'sell_count': actions.count('卖出'),
            'hold_count': actions.count('持有'),
            'ban_count': actions.count('禁止'),
            'avg_risk_score': sum(d.get('risk_score', 0) for d in self.decision_history) / total_decisions
        }

