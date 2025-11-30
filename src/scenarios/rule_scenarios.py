"""
规则触发场景演示模块
"""
from typing import Dict, Any, List
from ..rule_engine.advanced_rule_engine import AdvancedRuleEngine
from config.settings import RULES_FILE


class RuleScenarioDemo:
    """规则场景演示类"""
    
    def __init__(self):
        self.rule_engine = AdvancedRuleEngine(rules_file=RULES_FILE)
    
    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """获取所有预设场景"""
        return [
            self.scenario_large_trade(),
            self.scenario_high_risk_low_confidence(),
            self.scenario_high_confidence_high_return(),
            self.scenario_stop_loss(),
            self.scenario_take_profit(),
            self.scenario_position_limit(),
            self.scenario_insufficient_funds(),
            self.scenario_daily_limit(),
            self.scenario_trading_hours(),
        ]
    
    def scenario_large_trade(self) -> Dict[str, Any]:
        """场景1：大额交易成本优化"""
        return {
            'id': 'scenario_1',
            'name': '大额交易成本优化',
            'description': '交易金额超过10万元时，触发成本优化规则',
            'context': {
                'stock_code': '000560',
                'current_price': 10.0,
                'trade_amount': 150000,  # 15万元
                'position_ratio': 0.1,
                'available_cash': 200000,
                'total_assets': 500000,
                'predicted_return': 0.03,
                'confidence': 0.7,
                'risk_level': '中',
                'current_hour': 10
            },
            'expected_rules': ['CO001', 'TW001']
        }
    
    def scenario_high_risk_low_confidence(self) -> Dict[str, Any]:
        """场景2：高风险低置信度"""
        return {
            'id': 'scenario_2',
            'name': '高风险市场暂停',
            'description': '预测风险等级为高且置信度低于60%时，建议观望',
            'context': {
                'stock_code': '000560',
                'current_price': 10.0,
                'trade_amount': 10000,
                'position_ratio': 0.05,
                'available_cash': 50000,
                'total_assets': 100000,
                'predicted_return': 0.02,
                'confidence': 0.5,  # 低置信度
                'risk_level': '高',  # 高风险
                'current_hour': 10
            },
            'expected_rules': ['MS001', 'TW001']
        }
    
    def scenario_high_confidence_high_return(self) -> Dict[str, Any]:
        """场景3：高置信度高收益"""
        return {
            'id': 'scenario_3',
            'name': '高置信度优化',
            'description': '预测收益率超过5%且置信度超过80%时，建议增加投资',
            'context': {
                'stock_code': '000560',
                'current_price': 10.0,
                'trade_amount': 10000,
                'position_ratio': 0.05,
                'available_cash': 50000,
                'total_assets': 100000,
                'predicted_return': 0.06,  # 高收益率
                'confidence': 0.85,  # 高置信度
                'risk_level': '中',
                'current_hour': 10
            },
            'expected_rules': ['CO002', 'TW001']
        }
    
    def scenario_stop_loss(self) -> Dict[str, Any]:
        """场景4：止损规则"""
        return {
            'id': 'scenario_4',
            'name': '止损规则',
            'description': '持仓亏损超过5%时，建议止损卖出',
            'context': {
                'stock_code': '000560',
                'current_price': 9.0,  # 当前价9元
                'trade_amount': 10000,
                'position_ratio': 0.15,
                'available_cash': 50000,
                'total_assets': 100000,
                'predicted_return': 0.01,
                'confidence': 0.6,
                'risk_level': '中',
                'current_hour': 10,
                'loss_ratio': -0.1,  # 亏损10%（成本价10元，当前价9元）
                'total_loss_ratio': -0.015
            },
            'expected_rules': ['RC002', 'TW001']
        }
    
    def scenario_take_profit(self) -> Dict[str, Any]:
        """场景5：止盈规则"""
        return {
            'id': 'scenario_5',
            'name': '止盈规则',
            'description': '持仓盈利超过10%且持仓比例超过10%时，建议部分止盈',
            'context': {
                'stock_code': '000560',
                'current_price': 11.5,  # 当前价11.5元
                'trade_amount': 10000,
                'position_ratio': 0.15,  # 持仓比例15%
                'available_cash': 50000,
                'total_assets': 100000,
                'predicted_return': 0.01,
                'confidence': 0.6,
                'risk_level': '中',
                'current_hour': 10,
                'loss_ratio': 0.15,  # 盈利15%（成本价10元，当前价11.5元）
                'total_loss_ratio': 0.02
            },
            'expected_rules': ['RC005', 'TW001']
        }
    
    def scenario_position_limit(self) -> Dict[str, Any]:
        """场景6：持仓限制"""
        return {
            'id': 'scenario_6',
            'name': '单只股票持仓限制',
            'description': '单只股票持仓比例超过30%时，禁止继续买入',
            'context': {
                'stock_code': '000560',
                'current_price': 10.0,
                'trade_amount': 20000,
                'position_ratio': 0.35,  # 持仓比例35%，超过30%
                'available_cash': 30000,
                'total_assets': 100000,
                'predicted_return': 0.03,
                'confidence': 0.7,
                'risk_level': '中',
                'current_hour': 10
            },
            'expected_rules': ['RC001', 'TW001']
        }
    
    def scenario_insufficient_funds(self) -> Dict[str, Any]:
        """场景7：资金不足"""
        return {
            'id': 'scenario_7',
            'name': '资金不足限制',
            'description': '可用资金不足1000元时，无法开新仓',
            'context': {
                'stock_code': '000560',
                'current_price': 10.0,
                'trade_amount': 5000,
                'position_ratio': 0.05,
                'available_cash': 500,  # 可用资金只有500元
                'total_assets': 100000,
                'predicted_return': 0.03,
                'confidence': 0.7,
                'risk_level': '中',
                'current_hour': 10
            },
            'expected_rules': ['RC004', 'TW001']
        }
    
    def scenario_daily_limit(self) -> Dict[str, Any]:
        """场景8：单日限额"""
        return {
            'id': 'scenario_8',
            'name': '单日交易限额',
            'description': '单日交易金额超过总资产20%时，禁止交易',
            'context': {
                'stock_code': '000560',
                'current_price': 10.0,
                'trade_amount': 25000,  # 2.5万，超过总资产20%（2万）
                'position_ratio': 0.1,
                'available_cash': 30000,
                'total_assets': 100000,
                'predicted_return': 0.03,
                'confidence': 0.7,
                'risk_level': '中',
                'current_hour': 10
            },
            'expected_rules': ['RC006', 'TW001']
        }
    
    def scenario_trading_hours(self) -> Dict[str, Any]:
        """场景9：交易时间"""
        return {
            'id': 'scenario_9',
            'name': '交易时间限制',
            'description': '在交易时间内（9:00-15:59）允许交易',
            'context': {
                'stock_code': '000560',
                'current_price': 10.0,
                'trade_amount': 10000,
                'position_ratio': 0.05,
                'available_cash': 50000,
                'total_assets': 100000,
                'predicted_return': 0.03,
                'confidence': 0.7,
                'risk_level': '中',
                'current_hour': 14  # 下午2点，在交易时间内
            },
            'expected_rules': ['TW001']
        }
    
    def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个场景"""
        context = scenario['context']
        result = self.rule_engine.evaluate_all(context)
        
        triggered_rule_ids = [rule['rule_id'] for rule in result['triggered_rules']]
        expected_rule_ids = scenario.get('expected_rules', [])
        
        return {
            'scenario': scenario,
            'result': result,
            'triggered_rules': triggered_rule_ids,
            'expected_rules': expected_rule_ids,
            'match': set(triggered_rule_ids) == set(expected_rule_ids),
            'is_allowed': result['is_allowed'],
            'reason': result['reason']
        }
    
    def run_all_scenarios(self) -> List[Dict[str, Any]]:
        """运行所有场景"""
        scenarios = self.get_all_scenarios()
        results = []
        
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)
        
        return results


