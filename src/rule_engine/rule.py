"""
规则引擎模块 - 规则定义
"""
from typing import Dict, Any, Callable
from enum import Enum


class RuleType(Enum):
    """规则类型枚举"""
    RISK_CONTROL = "风控规则"
    COST_OPTIMIZATION = "成本优化规则"
    TIME_WINDOW = "时间窗口规则"
    MARKET_STATE = "市场状态规则"
    ACCOUNT_STATE = "账户状态规则"


class RuleAction(Enum):
    """规则动作类型"""
    ALLOW = "允许"
    FORBID = "禁止"
    WARN = "警告"
    OPTIMIZE = "优化"
    SUGGEST = "建议"


class BusinessRule:
    """业务规则类"""
    
    def __init__(self, rule_data: Dict[str, Any]):
        """
        初始化业务规则
        
        Args:
            rule_data: 规则数据字典，包含rule_id, type, name, condition, action, priority等
        """
        self.rule_id = rule_data.get('rule_id')
        self.type = rule_data.get('type', '')
        self.name = rule_data.get('name', '')
        self.description = rule_data.get('description', '')
        self.condition = rule_data.get('condition', {})
        self.action = rule_data.get('action', {})
        self.priority = rule_data.get('priority', 999)
    
    def evaluate_condition(self, context: Dict[str, Any]) -> bool:
        """
        评估规则条件是否满足
        
        Args:
            context: 评估上下文，包含各种数据字段
        
        Returns:
            条件是否满足
        """
        if not self.condition:
            return False
        
        # 如果condition是字符串，需要由AdvancedRuleEngine处理
        if isinstance(self.condition, str):
            return False  # 简单规则类不支持字符串条件
        
        # 处理字典格式的条件
        field = self.condition.get('field')
        operator = self.condition.get('operator')
        value = self.condition.get('value')
        
        if field not in context:
            return False
        
        field_value = context[field]
        
        # 根据操作符进行比较
        try:
            if operator == '>':
                return field_value > value
            elif operator == '<':
                return field_value < value
            elif operator == '>=':
                return field_value >= value
            elif operator == '<=':
                return field_value <= value
            elif operator == '==':
                return field_value == value
            elif operator == '!=':
                return field_value != value
            else:
                return False
        except Exception as e:
            print(f"评估规则条件时出错: {e}")
            return False
    
    def get_action(self) -> Dict[str, Any]:
        """获取规则动作"""
        return self.action
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rule_id': self.rule_id,
            'type': self.type,
            'name': self.name,
            'description': self.description,
            'condition': self.condition,
            'action': self.action,
            'priority': self.priority
        }
    
    def __repr__(self):
        return f"<BusinessRule {self.rule_id}: {self.name}>"

