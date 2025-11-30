"""
规则引擎模块 - 规则引擎核心
"""
import json
from typing import List, Dict, Any
from pathlib import Path
from .rule import BusinessRule, RuleType, RuleAction


class RuleEngine:
    """业务规则引擎"""
    
    def __init__(self, rules_file: str = None):
        """
        初始化规则引擎
        
        Args:
            rules_file: 规则文件路径
        """
        self.rules: List[BusinessRule] = []
        self.rules_file = rules_file
        if rules_file:
            self.load_rules(rules_file)
    
    def load_rules(self, rules_file: str):
        """从文件加载规则"""
        try:
            file_path = Path(rules_file)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                    rules_list = rules_data.get('rules', [])
                    self.rules = [BusinessRule(rule_data) for rule_data in rules_list]
                    # 按优先级排序
                    self.rules.sort(key=lambda x: x.priority)
            else:
                print(f"规则文件不存在: {rules_file}")
        except Exception as e:
            print(f"加载规则文件失败: {e}")
    
    def add_rule(self, rule: BusinessRule):
        """添加规则"""
        self.rules.append(rule)
        self.rules.sort(key=lambda x: x.priority)
    
    def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估所有规则
        
        Args:
            context: 评估上下文
        
        Returns:
            评估结果，包含触发的规则列表、是否允许、建议等
        """
        triggered_rules = []
        warnings = []
        optimizations = []
        is_allowed = True
        reason = ""
        
        # 遍历所有规则（按优先级）
        for rule in self.rules:
            if rule.evaluate_condition(context):
                triggered_rules.append(rule)
                action = rule.get_action()
                action_type = action.get('type', '')
                action_message = action.get('message', '')
                
                # 根据动作类型处理
                if action_type == '禁止':
                    is_allowed = False
                    reason = action_message
                elif action_type == '警告':
                    warnings.append({
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'message': action_message
                    })
                elif action_type == '优化':
                    optimizations.append({
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'message': action_message
                    })
                elif action_type == '建议卖出' or action_type == '建议':
                    warnings.append({
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'message': action_message
                    })
        
        return {
            'is_allowed': is_allowed,
            'reason': reason,
            'triggered_rules': [rule.to_dict() for rule in triggered_rules],
            'warnings': warnings,
            'optimizations': optimizations,
            'rule_count': len(triggered_rules)
        }
    
    def get_rules_by_type(self, rule_type: str) -> List[BusinessRule]:
        """根据类型获取规则"""
        return [rule for rule in self.rules if rule.type == rule_type]
    
    def get_all_rules(self) -> List[Dict[str, Any]]:
        """获取所有规则"""
        return [rule.to_dict() for rule in self.rules]


