"""
规则可视化模块
"""
import json
from typing import Dict, Any, List
from ..rule_engine.rule_engine import RuleEngine


class RuleVisualizer:
    """规则可视化器"""
    
    def __init__(self, rule_engine: RuleEngine):
        """
        初始化可视化器
        
        Args:
            rule_engine: 规则引擎实例
        """
        self.rule_engine = rule_engine
    
    def generate_rule_info(self) -> Dict[str, Any]:
        """生成规则信息（用于前端展示）"""
        all_rules = self.rule_engine.get_all_rules()
        
        # 按类型分组
        rules_by_type = {}
        for rule in all_rules:
            rule_type = rule.get('type', '其他')
            if rule_type not in rules_by_type:
                rules_by_type[rule_type] = []
            rules_by_type[rule_type].append(rule)
        
        return {
            'total_rules': len(all_rules),
            'rules_by_type': rules_by_type,
            'all_rules': all_rules
        }
    
    def generate_evaluation_flow(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成规则评估流程可视化数据（显示所有规则，匹配的标记）
        
        Args:
            evaluation_result: 规则评估结果
        
        Returns:
            可视化数据
        """
        triggered_rules = evaluation_result.get('triggered_rules', [])
        # 提取触发的规则ID（triggered_rules可能是字典列表）
        triggered_rule_ids = set()
        for rule in triggered_rules:
            if isinstance(rule, dict):
                rule_id = rule.get('rule_id', '')
            else:
                rule_id = rule.rule_id if hasattr(rule, 'rule_id') else ''
            if rule_id:
                triggered_rule_ids.add(rule_id)
        
        # 获取所有规则（返回字典列表）
        all_rules = self.rule_engine.get_all_rules()
        
        flow_data = {
            'total_rules': len(all_rules),
            'triggered_rules': len(triggered_rules),
            'is_allowed': evaluation_result.get('is_allowed', True),
            'reason': evaluation_result.get('reason', ''),
            'steps': []
        }
        
        # 添加所有规则，标记哪些被触发
        for rule in all_rules:
            # rule是字典格式
            rule_id = rule.get('rule_id', '')
            is_triggered = rule_id in triggered_rule_ids
            
            flow_data['steps'].append({
                'rule_id': rule_id,
                'rule_name': rule.get('name', ''),
                'rule_type': rule.get('type', ''),
                'condition': rule.get('condition', {}),
                'action': rule.get('action', {}),
                'description': rule.get('description', ''),
                'is_triggered': is_triggered
            })
        
        return flow_data
    
    def format_decision_explanation(self, decision: Dict[str, Any]) -> str:
        """
        格式化决策解释（用于文本展示）
        
        Args:
            decision: 决策结果
        
        Returns:
            格式化的解释文本
        """
        lines = []
        
        # 基本信息
        lines.append(f"【决策结果】{decision.get('action', '未知')}")
        lines.append(f"股票代码：{decision.get('stock_code', 'N/A')}")
        lines.append(f"当前价格：￥{decision.get('current_price', 0):.2f}")
        
        if decision.get('suggested_shares', 0) > 0:
            lines.append(f"建议数量：{decision.get('suggested_shares', 0)}股")
            lines.append(f"建议金额：￥{decision.get('suggested_amount', 0):.2f}")
        
        # 决策理由
        reasoning = decision.get('reasoning', '')
        if reasoning:
            lines.append("\n【决策理由】")
            lines.append(reasoning)
        
        # 规则评估详情
        rule_eval = decision.get('rule_evaluation', {})
        if rule_eval.get('triggered_rules'):
            lines.append(f"\n【规则评估】共触发{rule_eval.get('rule_count', 0)}条规则")
            
            if rule_eval.get('warnings'):
                lines.append("\n警告信息：")
                for warning in rule_eval['warnings']:
                    lines.append(f"  ⚠ {warning['message']}")
            
            if rule_eval.get('optimizations'):
                lines.append("\n优化建议：")
                for opt in rule_eval['optimizations']:
                    lines.append(f"  💡 {opt['message']}")
        
        return "\n".join(lines)

