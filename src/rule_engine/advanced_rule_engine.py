"""
增强的规则引擎 - 支持复杂条件表达式
"""
import json
import re
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from .rule import BusinessRule
from .rule_engine import RuleEngine

# 配置日志记录器
logger = logging.getLogger(__name__)


class AdvancedRuleEngine(RuleEngine):
    """增强的规则引擎，支持复杂条件表达式"""
    
    def __init__(self, rules_file: str = None):
        super().__init__(rules_file)
        self.rule_history = []  # 规则执行历史
    
    def parse_condition(self, condition_str: str, context: Dict[str, Any], depth: int = 0) -> bool:
        """
        解析复杂条件表达式
        
        支持：
        - 简单比较: field > value
        - AND逻辑: condition1 AND condition2
        - OR逻辑: condition1 OR condition2
        - 括号: (condition1 AND condition2) OR condition3
        
        Args:
            condition_str: 条件表达式字符串
            context: 评估上下文
            depth: 递归深度（防止无限递归）
        
        Returns:
            条件是否满足
        """
        # 调试信息
        if depth == 0:
            logger.info(f"parse_condition 开始: {condition_str}")
        else:
            logger.info(f"parse_condition 递归 (depth={depth}): {condition_str}")
        
        # 防止递归深度过深
        if depth > 10:
            logger.error(f"递归深度过深(depth={depth})，停止解析: {condition_str}")
            return False
        
        try:
            # 清理输入字符串
            condition_str = condition_str.strip()
            if not condition_str:
                logger.info(f"parse_condition: 空字符串，返回False")
                return False
            
            # 处理布尔值字符串（避免重复处理）
            if condition_str.lower() == 'true':
                logger.info(f"parse_condition (depth={depth}): 布尔值True，返回True")
                return True
            if condition_str.lower() == 'false':
                logger.info(f"parse_condition (depth={depth}): 布尔值False，返回False")
                return False
            
            # 处理括号（只在有括号时处理）
            if '(' in condition_str:
                logger.info(f"parse_condition (depth={depth}): 发现括号，调用_process_parentheses")
                condition_str, had_parentheses = self._process_parentheses(condition_str, context, depth)
                logger.info(f"parse_condition (depth={depth}): 括号处理后: {condition_str}")
                # 括号处理后，检查是否已经是布尔值
                condition_str = condition_str.strip()
                if condition_str.lower() == 'true':
                    logger.info(f"parse_condition (depth={depth}): 括号处理后为True，返回True")
                    return True
                if condition_str.lower() == 'false':
                    logger.info(f"parse_condition (depth={depth}): 括号处理后为False，返回False")
                    return False
            
            # 处理AND和OR（括号处理完后，递归处理剩余的AND/OR）
            # 注意：这里需要递归调用 parse_condition 来处理嵌套的 AND/OR
            # 但需要确保深度限制有效
            
            # 处理AND（必须在OR之前，因为可能有 "A AND B OR C"）
            if ' AND ' in condition_str:
                # 检查是否包含OR，如果有，需要按优先级处理
                if ' OR ' in condition_str:
                    # 先处理OR，因为OR优先级更低（需要递归处理）
                    parts = condition_str.split(' OR ')
                    bool_parts = []
                    for part in parts:
                        part = part.strip()
                        if part.lower() == 'true':
                            bool_parts.append(True)
                        elif part.lower() == 'false':
                            bool_parts.append(False)
                        elif ' AND ' in part or ' OR ' in part:
                            # 如果部分还包含逻辑运算符，递归处理（但深度+1）
                            bool_parts.append(self.parse_condition(part, context, depth + 1))
                        else:
                            # 简单条件
                            bool_parts.append(self._evaluate_simple_condition(part, context))
                    return any(bool_parts)
                else:
                    # 只有AND
                    parts = condition_str.split(' AND ')
                    bool_parts = []
                    for part in parts:
                        part = part.strip()
                        if part.lower() == 'true':
                            bool_parts.append(True)
                        elif part.lower() == 'false':
                            bool_parts.append(False)
                        elif ' OR ' in part:
                            # 如果部分包含OR，递归处理（但深度+1）
                            bool_parts.append(self.parse_condition(part, context, depth + 1))
                        else:
                            # 简单条件
                            bool_parts.append(self._evaluate_simple_condition(part, context))
                    return all(bool_parts)
            
            # 处理OR
            if ' OR ' in condition_str:
                parts = condition_str.split(' OR ')
                bool_parts = []
                for part in parts:
                    part = part.strip()
                    if part.lower() == 'true':
                        bool_parts.append(True)
                    elif part.lower() == 'false':
                        bool_parts.append(False)
                    elif ' AND ' in part:
                        # 如果部分包含AND，递归处理（但深度+1）
                        bool_parts.append(self.parse_condition(part, context, depth + 1))
                    else:
                        # 简单条件
                        bool_parts.append(self._evaluate_simple_condition(part, context))
                return any(bool_parts)
            
            # 简单条件
            return self._evaluate_simple_condition(condition_str, context)
            
        except RecursionError as e:
            logger.error(f"递归深度超限，停止解析: {condition_str}, depth={depth}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"解析条件表达式失败: {e}, condition={condition_str}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _process_parentheses(self, condition_str: str, context: Dict[str, Any], depth: int = 0) -> tuple:
        """
        处理括号，递归计算括号内的表达式
        
        Returns:
            (处理后的字符串, 是否处理过括号)
        """
        logger.info(f"_process_parentheses (depth={depth}) 开始: {condition_str}")
        max_depth = 10  # 最大递归深度
        if depth > max_depth:
            logger.error(f"_process_parentheses: 递归深度超限 (depth={depth})")
            return condition_str, False
        
        had_parentheses = False
        max_iterations = 50  # 防止无限循环
        iteration = 0
        
        while '(' in condition_str and iteration < max_iterations:
            iteration += 1
            logger.info(f"_process_parentheses (depth={depth}) 迭代 {iteration}: 处理 {condition_str}")
            had_parentheses = True
            
            # 找到最内层的括号
            start = condition_str.rfind('(')
            end = condition_str.find(')', start)
            
            if end == -1:
                logger.warning(f"_process_parentheses: 找不到匹配的右括号")
                break
            
            # 提取括号内的表达式
            inner_expr = condition_str[start + 1:end].strip()
            logger.info(f"_process_parentheses: 提取括号内容: {inner_expr}")
            
            # 如果内层表达式为空，跳过
            if not inner_expr:
                condition_str = condition_str[:start] + condition_str[end + 1:]
                continue
            
            # 检查括号内容是否是数学表达式（如 total_assets * 0.2）还是逻辑表达式
            # 如果包含比较运算符(> < >= <= == !=)，则可能是逻辑表达式
            # 如果只包含数学运算符(* / + -)，则只是数学表达式，直接计算
            has_comparison = any(op in inner_expr for op in ['>', '<', '>=', '<=', '==', '!='])
            has_logical = any(op in inner_expr for op in [' AND ', ' OR '])
            
            logger.info(f"_process_parentheses: 括号内容类型判断 - 比较运算符: {has_comparison}, 逻辑运算符: {has_logical}")
            
            if has_comparison or has_logical:
                # 是逻辑表达式，递归调用 parse_condition
                logger.info(f"_process_parentheses: 识别为逻辑表达式，调用 parse_condition")
                inner_result = self.parse_condition(inner_expr, context, depth + 1)
                # 替换括号表达式为布尔值字符串
                replacement = 'True' if inner_result else 'False'
                condition_str = condition_str[:start] + replacement + condition_str[end + 1:]
                logger.info(f"_process_parentheses: 替换为 {replacement}, 结果: {condition_str}")
            else:
                # 只是数学表达式（如 total_assets * 0.2），直接计算
                logger.info(f"_process_parentheses: 识别为数学表达式，调用 _evaluate_expression")
                try:
                    inner_value = self._evaluate_expression(inner_expr, context, depth + 1)
                    # 替换括号表达式为计算结果（数字字符串）
                    replacement = str(inner_value)
                    condition_str = condition_str[:start] + replacement + condition_str[end + 1:]
                    logger.info(f"_process_parentheses: 计算结果 {inner_value}, 替换后: {condition_str}")
                except Exception as e:
                    logger.error(f"计算括号内数学表达式失败: {inner_expr}, 错误: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # 如果计算失败，替换为 False
                    condition_str = condition_str[:start] + 'False' + condition_str[end + 1:]
        
        if iteration >= max_iterations:
            logger.error(f"括号处理迭代次数过多，停止处理: {condition_str}")
        
        logger.info(f"_process_parentheses (depth={depth}) 完成: {condition_str}")
        return condition_str, had_parentheses
    
    def _evaluate_simple_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """评估简单条件（单个比较）"""
        # 处理布尔值字符串
        condition = condition.strip()
        if condition.lower() == 'true':
            return True
        if condition.lower() == 'false':
            return False
        
        # 如果包含括号，不应该在这里处理（应该在parse_condition中处理）
        if '(' in condition or ')' in condition:
            print(f"警告：简单条件中包含括号，应该先处理: {condition}")
            return False
        
        # 先处理数学表达式（如 total_assets * 0.2）
        # 查找比较运算符的位置
        import re
        operator_match = re.search(r'\s*(>=|<=|>|<|==|!=)\s*', condition)
        if not operator_match:
            return False
        
        operator = operator_match.group(1)
        operator_pos = operator_match.start()
        
        left_expr = condition[:operator_pos].strip()
        right_expr = condition[operator_match.end():].strip()
        
        # 计算左侧表达式（可能是变量或数学表达式）
        try:
            left_value = self._evaluate_expression(left_expr, context, depth=0)
        except Exception as e:
            print(f"计算左侧表达式失败: {left_expr}, 错误: {e}")
            return False
        
        # 计算右侧表达式（可能是数字、变量或数学表达式）
        try:
            right_value = self._evaluate_expression(right_expr, context, depth=0)
        except Exception as e:
            print(f"计算右侧表达式失败: {right_expr}, 错误: {e}")
            return False
        
        # 比较
        try:
            if operator == '>':
                return left_value > right_value
            elif operator == '<':
                return left_value < right_value
            elif operator == '>=':
                return left_value >= right_value
            elif operator == '<=':
                return left_value <= right_value
            elif operator == '==':
                return left_value == right_value
            elif operator == '!=':
                return left_value != right_value
        except:
            return False
        
        return False
    
    def _evaluate_expression(self, expr: str, context: Dict[str, Any], depth: int = 0) -> float:
        """
        计算表达式值（支持简单的数学运算）
        支持：变量、数字、乘法、除法、加法、减法
        例如：total_assets * 0.2, position_ratio + 0.1
        
        Args:
            expr: 表达式字符串
            context: 上下文
            depth: 递归深度（防止无限递归）
        """
        logger.info(f"_evaluate_expression (depth={depth}): {expr}")
        # 防止递归深度过深
        if depth > 10:
            logger.error(f"_evaluate_expression: 递归深度超限 (depth={depth}): {expr}")
            raise ValueError(f"表达式递归深度过深: {expr}")
        
        expr = expr.strip()
        
        # 如果是纯数字
        try:
            if expr.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
                result = float(expr)
                logger.info(f"_evaluate_expression (depth={depth}): {expr} 是数字，返回 {result}")
                return result
        except:
            pass
        
        # 如果是变量引用
        if expr in context:
            value = context[expr]
            if isinstance(value, (int, float)):
                result = float(value)
                logger.info(f"_evaluate_expression (depth={depth}): {expr} 是变量，返回 {result}")
                return result
            logger.info(f"_evaluate_expression (depth={depth}): {expr} 是变量但值不是数字，返回 0.0")
            return 0.0
        
        # 处理数学表达式（简单支持：* / + -）
        # 注意：按优先级处理，先乘除后加减
        
        # 先处理乘法和除法（从左到右）
        # 必须按照运算符优先级和结合性处理
        if '*' in expr or '/' in expr:
            # 找到第一个 * 或 /
            mult_pos = expr.find('*')
            div_pos = expr.find('/')
            
            # 选择第一个出现的运算符
            if mult_pos != -1 and (div_pos == -1 or mult_pos < div_pos):
                # 处理乘法（只分割第一个*，这样可以从左到右处理）
                parts = expr.split('*', 1)
                if len(parts) == 2:
                    left_str = parts[0].strip()
                    right_str = parts[1].strip()
                    logger.info(f"_evaluate_expression (depth={depth}): 乘法分割 - 左: {left_str}, 右: {right_str}")
                    # 如果分割后的部分为空或和原表达式相同，说明有问题，避免无限递归
                    if not left_str or not right_str or left_str == expr or right_str == expr:
                        logger.error(f"_evaluate_expression: 表达式分割错误: {expr} -> [{left_str}, {right_str}]")
                        raise ValueError(f"表达式分割错误: {expr}")
                    left = self._evaluate_expression(left_str, context, depth + 1)
                    right = self._evaluate_expression(right_str, context, depth + 1)
                    result = left * right
                    logger.info(f"_evaluate_expression (depth={depth}): {expr} = {left} * {right} = {result}")
                    return result
            elif div_pos != -1:
                # 处理除法
                parts = expr.split('/', 1)
                if len(parts) == 2:
                    left_str = parts[0].strip()
                    right_str = parts[1].strip()
                    logger.info(f"_evaluate_expression (depth={depth}): 除法分割 - 左: {left_str}, 右: {right_str}")
                    if not left_str or not right_str or left_str == expr or right_str == expr:
                        logger.error(f"_evaluate_expression: 表达式分割错误: {expr} -> [{left_str}, {right_str}]")
                        raise ValueError(f"表达式分割错误: {expr}")
                    left = self._evaluate_expression(left_str, context, depth + 1)
                    right = self._evaluate_expression(right_str, context, depth + 1)
                    if right == 0:
                        logger.warning(f"_evaluate_expression: 除数为0，返回0.0")
                        return 0.0
                    result = left / right
                    logger.info(f"_evaluate_expression (depth={depth}): {expr} = {left} / {right} = {result}")
                    return result
        
        # 再处理加法和减法（从左到右）
        if '+' in expr and not expr.startswith('+'):
            parts = expr.split('+', 1)
            if len(parts) == 2:
                left_str = parts[0].strip()
                right_str = parts[1].strip()
                logger.info(f"_evaluate_expression (depth={depth}): 加法分割 - 左: {left_str}, 右: {right_str}")
                if not left_str or not right_str or left_str == expr or right_str == expr:
                    logger.error(f"_evaluate_expression: 表达式分割错误: {expr} -> [{left_str}, {right_str}]")
                    raise ValueError(f"表达式分割错误: {expr}")
                left = self._evaluate_expression(left_str, context, depth + 1)
                right = self._evaluate_expression(right_str, context, depth + 1)
                result = left + right
                logger.info(f"_evaluate_expression (depth={depth}): {expr} = {left} + {right} = {result}")
                return result
        
        if '-' in expr and not expr.startswith('-'):
            # 避免处理负数
            parts = expr.split('-', 1)
            if len(parts) == 2:
                left_str = parts[0].strip()
                right_str = parts[1].strip()
                logger.info(f"_evaluate_expression (depth={depth}): 减法分割 - 左: {left_str}, 右: {right_str}")
                if not left_str or not right_str or left_str == expr or right_str == expr:
                    logger.error(f"_evaluate_expression: 表达式分割错误: {expr} -> [{left_str}, {right_str}]")
                    raise ValueError(f"表达式分割错误: {expr}")
                left = self._evaluate_expression(left_str, context, depth + 1)
                right = self._evaluate_expression(right_str, context, depth + 1)
                result = left - right
                logger.info(f"_evaluate_expression (depth={depth}): {expr} = {left} - {right} = {result}")
                return result
        
        # 如果无法解析，尝试作为变量
        if expr in context:
            value = context[expr]
            if isinstance(value, (int, float)):
                result = float(value)
                logger.info(f"_evaluate_expression (depth={depth}): {expr} 是变量（第二次检查），返回 {result}")
                return result
            logger.info(f"_evaluate_expression (depth={depth}): {expr} 是变量但值不是数字，返回 0.0")
            return 0.0
        
        # 如果表达式是布尔值字符串，不应该在这里处理
        if expr.strip().lower() in ['true', 'false']:
            logger.error(f"_evaluate_expression (depth={depth}): 表达式不应包含布尔值字符串: {expr}")
            raise ValueError(f"表达式不应包含布尔值字符串: {expr}")
        
        # 如果表达式看起来像变量名（只包含字母、数字、下划线），但不在context中，返回0.0
        if expr.replace('_', '').isalnum() and not expr.replace('.', '').replace('-', '').isdigit():
            logger.warning(f"_evaluate_expression (depth={depth}): 变量 {expr} 不在context中，返回 0.0")
            return 0.0
        
        # 最后尝试转换为数字
        try:
            result = float(expr)
            logger.info(f"_evaluate_expression (depth={depth}): {expr} 转换为数字: {result}")
            return result
        except:
            logger.error(f"_evaluate_expression (depth={depth}): 无法计算表达式: {expr}")
            # 如果无法转换，返回0.0而不是抛出异常，避免导致递归错误
            logger.warning(f"_evaluate_expression (depth={depth}): 无法计算表达式 {expr}，返回默认值 0.0")
            return 0.0
    
    def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估所有规则（增强版，支持复杂条件）"""
        triggered_rules = []
        warnings = []
        optimizations = []
        is_allowed = True
        reason = ""
        conflict_rules = []
        
        # 遍历所有规则（按优先级）
        for rule in self.rules:
            # 检查条件（支持复杂表达式）
            condition_met = False
            
            # 获取条件（可能是dict或str）
            condition = rule.condition
            
            if isinstance(condition, dict):
                # 旧格式：简单条件
                condition_met = rule.evaluate_condition(context)
            elif isinstance(condition, str):
                # 新格式：复杂表达式
                logger.info(f"evaluate_all: 评估规则 {rule.rule_id} ({rule.name}) 的条件: {condition}")
                try:
                    condition_met = self.parse_condition(condition, context, depth=0)
                    logger.info(f"evaluate_all: 规则 {rule.rule_id} 条件结果: {condition_met}")
                except RecursionError as e:
                    logger.error(f"规则 {rule.rule_id} ({rule.name}) 的条件解析递归错误: {condition}")
                    import traceback
                    logger.error(traceback.format_exc())
                    condition_met = False
                except Exception as e:
                    logger.error(f"规则 {rule.rule_id} ({rule.name}) 的条件解析失败: {condition}, 错误: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    condition_met = False
            else:
                continue
            
            if condition_met:
                triggered_rules.append(rule)
                action = rule.get_action()
                action_type = action.get('type', '')
                action_message = action.get('message', '')
                
                # 记录规则执行
                self.rule_history.append({
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'condition': str(rule.condition),
                    'action': action_type,
                    'timestamp': str(pd.Timestamp.now())
                })
                
                # 根据动作类型处理
                if action_type == '禁止':
                    if is_allowed:  # 如果之前是允许的，现在被禁止
                        conflict_rules.append(rule)
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
        
        # 冲突检测
        if len(conflict_rules) > 1:
            # 按优先级解决冲突
            conflict_rules.sort(key=lambda x: x.priority)
            reason = f"规则冲突：{conflict_rules[0].name}（优先级最高）"
        
        return {
            'is_allowed': is_allowed,
            'reason': reason,
            'triggered_rules': [rule.to_dict() for rule in triggered_rules],
            'warnings': warnings,
            'optimizations': optimizations,
            'rule_count': len(triggered_rules),
            'conflicts': [r.to_dict() for r in conflict_rules] if conflict_rules else []
        }
    
    def add_rule_from_string(self, rule_id: str, rule_type: str, name: str, 
                            condition_str: str, action: Dict[str, Any], priority: int = 999):
        """从字符串添加规则（支持复杂条件）"""
        rule_data = {
            'rule_id': rule_id,
            'type': rule_type,
            'name': name,
            'description': f"条件: {condition_str}",
            'condition': condition_str,  # 直接使用字符串
            'action': action,
            'priority': priority
        }
        
        rule = BusinessRule(rule_data)
        # 修改rule的condition为字符串格式
        rule.condition = condition_str
        self.add_rule(rule)
    
    def get_rule_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取规则执行历史"""
        return self.rule_history[-limit:]

