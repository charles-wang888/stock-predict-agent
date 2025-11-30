"""
规则触发测试脚本
用于测试各种规则触发场景
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径，以便导入src和config模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rule_engine.advanced_rule_engine import AdvancedRuleEngine
from config.settings import RULES_FILE

def test_rule_scenarios():
    """测试各种规则触发场景"""
    
    # 初始化规则引擎
    rule_engine = AdvancedRuleEngine(RULES_FILE)
    
    print("=" * 60)
    print("规则触发测试")
    print("=" * 60)
    print()
    
    # 场景1：大额交易（触发CO001）
    print("【场景1】大额交易成本优化规则")
    context1 = {
        'stock_code': '000560',
        'current_price': 10.0,
        'trade_amount': 150000,  # 15万元，超过10万
        'position_ratio': 0.1,
        'available_cash': 200000,
        'total_assets': 500000,
        'predicted_return': 0.03,
        'confidence': 0.7,
        'risk_level': '中',
        'current_hour': 10
    }
    result1 = rule_engine.evaluate_all(context1)
    print(f"触发规则数: {result1['rule_count']}")
    for rule in result1['triggered_rules']:
        print(f"  - {rule['name']}: {rule.get('action', {}).get('message', '')}")
    print()
    
    # 场景2：高风险低置信度（触发MS001）
    print("【场景2】高风险市场暂停规则")
    context2 = {
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
    }
    result2 = rule_engine.evaluate_all(context2)
    print(f"触发规则数: {result2['rule_count']}")
    for rule in result2['triggered_rules']:
        print(f"  - {rule['name']}: {rule.get('action', {}).get('message', '')}")
    print()
    
    # 场景3：高置信度高收益（触发CO002）
    print("【场景3】高置信度优化规则")
    context3 = {
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
    }
    result3 = rule_engine.evaluate_all(context3)
    print(f"触发规则数: {result3['rule_count']}")
    for rule in result3['triggered_rules']:
        print(f"  - {rule['name']}: {rule.get('action', {}).get('message', '')}")
    print()
    
    # 场景4：有持仓且亏损（触发RC002止损）
    print("【场景4】止损规则")
    context4 = {
        'stock_code': '000560',
        'current_price': 9.0,  # 当前价格9元
        'trade_amount': 10000,
        'position_ratio': 0.15,
        'available_cash': 50000,
        'total_assets': 100000,
        'predicted_return': 0.01,
        'confidence': 0.6,
        'risk_level': '中',
        'current_hour': 10,
        'loss_ratio': -0.1,  # 亏损10%（成本价10元，当前价9元）
        'total_loss_ratio': -0.015  # 总资产亏损1.5%
    }
    result4 = rule_engine.evaluate_all(context4)
    print(f"触发规则数: {result4['rule_count']}")
    for rule in result4['triggered_rules']:
        print(f"  - {rule['name']}: {rule.get('action', {}).get('message', '')}")
    print()
    
    # 场景5：有持仓且盈利（触发RC005止盈）
    print("【场景5】止盈规则")
    context5 = {
        'stock_code': '000560',
        'current_price': 11.5,  # 当前价格11.5元
        'trade_amount': 10000,
        'position_ratio': 0.15,  # 持仓比例15%
        'available_cash': 50000,
        'total_assets': 100000,
        'predicted_return': 0.01,
        'confidence': 0.6,
        'risk_level': '中',
        'current_hour': 10,
        'loss_ratio': 0.15,  # 盈利15%（成本价10元，当前价11.5元）
        'total_loss_ratio': 0.02  # 总资产盈利2%
    }
    result5 = rule_engine.evaluate_all(context5)
    print(f"触发规则数: {result5['rule_count']}")
    for rule in result5['triggered_rules']:
        print(f"  - {rule['name']}: {rule.get('action', {}).get('message', '')}")
    print()
    
    # 场景6：持仓比例过高（触发RC001）
    print("【场景6】单只股票持仓限制规则")
    context6 = {
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
    }
    result6 = rule_engine.evaluate_all(context6)
    print(f"触发规则数: {result6['rule_count']}")
    print(f"是否允许: {result6['is_allowed']}")
    print(f"原因: {result6['reason']}")
    for rule in result6['triggered_rules']:
        print(f"  - {rule['name']}: {rule.get('action', {}).get('message', '')}")
    print()
    
    # 场景7：资金不足（触发RC004）
    print("【场景7】资金不足限制规则")
    context7 = {
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
    }
    result7 = rule_engine.evaluate_all(context7)
    print(f"触发规则数: {result7['rule_count']}")
    print(f"是否允许: {result7['is_allowed']}")
    print(f"原因: {result7['reason']}")
    for rule in result7['triggered_rules']:
        print(f"  - {rule['name']}: {rule.get('action', {}).get('message', '')}")
    print()
    
    # 场景8：单日限额（触发RC006）
    print("【场景8】单日限额规则")
    context8 = {
        'stock_code': '000560',
        'current_price': 10.0,
        'trade_amount': 25000,  # 交易金额2.5万，超过总资产20%（2万）
        'position_ratio': 0.1,
        'available_cash': 30000,
        'total_assets': 100000,
        'predicted_return': 0.03,
        'confidence': 0.7,
        'risk_level': '中',
        'current_hour': 10
    }
    result8 = rule_engine.evaluate_all(context8)
    print(f"触发规则数: {result8['rule_count']}")
    print(f"是否允许: {result8['is_allowed']}")
    print(f"原因: {result8['reason']}")
    for rule in result8['triggered_rules']:
        print(f"  - {rule['name']}: {rule.get('action', {}).get('message', '')}")
    print()
    
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    test_rule_scenarios()

