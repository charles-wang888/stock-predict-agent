"""
规则场景演示脚本
运行所有预设场景并显示结果
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径，以便导入src模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scenarios.rule_scenarios import RuleScenarioDemo

def main():
    """运行所有场景演示"""
    demo = RuleScenarioDemo()
    results = demo.run_all_scenarios()
    
    print("=" * 80)
    print("规则触发场景演示")
    print("=" * 80)
    print()
    
    for i, result in enumerate(results, 1):
        scenario = result['scenario']
        print(f"【场景{i}】{scenario['name']}")
        print(f"描述：{scenario['description']}")
        print(f"触发规则数：{result['result']['rule_count']}")
        print(f"是否允许：{'是' if result['is_allowed'] else '否'}")
        if result['reason']:
            print(f"原因：{result['reason']}")
        
        print("\n触发的规则：")
        for rule in result['result']['triggered_rules']:
            action = rule.get('action', {})
            print(f"  - [{rule['rule_id']}] {rule['name']}")
            print(f"    动作：{action.get('type', '')}")
            print(f"    消息：{action.get('message', '')}")
        
        if result['result']['warnings']:
            print("\n警告信息：")
            for warning in result['result']['warnings']:
                print(f"  - {warning['message']}")
        
        if result['result']['optimizations']:
            print("\n优化建议：")
            for opt in result['result']['optimizations']:
                print(f"  - {opt['message']}")
        
        print("\n" + "-" * 80)
        print()
    
    print("=" * 80)
    print("演示完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()

