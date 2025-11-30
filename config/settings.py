"""
系统配置文件
"""
import os

# Flask配置
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# 数据配置
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RULES_FILE = os.path.join(os.path.dirname(__file__), 'rules.json')

# 账户模拟配置（用于测试）
DEFAULT_ACCOUNT = {
    'total_assets': 100000,  # 总资产：10万元
    'available_cash': 50000,  # 可用资金：5万元
    'positions': {}  # 持仓：{股票代码: {'shares': 数量, 'cost': 成本价}}
}

# 预测配置
PREDICTION_DAYS = 5  # 预测未来5天
LOOKBACK_DAYS = 60   # 使用过去60天数据


