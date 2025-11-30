"""
测试股票价格获取 - 测试多数据源
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径，以便导入src模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.stock_data import StockDataProvider

def test_price():
    """测试获取股票价格"""
    provider = StockDataProvider()
    
    test_stocks = ['000001', '000876', '600036', '600276', '600519']
    
    print("=" * 60)
    print("测试股票价格获取（多数据源自动切换）")
    print("=" * 60)
    print()
    
    for stock_code in test_stocks:
        try:
            price = provider.get_current_price(stock_code)
            print(f"✓ 股票 {stock_code}: ￥{price:.2f}")
        except Exception as e:
            print(f"✗ 股票 {stock_code}: 获取失败 - {e}")
    
    print()
    print("=" * 60)
    print("测试历史数据获取")
    print("=" * 60)
    print()
    
    for stock_code in test_stocks[:2]:  # 只测试前2个
        try:
            df = provider.get_stock_data(stock_code, days=30)
            if not df.empty:
                print(f"✓ 股票 {stock_code}: 获取到 {len(df)} 条历史数据")
                print(f"  最新收盘价: ￥{df['close'].iloc[-1]:.2f}")
            else:
                print(f"✗ 股票 {stock_code}: 数据为空")
        except Exception as e:
            print(f"✗ 股票 {stock_code}: 获取失败 - {e}")
    
    print()
    print("=" * 60)

if __name__ == '__main__':
    test_price()

