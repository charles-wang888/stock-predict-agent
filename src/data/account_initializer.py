"""
账户初始化模块 - 从真实接口获取股票数据并初始化持仓
"""
import random
from typing import Dict, Any, List
from .stock_data import StockDataProvider
from config.settings import DEFAULT_ACCOUNT


class AccountInitializer:
    """账户初始化器"""
    
    # 一些常见的股票代码（用于初始化持仓）
    STOCK_POOL = [
        '000001',  # 平安银行
        '000002',  # 万科A
        '000858',  # 五粮液
        '600000',  # 浦发银行
        '600036',  # 招商银行
        '600887',  # 伊利股份
        '000560',  # 我爱我家
        '000876',  # 新希望
        '600276',  # 恒瑞医药
    ]
    
    def __init__(self, data_provider: StockDataProvider = None):
        """
        初始化账户初始化器
        
        Args:
            data_provider: 股票数据提供者
        """
        self.data_provider = data_provider or StockDataProvider()
    
    def initialize_account(self, num_stocks: int = 5, 
                          shares_range: tuple = (5000, 10000),
                          total_assets: float = None) -> Dict[str, Any]:
        """
        初始化账户，添加真实股票持仓
        
        Args:
            num_stocks: 要添加的股票数量
            shares_range: 持仓数量范围（最小，最大）
            total_assets: 总资产（如果None，使用默认值）
        
        Returns:
            初始化后的账户信息
        """
        if total_assets is None:
            total_assets = DEFAULT_ACCOUNT['total_assets']
        
        # 随机选择股票
        selected_stocks = random.sample(self.STOCK_POOL, min(num_stocks, len(self.STOCK_POOL)))
        
        positions = {}
        total_position_value = 0
        
        print(f"正在初始化账户，添加 {len(selected_stocks)} 只股票的持仓...")
        
        for stock_code in selected_stocks:
            try:
                # 获取当前股价
                current_price = self.data_provider.get_current_price(stock_code)
                
                if current_price <= 0:
                    print(f"  跳过 {stock_code}：无法获取有效价格")
                    continue
                
                # 随机生成持仓数量（必须是100的倍数，符合A股交易规则）
                min_shares = (shares_range[0] // 100) * 100  # 向下取整到100的倍数
                max_shares = (shares_range[1] // 100) * 100  # 向下取整到100的倍数
                if min_shares < 100:
                    min_shares = 100
                if max_shares < min_shares:
                    max_shares = min_shares
                shares = random.randint(min_shares // 100, max_shares // 100) * 100
                
                # 计算持仓价值
                position_value = shares * current_price
                total_position_value += position_value
                
                # 设置成本价（略低于当前价，模拟买入时的价格）
                cost_price = current_price * random.uniform(0.85, 1.05)  # 成本价在85%-105%之间
                
                # 存储基础持仓信息（决策引擎需要的格式）
                positions[stock_code] = {
                    'shares': shares,
                    'cost': float(cost_price)
                }
                
                # 额外信息用于展示（不用于规则评估）
                positions[stock_code + '_info'] = {
                    'current_price': float(current_price),
                    'position_value': float(position_value),
                    'profit': float((current_price - cost_price) * shares),
                    'profit_ratio': float((current_price - cost_price) / cost_price) if cost_price > 0 else 0
                }
                
                info = positions[stock_code + '_info']
                profit_pct = info['profit_ratio'] * 100
                print(f"  ✓ {stock_code}: {shares}股, 成本价￥{cost_price:.2f}, 当前价￥{current_price:.2f}, "
                      f"盈亏{profit_pct:+.2f}%")
                
            except Exception as e:
                print(f"  跳过 {stock_code}：获取数据失败 - {e}")
                continue
        
        # 计算可用资金（总资产的30%作为可用资金）
        available_cash = total_assets * 0.3
        
        # 如果持仓价值超过总资产的70%，调整总资产
        if total_position_value > total_assets * 0.7:
            total_assets = total_position_value / 0.7
            available_cash = total_assets * 0.3
        
        account = {
            'total_assets': float(total_assets),
            'available_cash': float(available_cash),
            'positions': positions
        }
        
        # 计算账户总价值
        account_value = total_position_value + available_cash
        if account_value != total_assets:
            # 调整总资产以匹配实际价值
            account['total_assets'] = account_value
        
        print(f"\n账户初始化完成：")
        print(f"  总资产：￥{account['total_assets']:,.2f}")
        print(f"  可用资金：￥{account['available_cash']:,.2f}")
        print(f"  持仓价值：￥{total_position_value:,.2f}")
        print(f"  持仓股票数：{len(positions)}")
        
        return account
    
    def get_account_summary(self, account: Dict[str, Any]) -> Dict[str, Any]:
        """获取账户摘要信息（会更新当前价格）"""
        positions = account.get('positions', {})
        
        # 分离基础持仓和额外信息
        base_positions = {k: v for k, v in positions.items() if not k.endswith('_info')}
        info_positions = {k: v for k, v in positions.items() if k.endswith('_info')}
        
        # 更新当前价格和盈亏信息
        total_position_value = 0
        total_profit = 0
        
        # 构建持仓详情
        position_details = {}
        for stock_code, pos in base_positions.items():
            try:
                # 获取最新价格
                current_price = self.data_provider.get_current_price(stock_code)
                shares = pos.get('shares', 0)
                cost = pos.get('cost', 0)
                
                # 计算最新持仓价值
                position_value = shares * current_price
                profit = (current_price - cost) * shares
                profit_ratio = (current_price - cost) / cost if cost > 0 else 0
                
                # 更新账户中的价格信息
                info_key = stock_code + '_info'
                if info_key not in info_positions:
                    account['positions'][info_key] = {}
                
                account['positions'][info_key].update({
                    'current_price': float(current_price),
                    'position_value': float(position_value),
                    'profit': float(profit),
                    'profit_ratio': float(profit_ratio)
                })
                
                total_position_value += position_value
                total_profit += profit
                
                # 获取股票名称
                stock_name = self.data_provider.get_stock_name(stock_code)
                
                position_details[stock_code] = {
                    'shares': shares,
                    'cost': float(cost),
                    'current_price': float(current_price),
                    'position_value': float(position_value),
                    'profit': float(profit),
                    'profit_ratio': float(profit_ratio),
                    'stock_name': stock_name
                }
            except Exception as e:
                # 如果获取价格失败，使用旧数据
                info_key = stock_code + '_info'
                if info_key in info_positions:
                    info = info_positions[info_key]
                    # 获取股票名称
                    stock_name = self.data_provider.get_stock_name(stock_code)
                    position_details[stock_code] = {
                        **pos,
                        **info,
                        'stock_name': stock_name
                    }
                    total_position_value += info.get('position_value', 0)
                    total_profit += info.get('profit', 0)
                else:
                    # 即使没有旧数据，也获取股票名称
                    stock_name = self.data_provider.get_stock_name(stock_code)
                    position_details[stock_code] = {
                        **pos,
                        'stock_name': stock_name
                    }
        
        return {
            'total_assets': account.get('total_assets', 0),
            'available_cash': account.get('available_cash', 0),
            'total_position_value': total_position_value,
            'total_profit': total_profit,
            'total_profit_ratio': total_profit / account.get('total_assets', 1) if account.get('total_assets', 0) > 0 else 0,
            'num_positions': len(base_positions),
            'positions': position_details
        }

