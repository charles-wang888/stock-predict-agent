"""
股票数据获取模块
支持多个免费数据源，按优先级自动切换
"""
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from .technical_indicators import TechnicalIndicators

# 尝试导入akshare，如果失败则设为None
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    print(f"[初始化] akshare已安装，版本: {ak.__version__ if hasattr(ak, '__version__') else '未知'}")
except ImportError:
    AKSHARE_AVAILABLE = False
    print(f"[初始化] ⚠️ akshare未安装，将无法使用akshare数据源")
    ak = None


class StockDataProvider:
    """股票数据提供者类 - 支持多数据源自动切换"""
    
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_data(self, stock_code: str, days: int = 60) -> pd.DataFrame:
        """
        获取股票历史数据 - 多数据源自动切换
        
        Args:
            stock_code: 股票代码（如：'000560'）
            days: 获取最近多少天的数据
        
        Returns:
            包含股票数据的DataFrame
        """
        # 优先使用akshare（最可靠），然后是其他接口
        # 方法1: 尝试akshare（如果可用，最可靠的数据源）
        print(f"[数据获取] 开始获取 {stock_code} 的历史数据，需要 {days} 天")
        if AKSHARE_AVAILABLE:
            try:
                print(f"[数据获取] 尝试从akshare获取 {stock_code} 数据...")
                # 计算日期范围
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=days + 30)).strftime('%Y%m%d')  # 多获取一些数据，确保有足够的历史数据
                
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=""  # 不复权，使用实际价格
                )
                if df is not None and not df.empty:
                    print(f"[akshare数据] 原始数据列名: {list(df.columns)}")
                    print(f"[akshare数据] 原始数据行数: {len(df)}")
                    
                    # akshare返回的列名可能是中文，需要检查
                    date_col = None
                    for col in ['日期', 'date', 'Date', '交易日期']:
                        if col in df.columns:
                            date_col = col
                            break
                    
                    if date_col is None:
                        print(f"[akshare数据] 数据格式异常，缺少日期列，列名: {list(df.columns)}")
                        raise ValueError("数据格式异常")
                    
                    # 检查价格列
                    price_cols = {}
                    for col in df.columns:
                        col_lower = col.lower()
                        if '收盘' in col or 'close' in col_lower:
                            price_cols['close'] = col
                        elif '开盘' in col or 'open' in col_lower:
                            price_cols['open'] = col
                        elif '最高' in col or 'high' in col_lower:
                            price_cols['high'] = col
                        elif '最低' in col or 'low' in col_lower:
                            price_cols['low'] = col
                        elif '成交量' in col or 'volume' in col_lower:
                            price_cols['volume'] = col
                    
                    if 'close' not in price_cols:
                        print(f"[akshare数据] 缺少收盘价列，列名: {list(df.columns)}")
                        raise ValueError("缺少收盘价列")
                    
                    # 按日期排序并取最近days条
                    df = df.sort_values(date_col, ascending=True)
                    df = df.tail(days)
                    
                    # 构建标准格式的DataFrame
                    records = []
                    for idx, row in df.iterrows():
                        try:
                            close_price = float(row[price_cols['close']])
                            open_price = float(row.get(price_cols.get('open', price_cols['close']), close_price))
                            high_price = float(row.get(price_cols.get('high', close_price), close_price))
                            low_price = float(row.get(price_cols.get('low', close_price), close_price))
                            volume = float(row.get(price_cols.get('volume', 0), 0))
                            
                            # 检查价格是否合理（A股价格通常在0.01-10000元之间）
                            if close_price < 0.01 or close_price > 10000:
                                print(f"[akshare数据] 价格异常: {close_price}，跳过该条数据")
                                continue
                            
                            date_val = row[date_col]
                            if isinstance(date_val, str):
                                date_val = pd.to_datetime(date_val, errors='coerce')
                            elif not isinstance(date_val, pd.Timestamp):
                                date_val = pd.to_datetime(date_val, errors='coerce')
                            
                            records.append({
                                'date': date_val,
                                'open': open_price,
                                'close': close_price,
                                'high': high_price,
                                'low': low_price,
                                'volume': volume
                            })
                        except Exception as e:
                            print(f"[akshare数据] 解析数据行失败: {str(e)[:50]}")
                            continue
                    
                    if not records:
                        print(f"[akshare数据] 解析后无有效记录")
                        raise ValueError("解析后无有效记录")
                    
                    df = pd.DataFrame(records)
                    df = df.sort_values('date').reset_index(drop=True)
                    
                    # 添加缺失的列
                    df['turnover'] = 0
                    df['amplitude'] = ((df['high'] - df['low']) / df['close'] * 100).round(2)
                    df['change_pct'] = df['close'].pct_change().fillna(0) * 100
                    df['change_amount'] = df['close'].diff().fillna(0)
                    df['turnover_rate'] = 0
                    
                    # 验证数据有效性
                    if df.empty or df['close'].isna().all():
                        print(f"[akshare数据] 数据验证失败：数据为空或价格全为NaN")
                        raise ValueError("数据验证失败")
                    
                    # 检查价格范围是否合理
                    min_price = df['close'].min()
                    max_price = df['close'].max()
                    if min_price < 0.01 or max_price > 10000:
                        print(f"[akshare数据] ⚠️ 价格范围异常: {min_price:.2f} - {max_price:.2f}，可能单位有误")
                    
                    price_range = f"{min_price:.2f} - {max_price:.2f}"
                    print(f"[akshare数据] ✅ 成功获取 {stock_code} 真实数据，共 {len(df)} 条，价格范围: {price_range}")
                    
                    df = TechnicalIndicators.add_all_indicators(df)
                    df['_data_source'] = 'akshare'  # 标记数据来源
                    return df.reset_index(drop=True)
                else:
                    print(f"[akshare数据] ⚠️ 返回数据为空，将尝试其他数据源")
            except ImportError:
                print(f"[数据获取] ⚠️ akshare未安装，将尝试其他数据源")
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()[:500]
                print(f"[数据获取] ⚠️ akshare接口失败: {str(e)[:200]}")
                print(f"[数据获取] 错误详情: {error_detail}")
        else:
            print(f"[数据获取] ⚠️ akshare不可用，将尝试其他数据源")
        
        # 方法2: 尝试腾讯财经接口
        try:
            df = self._get_data_from_tencent(stock_code, days)
            if df is not None and not df.empty:
                print(f"[数据获取] ✅ 从腾讯财经成功获取 {stock_code} 真实数据，价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
                df['_data_source'] = 'tencent'  # 标记数据来源
                return df
        except Exception as e:
            print(f"[数据获取] 腾讯财经接口失败: {str(e)[:100]}")
        
        # 方法3: 尝试网易财经接口
        try:
            df = self._get_data_from_163(stock_code, days)
            if df is not None and not df.empty:
                print(f"[数据获取] ✅ 从网易财经成功获取 {stock_code} 真实数据，价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
                df['_data_source'] = '163'  # 标记数据来源
                return df
        except Exception as e:
            print(f"[数据获取] 网易财经接口失败: {str(e)[:100]}")
        
        # 方法4: 尝试新浪财经接口
        try:
            df = self._get_data_from_sina(stock_code, days)
            if df is not None and not df.empty:
                print(f"[数据获取] ✅ 从新浪财经成功获取 {stock_code} 真实数据，价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
                df['_data_source'] = 'sina'  # 标记数据来源
                return df
        except Exception as e:
            print(f"[数据获取] 新浪财经接口失败: {str(e)[:100]}")
        
        # 如果所有方法都失败，返回模拟数据
        print(f"[数据获取] ⚠️⚠️⚠️ 所有真实数据接口均失败，使用模拟数据（将尝试获取实际价格作为基础）")
        print(f"[数据获取] 请检查：1) 网络连接 2) akshare是否安装(pip install akshare) 3) 后端控制台的错误信息")
        mock_data = self._generate_mock_data(stock_code, days)
        if not mock_data.empty:
            print(f"[模拟数据] ⚠️⚠️⚠️ 生成 {stock_code} 模拟数据，价格范围: {mock_data['close'].min():.2f} - {mock_data['close'].max():.2f}")
            print(f"[模拟数据] ⚠️ 注意：这是模拟数据，仅供参考！请检查网络连接和数据源配置。")
            mock_data['_data_source'] = 'mock'  # 标记为模拟数据
        return mock_data
    
    def _get_data_from_sina(self, stock_code: str, days: int) -> pd.DataFrame:
        """从新浪财经获取股票数据"""
        # 确定市场代码
        if stock_code.startswith('6'):
            symbol = f"sh{stock_code}"  # 上海
        else:
            symbol = f"sz{stock_code}"  # 深圳
        
        # 新浪财经K线数据接口
        url = f"http://stock.finance.sina.com.cn/usstock/api/json.php/US_MarketDataService.getKLineData"
        params = {
            'symbol': symbol,
            'scale': '240',  # 日线
            'datalen': str(days)
        }
        
        try:
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code == 200 and response.text:
                # 尝试解析JSON格式
                try:
                    data = json.loads(response.text)
                    if data and len(data) > 0:
                        records = []
                        for item in data[-days:]:
                            if len(item) >= 5:
                                try:
                                    records.append({
                                        'date': pd.to_datetime(item[0], errors='coerce'),
                                        'open': float(item[1]),
                                        'close': float(item[2]),
                                        'high': float(item[3]),
                                        'low': float(item[4]),
                                        'volume': float(item[5]) if len(item) > 5 else 0
                                    })
                                except:
                                    continue
                        
                        if records:
                            df = pd.DataFrame(records)
                            df = df.dropna(subset=['date'])
                            df = df.sort_values('date').tail(days).reset_index(drop=True)
                            
                            # 添加缺失的列
                            df['turnover'] = 0
                            df['amplitude'] = ((df['high'] - df['low']) / df['close'] * 100).round(2)
                            df['change_pct'] = df['close'].pct_change().fillna(0) * 100
                            df['change_amount'] = df['close'].diff().fillna(0)
                            df['turnover_rate'] = 0
                            
                            df = TechnicalIndicators.add_all_indicators(df)
                            print(f"[新浪数据] 成功获取 {stock_code} 数据，共 {len(df)} 条，价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
                            return df
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            pass
        
        # 尝试另一个新浪接口
        try:
            # 新浪财经日线数据接口（另一种格式）
            url2 = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
            params2 = {
                'symbol': symbol,
                'scale': 'd',  # 日线
                'ma': 'no',
                'datalen': str(days)
            }
            response2 = self.session.get(url2, params=params2, timeout=5)
            if response2.status_code == 200 and response2.text:
                try:
                    data = json.loads(response2.text)
                    if data and len(data) > 0:
                        records = []
                        for item in data[-days:]:
                            if isinstance(item, dict) and 'day' in item:
                                try:
                                    day_str = item['day']
                                    records.append({
                                        'date': pd.to_datetime(day_str, errors='coerce'),
                                        'open': float(item.get('open', 0)),
                                        'close': float(item.get('close', 0)),
                                        'high': float(item.get('high', 0)),
                                        'low': float(item.get('low', 0)),
                                        'volume': float(item.get('volume', 0))
                                    })
                                except:
                                    continue
                        
                        if records:
                            df = pd.DataFrame(records)
                            df = df.dropna(subset=['date'])
                            df = df.sort_values('date').tail(days).reset_index(drop=True)
                            
                            # 添加缺失的列
                            df['turnover'] = 0
                            df['amplitude'] = ((df['high'] - df['low']) / df['close'] * 100).round(2)
                            df['change_pct'] = df['close'].pct_change().fillna(0) * 100
                            df['change_amount'] = df['close'].diff().fillna(0)
                            df['turnover_rate'] = 0
                            
                            df = TechnicalIndicators.add_all_indicators(df)
                            print(f"[新浪数据] 成功获取 {stock_code} 数据，共 {len(df)} 条，价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
                            return df
                except:
                    pass
        except:
            pass
        
        return None
    
    def _get_data_from_tencent(self, stock_code: str, days: int) -> pd.DataFrame:
        """从腾讯财经获取股票数据"""
        # 确定市场代码
        if stock_code.startswith('6'):
            market_code = 'sh'
        else:
            market_code = 'sz'
        
        symbol = f"{market_code}{stock_code}"
        
        # 腾讯财经K线数据接口
        url = f"http://web.ifzq.gtimg.cn/appstock/app/kline/kline"
        params = {
            'param': f'{symbol},day,,,{days}',
            '_var': 'kline_day'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)  # 增加超时时间
            if response.status_code == 200:
                text = response.text
                # 检查响应是否有效
                if not text or len(text) < 10:
                    print(f"[腾讯数据] 响应内容为空或过短")
                    return None
                # 解析JSONP格式
                if '=' in text:
                    try:
                        json_str = text.split('=', 1)[1].strip().rstrip(';')
                        data = json.loads(json_str)
                        
                        if 'data' in data and symbol in data['data']:
                            kline_data = data['data'][symbol].get('day', [])
                            if kline_data and len(kline_data) > 0:
                                # 解析K线数据
                                records = []
                                # 先检查第一个数据点的价格，判断单位
                                sample_price = float(kline_data[0][2]) if len(kline_data[0]) >= 3 else 0
                                # 对于A股，正常价格应该在0.01-10000元之间
                                # 如果价格在100-1000之间，可能是"分"（如289分=2.89元）
                                # 但如果价格>1000，很可能是"元"（如1471元），不应该转换
                                # 只有当价格在100-1000之间时，才可能是"分"
                                need_convert = 100 < sample_price < 1000
                                
                                if need_convert:
                                    print(f"[腾讯数据] 检测到价格单位可能是'分'，将除以100转换。样本价格: {sample_price}")
                                elif sample_price >= 1000:
                                    print(f"[腾讯数据] 价格较高({sample_price:.2f})，保持原单位（元）")
                                
                                for item in kline_data[-days:]:
                                    if len(item) >= 6:
                                        open_price = float(item[1])
                                        close_price = float(item[2])
                                        high_price = float(item[3])
                                        low_price = float(item[4])
                                        
                                        # 如果价格看起来是以"分"为单位，则除以100
                                        if need_convert:
                                            open_price = open_price / 100
                                            close_price = close_price / 100
                                            high_price = high_price / 100
                                            low_price = low_price / 100
                                        
                                        records.append({
                                            'date': pd.to_datetime(item[0], format='%Y%m%d'),
                                            'open': open_price,
                                            'close': close_price,
                                            'high': high_price,
                                            'low': low_price,
                                            'volume': float(item[5])
                                        })
                                
                                if records:
                                    print(f"[腾讯数据] 成功获取 {stock_code} 数据，共 {len(records)} 条，价格范围: {min([r['close'] for r in records]):.2f} - {max([r['close'] for r in records]):.2f}")
                                    df = pd.DataFrame(records)
                                    df = df.sort_values('date').reset_index(drop=True)
                                    
                                    # 验证数据有效性
                                    if df.empty or df['close'].isna().all():
                                        print(f"[腾讯数据] 数据验证失败：数据为空或价格全为NaN")
                                        return None
                                    
                                    # 添加缺失的列
                                    df['turnover'] = 0
                                    df['amplitude'] = ((df['high'] - df['low']) / df['close'] * 100).round(2)
                                    df['change_pct'] = df['close'].pct_change().fillna(0) * 100
                                    df['change_amount'] = df['close'].diff().fillna(0)
                                    df['turnover_rate'] = 0
                                    
                                    df = TechnicalIndicators.add_all_indicators(df)
                                    return df
                                else:
                                    print(f"[腾讯数据] 解析后无有效记录")
                            else:
                                print(f"[腾讯数据] K线数据为空")
                        else:
                            print(f"[腾讯数据] 未找到K线数据，symbol: {symbol}")
                    except json.JSONDecodeError as e:
                        print(f"[腾讯数据] JSON解析失败: {str(e)[:100]}")
                else:
                    print(f"[腾讯数据] 响应格式不正确，缺少'='分隔符")
            else:
                print(f"[腾讯数据] HTTP请求失败，状态码: {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"[腾讯数据] 请求超时")
        except requests.exceptions.RequestException as e:
            print(f"[腾讯数据] 网络请求失败: {str(e)[:100]}")
        except Exception as e:
            print(f"[腾讯数据] 未知错误: {str(e)[:100]}")
        
        return None
    
    def _get_data_from_163(self, stock_code: str, days: int) -> pd.DataFrame:
        """从网易财经获取股票数据"""
        # 确定市场代码
        if stock_code.startswith('6'):
            market_code = '0'  # 上海
        else:
            market_code = '1'  # 深圳
        
        symbol = f"{market_code}{stock_code}"
        
        # 网易财经K线数据接口
        end_date = int(time.time() * 1000)
        start_date = int((time.time() - days * 24 * 3600) * 1000)
        
        url = f"http://quotes.money.163.com/service/chddata.html"
        params = {
            'code': symbol,
            'start': start_date,
            'end': end_date,
            'fields': 'TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)  # 增加超时时间
            if response.status_code == 200 and response.text:
                text = response.text.strip()
                if not text or len(text) < 10:
                    print(f"[网易数据] 响应内容为空或过短")
                    return None
                # 解析CSV格式
                lines = text.split('\n')
                if len(lines) > 1:
                    records = []
                    # 先检查第一个数据点的价格，判断单位
                    first_line_parts = lines[1].split(',') if len(lines) > 1 else []
                    sample_price = float(first_line_parts[3]) if len(first_line_parts) > 3 and first_line_parts[3] else 0
                    # 对于A股，正常价格应该在0.01-10000元之间
                    # 如果价格在100-1000之间，可能是"分"（如289分=2.89元）
                    # 但如果价格>1000，很可能是"元"（如1471元），不应该转换
                    need_convert_163 = 100 < sample_price < 1000
                    
                    if need_convert_163:
                        print(f"[网易数据] 检测到价格单位可能是'分'，将除以100转换。样本价格: {sample_price}")
                    elif sample_price >= 1000:
                        print(f"[网易数据] 价格较高({sample_price:.2f})，保持原单位（元）")
                    
                    for line in lines[1:]:  # 跳过标题行
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                date_str = parts[0]
                                close = float(parts[3]) if parts[3] else 0
                                high = float(parts[4]) if len(parts) > 4 and parts[4] else close
                                low = float(parts[5]) if len(parts) > 5 and parts[5] else close
                                open_price = float(parts[6]) if len(parts) > 6 and parts[6] else close
                                volume = float(parts[11]) if len(parts) > 11 and parts[11] else 0
                                
                                # 如果价格看起来是以"分"为单位，则除以100
                                if need_convert_163:
                                    close = close / 100
                                    high = high / 100
                                    low = low / 100
                                    open_price = open_price / 100
                                
                                records.append({
                                    'date': pd.to_datetime(date_str, errors='coerce'),
                                    'open': open_price,
                                    'close': close,
                                    'high': high,
                                    'low': low,
                                    'volume': volume
                                })
                            except:
                                continue
                    
                    if records:
                        print(f"[网易数据] 成功获取 {stock_code} 数据，共 {len(records)} 条，价格范围: {min([r['close'] for r in records]):.2f} - {max([r['close'] for r in records]):.2f}")
                        df = pd.DataFrame(records)
                        df = df.dropna(subset=['date'])
                        df = df.sort_values('date').tail(days).reset_index(drop=True)
                        
                        # 验证数据有效性
                        if df.empty or df['close'].isna().all():
                            print(f"[网易数据] 数据验证失败：数据为空或价格全为NaN")
                            return None
                        
                        # 添加缺失的列
                        df['turnover'] = 0
                        df['amplitude'] = ((df['high'] - df['low']) / df['close'] * 100).round(2)
                        df['change_pct'] = df['close'].pct_change().fillna(0) * 100
                        df['change_amount'] = df['close'].diff().fillna(0)
                        df['turnover_rate'] = 0
                        
                        df = TechnicalIndicators.add_all_indicators(df)
                        return df
                else:
                    print(f"[网易数据] CSV数据行数不足，只有 {len(lines)} 行")
            else:
                print(f"[网易数据] HTTP请求失败，状态码: {response.status_code if response else 'None'}")
        except requests.exceptions.Timeout:
            print(f"[网易数据] 请求超时")
        except requests.exceptions.RequestException as e:
            print(f"[网易数据] 网络请求失败: {str(e)[:100]}")
        except Exception as e:
            print(f"[网易数据] 未知错误: {str(e)[:100]}")
        
        return None
    
    def _generate_mock_data(self, stock_code: str, days: int) -> pd.DataFrame:
        """生成模拟股票数据（用于测试）"""
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        # 尝试获取实际价格作为基础价格
        base_price = 10.0
        try:
            # 方法1: 尝试从缓存获取
            if stock_code in self.cache:
                cached_data = self.cache[stock_code]
                if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                    if 'close' in cached_data.columns:
                        last_price = cached_data['close'].dropna().iloc[-1]
                        if last_price > 0:
                            base_price = float(last_price)
                            print(f"[模拟数据] 使用缓存价格 {stock_code}: {base_price:.2f}")
        except:
            pass
        
        # 方法2: 如果缓存中没有，尝试获取实时价格（避免递归调用get_stock_data）
        if base_price == 10.0:
            try:
                # 直接调用价格获取方法，不调用get_stock_data
                actual_price = self._get_price_from_tencent(stock_code)
                if actual_price and actual_price > 0:
                    # 检查是否需要单位转换（A股正常价格在0.01-10000元之间）
                    # 如果价格在100-1000之间，可能是"分"（如289分=2.89元）
                    # 但如果价格>1000，很可能是"元"（如1471元），不应该转换
                    if 100 < actual_price < 1000:
                        actual_price = actual_price / 100
                        print(f"[模拟数据] 价格单位转换: {actual_price * 100:.2f}分 -> {actual_price:.2f}元")
                    base_price = actual_price
                    print(f"[模拟数据] 使用腾讯实时价格 {stock_code}: {base_price:.2f}")
            except Exception as e1:
                try:
                    actual_price = self._get_price_from_sina(stock_code)
                    if actual_price and actual_price > 0:
                        if 100 < actual_price < 1000:
                            actual_price = actual_price / 100
                            print(f"[模拟数据] 价格单位转换: {actual_price * 100:.2f}分 -> {actual_price:.2f}元")
                        base_price = actual_price
                        print(f"[模拟数据] 使用新浪实时价格 {stock_code}: {base_price:.2f}")
                except Exception as e2:
                    try:
                        actual_price = self._get_price_from_163(stock_code)
                        if actual_price and actual_price > 0:
                            if 100 < actual_price < 1000:
                                actual_price = actual_price / 100
                                print(f"[模拟数据] 价格单位转换: {actual_price * 100:.2f}分 -> {actual_price:.2f}元")
                            base_price = actual_price
                            print(f"[模拟数据] 使用网易实时价格 {stock_code}: {base_price:.2f}")
                    except:
                        print(f"[模拟数据] 无法获取实时价格，使用默认价格10.0")
                        pass
        
        # 生成模拟价格数据（随机游走）
        prices = [base_price]
        for _ in range(days - 1):
            change = np.random.normal(0, 0.02)  # 每日波动约2%
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.1))  # 价格不能为负
        
        df = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'close': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
            'volume': np.random.randint(1000000, 10000000, days),
            'turnover': [0] * days,
            'amplitude': [0] * days,
            'change_pct': [0] * days,
            'change_amount': [0] * days,
            'turnover_rate': [0] * days
        })
        
        # 添加技术指标
        df = TechnicalIndicators.add_all_indicators(df)
        
        return df
    
    def get_current_price(self, stock_code: str) -> float:
        """
        获取当前股价（实时价格）- 多数据源自动切换
        
        Args:
            stock_code: 股票代码
        
        Returns:
            当前股价
        """
        # 方法1: 尝试新浪财经实时价格
        try:
            price = self._get_price_from_sina(stock_code)
            if price and price > 0:
                print(f"[价格获取] 从新浪财经获取 {stock_code} 价格: {price:.2f}")
                return price
        except Exception as e:
            pass
        
        # 方法2: 尝试腾讯财经实时价格
        try:
            price = self._get_price_from_tencent(stock_code)
            if price and price > 0:
                print(f"[价格获取] 从腾讯财经获取 {stock_code} 价格: {price:.2f}")
                return price
        except Exception as e:
            pass
        
        # 方法3: 尝试网易财经实时价格
        try:
            price = self._get_price_from_163(stock_code)
            if price and price > 0:
                print(f"[价格获取] 从网易财经获取 {stock_code} 价格: {price:.2f}")
                return price
        except Exception as e:
            pass
        
        # 方法4: 尝试akshare实时行情（如果可用）
        if AKSHARE_AVAILABLE:
            try:
                realtime_data = ak.stock_zh_a_spot_em()
                if realtime_data is not None and not realtime_data.empty:
                    code_column = None
                    for col in ['代码', 'code', '股票代码']:
                        if col in realtime_data.columns:
                            code_column = col
                            break
                    
                    if code_column:
                        stock_row = realtime_data[realtime_data[code_column] == stock_code]
                        if not stock_row.empty:
                            for col in ['最新价', '现价', 'price', '当前价']:
                                if col in stock_row.columns:
                                    current_price = float(stock_row[col].iloc[0])
                                    if current_price > 0 and not pd.isna(current_price):
                                        print(f"[价格获取] 从akshare获取 {stock_code} 价格: {current_price:.2f}")
                                        return current_price
            except Exception as e:
                pass
        
        # 方法5: 从历史数据获取最新收盘价
        try:
            df = self.get_stock_data(stock_code, days=30)
            if not df.empty and 'close' in df.columns:
                last_price = df['close'].dropna().iloc[-1]
                if last_price > 0:
                    price = float(last_price)
                    print(f"[价格获取] 从历史数据获取 {stock_code} 最新收盘价: {price:.2f}")
                    return price
        except Exception as e:
            pass
        
        # 如果所有方法都失败，返回默认价格
        print(f"[价格获取] 警告: 无法获取股票 {stock_code} 的真实价格，使用默认价格10.00元")
        return 10.0
    
    def get_stock_name(self, stock_code: str) -> str:
        """
        获取股票名称
        
        Args:
            stock_code: 股票代码
        
        Returns:
            股票名称，如果获取失败返回股票代码
        """
        # 方法1: 从新浪财经获取
        try:
            if stock_code.startswith('6'):
                symbol = f"sh{stock_code}"
            else:
                symbol = f"sz{stock_code}"
            
            url = f"https://hq.sinajs.cn/list={symbol}"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                text = response.text
                if '=' in text and ',' in text:
                    data_str = text.split('=')[1].strip().strip('";')
                    parts = data_str.split(',')
                    if len(parts) >= 1:
                        name = parts[0].strip()
                        if name and name != '':
                            return name
        except:
            pass
        
        # 方法2: 从腾讯财经获取
        try:
            if stock_code.startswith('6'):
                symbol = f"sh{stock_code}"
            else:
                symbol = f"sz{stock_code}"
            
            url = f"http://qt.gtimg.cn/q={symbol}"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                text = response.text
                if '=' in text and '~' in text:
                    data_str = text.split('=')[1].strip().strip('";')
                    parts = data_str.split('~')
                    if len(parts) >= 2:
                        name = parts[1].strip()
                        if name and name != '':
                            return name
        except:
            pass
        
        # 如果都失败，返回股票代码
        return stock_code
    
    def _get_price_from_sina(self, stock_code: str) -> float:
        """从新浪财经获取实时价格"""
        if stock_code.startswith('6'):
            symbol = f"sh{stock_code}"
        else:
            symbol = f"sz{stock_code}"
        
        url = f"https://hq.sinajs.cn/list={symbol}"
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                text = response.text
                # 解析格式: var hq_str_sh600000="股票名称,今日开盘价,昨日收盘价,当前价格,...";
                if '=' in text and ',' in text:
                    data_str = text.split('=')[1].strip().strip('";')
                    parts = data_str.split(',')
                    if len(parts) >= 4:
                        price = float(parts[3])
                        if price > 0:
                            return price
        except:
            pass
        return None
    
    def _get_price_from_tencent(self, stock_code: str) -> float:
        """从腾讯财经获取实时价格"""
        if stock_code.startswith('6'):
            symbol = f"sh{stock_code}"
        else:
            symbol = f"sz{stock_code}"
        
        url = f"http://qt.gtimg.cn/q={symbol}"
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                text = response.text
                # 解析格式: v_sh600000="1~股票名称~代码~当前价格~...";
                if '=' in text and '~' in text:
                    data_str = text.split('=')[1].strip().strip('";')
                    parts = data_str.split('~')
                    if len(parts) >= 4:
                        price = float(parts[3])
                        if price > 0:
                            return price
        except:
            pass
        return None
    
    def _get_price_from_163(self, stock_code: str) -> float:
        """从网易财经获取实时价格"""
        if stock_code.startswith('6'):
            symbol = f"0{stock_code}"
        else:
            symbol = f"1{stock_code}"
        
        url = f"http://api.money.126.net/data/feed/{symbol}"
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                text = response.text
                # 解析JSONP格式
                if '{' in text and 'price' in text:
                    json_str = text.split('(')[1].rstrip(');')
                    data = json.loads(json_str)
                    if symbol in data:
                        price = float(data[symbol].get('price', 0))
                        if price > 0:
                            return price
        except:
            pass
        return None

