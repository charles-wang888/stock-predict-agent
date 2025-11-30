"""
智能体模块 - 股票交易智能体
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, Any
from ..data.stock_data import StockDataProvider
from ..data.account_initializer import AccountInitializer
from ..prediction.enhanced_predictor import EnhancedStockPredictor
from ..rule_engine.advanced_rule_engine import AdvancedRuleEngine
from ..decision.decision_engine import DecisionEngine
from ..decision.advanced_decision_engine import AdvancedDecisionEngine
from ..explainability.model_explainer import ModelExplainer
from config.settings import RULES_FILE, PREDICTION_DAYS, LOOKBACK_DAYS, DEFAULT_ACCOUNT


class StockAgent:
    """股票交易智能体"""
    
    def __init__(self, use_lstm: bool = True, auto_init_account: bool = True, 
                 use_advanced_decision: bool = True):
        """
        初始化智能体
        
        Args:
            use_lstm: 是否使用LSTM模型
            auto_init_account: 是否自动初始化账户（添加模拟持仓）
            use_advanced_decision: 是否使用高级决策引擎（体现复杂业务逻辑）
        """
        self.data_provider = StockDataProvider()
        self.predictor = EnhancedStockPredictor(
            lookback_days=LOOKBACK_DAYS,
            prediction_days=PREDICTION_DAYS,
            use_lstm=use_lstm,
            auto_train=True
        )
        self.rule_engine = AdvancedRuleEngine(rules_file=RULES_FILE)
        
        # 初始化账户
        if auto_init_account:
            initializer = AccountInitializer(self.data_provider)
            account = initializer.initialize_account(
                num_stocks=5,
                shares_range=(5000, 10000),
                total_assets=DEFAULT_ACCOUNT['total_assets']
            )
        else:
            account = DEFAULT_ACCOUNT.copy()
        
        # 选择使用高级决策引擎（体现复杂业务逻辑）或基础决策引擎
        if use_advanced_decision:
            self.decision_engine = AdvancedDecisionEngine(rule_engine=self.rule_engine, account=account)
        else:
            self.decision_engine = DecisionEngine(rule_engine=self.rule_engine, account=account)
        
        self.use_advanced_decision = use_advanced_decision
        self.explainer = ModelExplainer()
        self.account = account
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户输入的查询文本
        
        Returns:
            处理结果
        """
        # 提取股票代码
        stock_code = self._extract_stock_code(query)
        
        if not stock_code:
            return {
                'success': False,
                'message': '未能识别股票代码，请输入正确的股票代码（如：000560、000001）',
                'stock_code': None
            }
        
        try:
            # 获取股票数据
            stock_data = self.data_provider.get_stock_data(stock_code)
            # 检查数据来源
            data_source = 'unknown'
            if stock_data is not None and not stock_data.empty:
                if '_data_source' in stock_data.columns:
                    data_source = stock_data['_data_source'].iloc[0] if len(stock_data) > 0 else 'unknown'
                # 移除内部标记列，避免传递给前端
                if '_data_source' in stock_data.columns:
                    stock_data = stock_data.drop(columns=['_data_source'])
            current_price = self.data_provider.get_current_price(stock_code)
            
            # 进行预测
            prediction = self.predictor.predict(stock_data)
            
            # 可解释性分析
            feature_names = None
            if hasattr(self.predictor, 'lstm_predictor') and self.predictor.lstm_predictor:
                if hasattr(self.predictor.lstm_predictor, 'feature_columns') and self.predictor.lstm_predictor.feature_columns:
                    feature_names = self.predictor.lstm_predictor.feature_columns
            
            explanations = self.explainer.explain_prediction(
                stock_data,
                feature_names=feature_names
            )
            
            # 生成自然语言解释
            nl_explanation = self.explainer.generate_natural_language_explanation(prediction, explanations)
            
            # 做出决策（使用高级决策引擎时传入股票数据用于风险评估）
            if self.use_advanced_decision and hasattr(self.decision_engine, '_comprehensive_risk_assessment'):
                decision = self.decision_engine.make_decision(
                    stock_code=stock_code,
                    prediction=prediction,
                    current_price=current_price,
                    stock_data=stock_data  # 传入股票数据用于风险评估
                )
            else:
                decision = self.decision_engine.make_decision(
                    stock_code=stock_code,
                    prediction=prediction,
                    current_price=current_price
                )
            
            # 评估模型准确性（如果可能）
            accuracy_metrics = None
            if hasattr(self.predictor, 'evaluate_model_accuracy') and len(stock_data) >= LOOKBACK_DAYS + 15:
                try:
                    accuracy_metrics = self.predictor.evaluate_model_accuracy(stock_data, test_days=15)
                except:
                    pass
            
            # 处理NaN值，转换为None以便JSON序列化
            stock_data_dict = []
            if stock_data is not None:
                stock_data_records = stock_data.to_dict('records')
                for record in stock_data_records:
                    cleaned_record = {}
                    for key, value in record.items():
                        if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                            cleaned_record[key] = None
                        else:
                            cleaned_record[key] = value
                    stock_data_dict.append(cleaned_record)
            
            # 清理prediction中的NaN值
            cleaned_prediction = self._clean_dict_for_json(prediction)
            
            # 清理decision中的NaN值
            cleaned_decision = self._clean_dict_for_json(decision)
            
            # 清理explanations中的NaN值
            cleaned_explanations = self._clean_dict_for_json(explanations)
            
            # 清理accuracy_metrics中的NaN值
            cleaned_accuracy = self._clean_dict_for_json(accuracy_metrics) if accuracy_metrics else None
            
            # 数据源名称映射（不显示"模拟数据"文字，只通过颜色标识）
            data_source_names = {
                'akshare': 'akshare',
                'tencent': '腾讯财经',
                '163': '网易财经',
                'sina': '新浪财经',
                'mock': '',  # 模拟数据不显示文字，只显示黄色徽章
                'unknown': '未知来源'
            }
            data_source_display = data_source_names.get(data_source, '未知来源')
            
            return {
                'success': True,
                'stock_code': stock_code,
                'stock_data': stock_data_dict,
                'data_source': data_source,
                'data_source_display': data_source_display,
                'prediction': cleaned_prediction,
                'decision': cleaned_decision,
                'explanations': cleaned_explanations,
                'nl_explanation': nl_explanation,
                'accuracy_metrics': cleaned_accuracy,
                'message': f'成功分析股票 {stock_code}'
            }
            
        except RecursionError as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"递归错误详情:\n{error_trace}")
            return {
                'success': False,
                'message': f'处理过程中出错: 递归深度超限 - {str(e)}\n错误位置: 请查看控制台日志',
                'stock_code': stock_code
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"处理错误详情:\n{error_trace}")
            return {
                'success': False,
                'message': f'处理过程中出错: {str(e)}\n错误位置: 请查看控制台日志',
                'stock_code': stock_code
            }
    
    def _extract_stock_code(self, query: str) -> str:
        """从查询文本中提取股票代码（先尝试正则，失败则使用LLM意图识别）"""
        # 第一步：尝试正则表达式匹配股票代码
        patterns = [
            r'([0-9]{6})',  # 直接匹配6位数字（最宽松，放在最前面）
            r'股票代码[：:]\s*([0-9]{6})',  # "股票代码：000560"
            r'代码[：:]\s*([0-9]{6})',  # "代码：000560"
            r'\b([0-9]{6})\b',  # 单词边界的6位数字
        ]
        
        # 先尝试所有模式
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, query):
                code = match.group(1)
                # 验证股票代码格式（以0、3、6开头）
                if code.startswith(('0', '3', '6')):
                    matches.append((code, match.start(), match.end()))
        
        if matches:
            # 如果有多个匹配，选择第一个（通常是最先出现的）
            return matches[0][0]
        
        # 第二步：如果正则匹配失败，使用LLM进行意图识别
        print(f"[意图识别] 正则匹配失败，开始使用LLM进行意图识别，查询内容: {query}")
        try:
            import requests
            ollama_url = 'http://localhost:11434/api/generate'
            
            # 构建股票意图识别提示词（优化提示词，增加更多示例）
            prompt = f"""你是一个专业的股票助手。请从用户的问题中识别出股票名称或股票代码，并返回对应的6位股票代码。

用户问题：{query}

请按照以下规则识别：
1. 如果问题中包含6位数字的股票代码（以0、3、6开头），直接返回该代码
2. 如果问题中包含股票名称，请识别出股票名称并返回对应的6位股票代码
   - 示例："常山北明" → 000158
   - 示例："贵州茅台" → 600519
   - 示例："平安银行" → 000001
   - 示例："长亮科技" → 300348
   - 示例："万科A" → 000002
3. 如果无法识别，返回"UNKNOWN"

重要提示：
- 只返回6位数字的股票代码，不要包含任何其他文字、标点符号或说明
- 股票代码必须以0、3或6开头
- 如果识别出股票名称，必须返回对应的准确股票代码
- 格式示例：300348（不要写成"300348"或"代码：300348"）

股票代码："""
            
            payload = {
                'model': 'qwen2.5:32b',
                'prompt': prompt,
                'stream': False
            }
            
            print(f"[意图识别] 正在调用Ollama API...")
            try:
                response = requests.post(ollama_url, json=payload, timeout=15)
                print(f"[意图识别] Ollama API响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    stock_code = result.get('response', '').strip()
                    print(f"[意图识别] LLM原始返回: {stock_code[:200]}")
                    
                    # 清理返回结果，提取股票代码
                    stock_code = stock_code.replace('```', '').replace('\n', '').replace(' ', '').strip()
                    
                    # 验证是否为有效的6位股票代码
                    code_match = re.search(r'([0-9]{6})', stock_code)
                    if code_match:
                        extracted_code = code_match.group(1)
                        print(f"[意图识别] 提取的股票代码: {extracted_code}")
                        
                        # 验证股票代码格式（以0、3、6开头）
                        if extracted_code.startswith(('0', '3', '6')):
                            print(f"[意图识别] ✅ LLM成功识别出股票代码: {extracted_code}")
                            return extracted_code
                        else:
                            print(f"[意图识别] ❌ 股票代码格式不正确（不以0、3、6开头）: {extracted_code}")
                    else:
                        print(f"[意图识别] ❌ 无法从LLM返回中提取6位数字代码")
                    
                    # 如果返回的是UNKNOWN或无法识别
                    if 'UNKNOWN' in stock_code.upper() or len(stock_code) < 6:
                        print(f"[意图识别] ❌ LLM无法识别股票代码，返回: {stock_code[:50]}")
                        return None
                    
                    print(f"[意图识别] ❌ LLM返回格式不正确: {stock_code[:100]}")
                    return None
                else:
                    error_detail = ''
                    try:
                        error_detail = response.text[:200]
                    except:
                        pass
                    print(f"[意图识别] ❌ Ollama API返回错误: {response.status_code}, 详情: {error_detail}")
                    return None
            except requests.exceptions.ConnectionError:
                print(f"[意图识别] ❌ 无法连接到Ollama服务，请确保Ollama已启动并运行在 http://localhost:11434")
                return None
            except requests.exceptions.Timeout:
                print(f"[意图识别] ❌ LLM识别超时（超过15秒）")
                return None
            except Exception as e:
                import traceback
                print(f"[意图识别] ❌ LLM识别出错: {str(e)}")
                print(f"[意图识别] 错误详情: {traceback.format_exc()}")
                return None
        except Exception as e:
            import traceback
            print(f"[意图识别] ❌ 调用LLM失败: {str(e)}")
            print(f"[意图识别] 错误详情: {traceback.format_exc()}")
            return None
    
    def get_rule_engine(self):
        """获取规则引擎（用于规则可视化）"""
        return self.rule_engine
    
    def _clean_dict_for_json(self, data: Any, visited: set = None, max_depth: int = 50) -> Any:
        """
        清理字典中的NaN值，使其可以JSON序列化
        添加循环引用检测和深度限制，避免无限递归
        
        Args:
            data: 要清理的数据（可以是dict、list、基本类型）
            visited: 已访问对象的ID集合（用于检测循环引用）
            max_depth: 最大递归深度（防止过深嵌套）
        
        Returns:
            清理后的数据
        """
        # 初始化visited集合（如果是第一次调用）
        if visited is None:
            visited = set()
        
        # 深度限制检查
        if max_depth <= 0:
            return "[深度超限]"
        
        # 对于可变对象（dict、list），检查是否已访问过（循环引用检测）
        is_mutable = isinstance(data, (dict, list))
        data_id = None
        if is_mutable:
            data_id = id(data)
            if data_id in visited:
                # 检测到循环引用，返回占位符
                return "[循环引用]"
            visited.add(data_id)
        
        try:
            if isinstance(data, dict):
                result = {}
                for k, v in data.items():
                    # 跳过不可序列化的键
                    if not isinstance(k, (str, int, float, bool, type(None))):
                        continue
                    # 跳过不可序列化的值类型（函数、类、模块等）
                    if callable(v) or isinstance(v, type):
                        continue
                    try:
                        result[k] = self._clean_dict_for_json(v, visited, max_depth - 1)
                    except RecursionError:
                        print(f"[ERROR] _clean_dict_for_json: 处理键 {k} 时递归错误，跳过")
                        continue
                    except Exception as e:
                        print(f"[WARN] _clean_dict_for_json: 处理键 {k} 时出错: {e}，跳过")
                        continue
                return result
            elif isinstance(data, list):
                result = []
                for item in data:
                    # 跳过不可序列化的项
                    if callable(item) or isinstance(item, type):
                        continue
                    try:
                        result.append(self._clean_dict_for_json(item, visited, max_depth - 1))
                    except RecursionError:
                        print(f"[ERROR] _clean_dict_for_json: 处理列表项时递归错误，跳过")
                        continue
                    except Exception as e:
                        print(f"[WARN] _clean_dict_for_json: 处理列表项时出错: {e}，跳过")
                        continue
                return result
            elif isinstance(data, (float, np.floating)):
                if pd.isna(data) or np.isnan(data):
                    return None
                return float(data)
            elif isinstance(data, (int, np.integer)):
                return int(data)
            elif isinstance(data, np.ndarray):
                return [self._clean_dict_for_json(item, visited, max_depth - 1) for item in data.tolist()]
            elif pd.isna(data):
                return None
            else:
                # 对于其他类型（如datetime等），尝试转换为字符串
                try:
                    # 尝试JSON序列化常见类型
                    if hasattr(data, 'isoformat'):  # datetime对象
                        return data.isoformat()
                    elif hasattr(data, '__dict__'):
                        # 如果是对象，尝试转换为字典
                        try:
                            return self._clean_dict_for_json(data.__dict__, visited, max_depth - 1)
                        except:
                            return str(data)
                    return str(data)
                except:
                    return None
        except RecursionError as e:
            print(f"[ERROR] _clean_dict_for_json: 递归错误")
            return None
        except Exception as e:
            print(f"[WARN] _clean_dict_for_json: 处理数据时出错: {e}")
            return None
        finally:
            # 确保在异常情况下也移除visited标记
            if is_mutable and data_id is not None:
                visited.discard(data_id)

