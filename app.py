"""
Flask应用主入口
"""
import os
import sys
import io

# 设置UTF-8编码环境（在导入其他模块之前）
if sys.platform == 'win32':
    # Windows系统设置UTF-8编码
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # 重新配置标准输出流的编码
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    # 如果reconfigure不可用，使用TextIOWrapper包装
    elif not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time
import uuid

# 配置日志（确保UTF-8编码）
logging.basicConfig(
    level=logging.INFO,  # 设置为INFO级别，可以看到所有INFO及以上级别的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# 确保日志处理器使用UTF-8编码
for handler in logging.root.handlers:
    if hasattr(handler, 'stream') and hasattr(handler.stream, 'reconfigure'):
        try:
            handler.stream.reconfigure(encoding='utf-8')
        except:
            pass

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.agent.stock_agent import StockAgent
from src.visualization.rule_visualizer import RuleVisualizer
from config.settings import HOST, PORT, DEBUG

app = Flask(__name__)
CORS(app)

# 初始化智能体（自动初始化账户，添加5只股票的持仓）
agent = StockAgent(auto_init_account=True)
visualizer = RuleVisualizer(agent.get_rule_engine())

# 训练进度存储（用于实时进度更新）
training_progress_store = {}


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/multi-model-predict', methods=['POST'])
def multi_model_predict():
    """多模型预测API"""
    try:
        data = request.get_json()
        stock_code = data.get('stock_code', '').strip()
        
        if not stock_code:
            return jsonify({
                'success': False,
                'message': '股票代码不能为空'
            }), 400
        
        # 导入多模型预测器
        from src.prediction.multi_model_predictor import MultiModelPredictor
        from src.data.stock_data import StockDataProvider
        from src.data.technical_indicators import TechnicalIndicators
        from config.settings import LOOKBACK_DAYS, PREDICTION_DAYS
        
        # 获取股票数据（确保获取足够的数据用于训练）
        data_provider = StockDataProvider()
        # 获取更多数据以确保有足够的历史数据用于训练
        required_days = LOOKBACK_DAYS + PREDICTION_DAYS + 10
        stock_data = data_provider.get_stock_data(stock_code, days=required_days)
        # 检查数据来源
        data_source = 'unknown'
        if stock_data is not None and not stock_data.empty:
            if '_data_source' in stock_data.columns:
                data_source = stock_data['_data_source'].iloc[0] if len(stock_data) > 0 else 'unknown'
            # 移除内部标记列，避免传递给前端
            if '_data_source' in stock_data.columns:
                stock_data = stock_data.drop(columns=['_data_source'])
        
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
        
        if stock_data is None or stock_data.empty:
            return jsonify({
                'success': False,
                'message': f'无法获取股票 {stock_code} 的数据'
            }), 400
        
        # 添加技术指标
        stock_data = TechnicalIndicators.add_all_indicators(stock_data)
        
        # 创建训练进度ID
        progress_id = str(uuid.uuid4())
        
        # 初始化训练进度
        training_progress_store[progress_id] = {
            'status': 'initializing',
            'progress': 0,
            'message': '正在初始化...',
            'models': {},
            'current_step': 'initializing',
            'start_time': time.time()
        }
        
        # 定义进度回调函数
        def progress_callback(progress_info):
            if progress_id in training_progress_store:
                training_progress_store[progress_id].update({
                    'status': 'training',
                    'progress': progress_info.get('progress', 0),
                    'message': progress_info.get('message', ''),
                    'current_step': progress_info.get('step', ''),
                    'current_model': progress_info.get('model_name_cn', ''),
                })
                
                # 更新单个模型的状态
                if 'model_name' in progress_info:
                    model_name = progress_info['model_name']
                    if model_name not in training_progress_store[progress_id]['models']:
                        training_progress_store[progress_id]['models'][model_name] = {
                            'name_cn': progress_info.get('model_name_cn', model_name),
                            'status': 'pending',
                            'message': ''
                        }
                    
                    training_progress_store[progress_id]['models'][model_name].update({
                        'status': progress_info.get('current_model_status', 'training'),
                        'message': progress_info.get('message', ''),
                        'val_r2': progress_info.get('val_r2'),
                        'error': progress_info.get('error')
                    })
        
        # 初始化多模型预测器（传入进度回调）
        predictor = MultiModelPredictor(
            lookback_days=LOOKBACK_DAYS,
            prediction_days=PREDICTION_DAYS,
            progress_callback=progress_callback
        )
        
        # 训练模型（尽量训练，即使数据不足也尝试）
        training_results = {}
        min_required = LOOKBACK_DAYS + PREDICTION_DAYS + 3  # 降低最小要求
        min_recommended = LOOKBACK_DAYS + PREDICTION_DAYS + 10
        
        print(f"[多模型预测] 数据量检查：实际{len(stock_data)}天，最小要求{min_required}天，推荐{min_recommended}天")
        
        if len(stock_data) >= min_required:
            try:
                # 根据数据量调整验证集比例
                if len(stock_data) >= 100:
                    validation_split = 0.2
                elif len(stock_data) >= 70:
                    validation_split = 0.15
                else:
                    validation_split = 0.1
                
                print(f"[多模型预测] 开始训练模型，验证集比例: {validation_split}")
                training_results = predictor.train_models(stock_data, validation_split=validation_split)
                trained_count = len([k for k in training_results.keys() if k != 'error' and training_results[k] and 'error' not in str(training_results[k]).lower()])
                print(f"[多模型预测] 模型训练完成，成功训练了 {trained_count} 个模型")
                
                # 更新最终状态
                if progress_id in training_progress_store:
                    training_progress_store[progress_id].update({
                        'status': 'completed',
                        'progress': 100,
                        'message': f'训练完成，成功训练了 {trained_count} 个模型',
                        'trained_count': trained_count
                    })
            except Exception as e:
                print(f"[多模型预测] 训练模型时出错: {e}")
                import traceback
                traceback.print_exc()
                training_results = {}
        else:
            print(f"[多模型预测] 数据不足（需要至少{min_required}天，实际{len(stock_data)}天），将使用简单预测方法")
        
        # 进行预测（如果模型未训练，会使用回退方法）
        try:
            predictions_all = predictor.predict_all_models(stock_data)
        except Exception as e:
            print(f"[多模型预测] 预测过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果预测失败，返回空预测结果，但不要完全失败
            predictions_all = {}
        
        # 清理NaN值
        cleaned_predictions = {}
        for model_name, pred in predictions_all.items():
            try:
                cleaned_predictions[model_name] = agent._clean_dict_for_json(pred)
            except Exception as e:
                print(f"[多模型预测] 清理模型 {model_name} 的预测结果时出错: {e}")
                import traceback
                traceback.print_exc()
                # 如果清理失败，使用简单的回退值
                cleaned_predictions[model_name] = {
                    'predicted_prices': [],
                    'predicted_return': 0.0,
                    'confidence': 0.5,
                    'error': str(e)
                }
        
        # 统计训练状态
        trained_count = len([k for k in training_results.keys() 
                           if training_results[k] and 'error' not in training_results[k] 
                           and isinstance(training_results[k], dict) 
                           and ('train_mae' in training_results[k] or 'val_mae' in training_results[k])])
        total_models = len(predictor.model_names)
        
        # 获取数据状态信息
        data_days = len(stock_data)
        error_info = training_results.get('error', {})
        if isinstance(error_info, dict):
            data_days = error_info.get('data_days', data_days)
            min_required = error_info.get('min_required', min_required)
            min_recommended = error_info.get('min_recommended', min_recommended)
        
        # 准备返回数据
        try:
            # 清理训练结果
            cleaned_training_results = agent._clean_dict_for_json(training_results)
        except Exception as e:
            print(f"[多模型预测] 清理训练结果时出错: {e}")
            import traceback
            traceback.print_exc()
            cleaned_training_results = {}
        
        # 获取模型信息
        try:
            model_info = predictor.get_model_info()
        except Exception as e:
            print(f"[多模型预测] 获取模型信息时出错: {e}")
            model_info = {'model_count': total_models}
        
        print(f"[多模型预测] 准备返回数据: predictions数量={len(cleaned_predictions)}, trained_count={trained_count}, total_models={total_models}")
        
        # 尝试构建响应
        try:
            response_data = {
                'success': True,
                'stock_code': stock_code,
                'predictions': cleaned_predictions,
                'training_results': cleaned_training_results,
                'model_info': model_info,
                'training_status': {
                    'trained_count': trained_count,
                    'total_models': total_models,
                    'has_training': trained_count > 0,
                    'training_progress': int((trained_count / total_models) * 100) if total_models > 0 else 0,
                    'data_days': data_days,
                    'min_required': min_required,
                    'min_recommended': min_recommended,
                    'data_sufficient': data_days >= min_required
                },
                'progress_id': progress_id  # 返回进度ID，用于前端轮询
            }
            
            # 尝试JSON序列化，如果失败则使用备用方法
            try:
                import json
                # 测试JSON序列化
                json.dumps(response_data, default=str)
                print(f"[多模型预测] JSON序列化测试成功")
            except Exception as json_error:
                print(f"[多模型预测] JSON序列化测试失败: {json_error}")
                import traceback
                traceback.print_exc()
                # 如果JSON序列化失败，尝试更激进的清理
                response_data = _clean_nan_values(response_data)
            
            return jsonify(response_data)
        except Exception as e:
            print(f"[多模型预测] 构建响应时出错: {e}")
            import traceback
            traceback.print_exc()
            # 即使出错，也返回一个基本的响应
            return jsonify({
                'success': True,
                'stock_code': stock_code,
                'predictions': {},
                'training_results': {},
                'model_info': {'model_count': total_models},
                'training_status': {
                    'trained_count': trained_count,
                    'total_models': total_models,
                    'has_training': trained_count > 0,
                    'training_progress': int((trained_count / total_models) * 100) if total_models > 0 else 0,
                    'data_days': data_days,
                    'min_required': min_required,
                    'min_recommended': min_recommended,
                    'data_sufficient': data_days >= min_required
                },
                'progress_id': progress_id,
                'error': f'构建响应时出错: {str(e)}'
            })
        
    except ImportError as e:
        # 导入错误（可能是缺少依赖）
        import traceback
        error_trace = traceback.format_exc()
        print(f"多模型预测导入错误: {error_trace}")
        return jsonify({
            'success': False,
            'message': f'导入模块失败: {str(e)}。请检查是否安装了所有必需的依赖包（如XGBoost、LightGBM等）。'
        }), 500
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"多模型预测错误: {error_trace}")
        # 返回更详细的错误信息
        error_message = str(e)
        if 'MultiModelPredictor' in error_message or 'multi_model_predictor' in error_message:
            error_message = f'无法加载多模型预测器: {error_message}。请检查模块是否正确安装。'
        return jsonify({
            'success': False,
            'message': f'预测过程中出错: {error_message}'
        }), 500


@app.route('/api/query', methods=['POST'])
def handle_query():
    """处理用户查询"""
    logger = logging.getLogger(__name__)
    logger.info("=== handle_query 函数被调用 ===")
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        logger.info(f"收到查询: {query}")
        
        if not query:
            return jsonify({
                'success': False,
                'message': '查询内容不能为空'
            }), 400
        
        # 处理查询
        result = agent.process_query(query)
        
        # 如果成功，添加可视化信息
        if result.get('success'):
            decision = result.get('decision', {})
            # 添加规则评估流程
            if 'rule_evaluation' in decision:
                result['visualization'] = visualizer.generate_evaluation_flow(
                    decision['rule_evaluation']
                )
            # 添加决策解释
            result['explanation'] = visualizer.format_decision_explanation(decision)
            
            # 只要识别到股票代码，就标记需要显示多模型预测Tab（但不立即训练，等用户点击Tab时再训练）
            if result.get('stock_code'):
                # 不在这里训练模型，而是标记需要显示Tab，训练将在用户点击Tab时触发
                result['has_multi_model_prediction'] = True
                result['multi_model_prediction'] = {
                    'predictions': {},
                    'training_results': {},
                    'model_info': {'model_count': 0},
                    'training_status': {
                        'trained_count': 0,
                        'total_models': 0,
                        'has_training': False,
                        'training_progress': 0,
                        'data_days': 0,
                        'min_required': 68,
                        'min_recommended': 75,
                        'data_sufficient': False
                    },
                    'progress_id': None,
                    'stock_code': result.get('stock_code')  # 保存股票代码，用于后续训练
                }
            else:
                # 如果没有股票代码，不显示多模型预测
                result['has_multi_model_prediction'] = False
        
        # 确保返回的数据可以JSON序列化
        try:
            return jsonify(result)
        except TypeError as e:
            # 如果JSON序列化失败，尝试清理数据
            import json
            try:
                # 使用自定义的JSON编码器处理NaN
                class NaNEncoder(json.JSONEncoder):
                    def encode(self, obj):
                        if isinstance(obj, float) and (pd.isna(obj) or np.isnan(obj)):
                            return 'null'
                        return super().encode(obj)
                
                # 手动清理NaN值
                cleaned_result = _clean_nan_values(result)
                return jsonify(cleaned_result)
            except Exception as e2:
                return jsonify({
                    'success': False,
                    'message': f'数据序列化失败: {str(e2)}'
                }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'服务器错误: {str(e)}'
        }), 500


@app.route('/api/training-progress/<progress_id>', methods=['GET'])
def get_training_progress(progress_id):
    """获取训练进度"""
    if progress_id in training_progress_store:
        progress = training_progress_store[progress_id].copy()
        # 计算已用时间
        if 'start_time' in progress:
            elapsed_time = time.time() - progress['start_time']
            progress['elapsed_time'] = int(elapsed_time)
        return jsonify({
            'success': True,
            'progress': progress
        })
    else:
        return jsonify({
            'success': False,
            'message': '进度ID不存在'
        }), 404


@app.route('/api/rules', methods=['GET'])
def get_rules():
    """获取所有规则信息"""
    try:
        rule_info = visualizer.generate_rule_info()
        return jsonify({
            'success': True,
            'data': rule_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取规则失败: {str(e)}'
        }), 500


@app.route('/api/account', methods=['GET'])
def get_account():
    """获取账户信息"""
    try:
        account = agent.account
        from src.data.account_initializer import AccountInitializer
        initializer = AccountInitializer()
        summary = initializer.get_account_summary(account)
        return jsonify({
            'success': True,
            'data': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取账户信息失败: {str(e)}'
        }), 500


@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """获取所有预设场景"""
    try:
        from src.scenarios.rule_scenarios import RuleScenarioDemo
        demo = RuleScenarioDemo()
        scenarios = demo.get_all_scenarios()
        return jsonify({
            'success': True,
            'data': scenarios
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取场景失败: {str(e)}'
        }), 500


@app.route('/api/scenarios/<scenario_id>', methods=['POST'])
def run_scenario(scenario_id):
    """运行指定场景"""
    try:
        from src.scenarios.rule_scenarios import RuleScenarioDemo
        demo = RuleScenarioDemo()
        scenarios = demo.get_all_scenarios()
        
        scenario = next((s for s in scenarios if s['id'] == scenario_id), None)
        if not scenario:
            return jsonify({
                'success': False,
                'message': f'场景 {scenario_id} 不存在'
            }), 404
        
        result = demo.run_scenario(scenario)
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'运行场景失败: {str(e)}'
        }), 500


@app.route('/api/scenarios/run-all', methods=['POST'])
def run_all_scenarios():
    """运行所有场景"""
    try:
        from src.scenarios.rule_scenarios import RuleScenarioDemo
        demo = RuleScenarioDemo()
        results = demo.run_all_scenarios()
        return jsonify({
            'success': True,
            'data': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'运行场景失败: {str(e)}'
        }), 500


@app.route('/api/decision-explanation', methods=['POST'])
def decision_explanation():
    """使用AI生成决策解释（结合规则评估和技术指标）"""
    try:
        import requests
        import json
        
        data = request.get_json()
        stock_code = data.get('stock_code', '')
        decision = data.get('decision', {})
        rule_evaluation = data.get('rule_evaluation', {})
        technical_indicators = data.get('technical_indicators', [])
        
        # 构建决策解释提示词
        prompt = build_decision_explanation_prompt(
            stock_code, decision, rule_evaluation, technical_indicators
        )
        
        # 调用Ollama API
        ollama_url = 'http://localhost:11434/api/generate'
        payload = {
            'model': 'qwen2.5:32b',
            'prompt': prompt,
            'stream': False
        }
        
        try:
            response = requests.post(ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                explanation = result.get('response', '').strip()
                
                # 清理分析文本（移除可能的markdown格式）
                if explanation.startswith('```'):
                    lines = explanation.split('\n')
                    explanation = '\n'.join([l for l in lines if not l.startswith('```')])
                
                # 如果解释为空，生成默认解释
                if not explanation or len(explanation) < 10:
                    explanation = generate_default_explanation(decision, rule_evaluation, technical_indicators)
                
                return jsonify({
                    'success': True,
                    'explanation': explanation
                })
            else:
                error_msg = f'Ollama API返回错误: {response.status_code}'
                try:
                    error_detail = response.text[:100]
                    error_msg += f' ({error_detail})'
                except:
                    pass
                # 即使Ollama出错，也返回默认解释，而不是失败
                explanation = generate_default_explanation(decision, rule_evaluation, technical_indicators)
                return jsonify({
                    'success': True,
                    'explanation': explanation
                })
        except requests.exceptions.ConnectionError:
            # 如果Ollama未启动，生成默认解释
            explanation = generate_default_explanation(decision, rule_evaluation, technical_indicators)
            return jsonify({
                'success': True,
                'explanation': explanation
            })
        except requests.exceptions.Timeout:
            # 超时时返回默认解释
            explanation = generate_default_explanation(decision, rule_evaluation, technical_indicators)
            return jsonify({
                'success': True,
                'explanation': explanation
            })
        except Exception as e:
            # 出错时返回默认解释
            explanation = generate_default_explanation(decision, rule_evaluation, technical_indicators)
            return jsonify({
                'success': True,
                'explanation': explanation
            })
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"决策解释生成错误: {error_trace}")
        # 即使出错，也尝试返回默认解释
        try:
            explanation = generate_default_explanation(decision, rule_evaluation, technical_indicators)
            return jsonify({
                'success': True,
                'explanation': explanation
            })
        except:
            return jsonify({
                'success': False,
                'explanation': f'服务器错误: {str(e)}'
            }), 500


def build_decision_explanation_prompt(stock_code, decision, rule_evaluation, technical_indicators):
    """构建决策解释提示词"""
    
    # 提取决策信息
    action = decision.get('action', '未知')
    current_price = decision.get('current_price', 0)
    suggestion = decision.get('suggestion', '')
    prediction = decision.get('prediction', {})
    
    # 提取规则评估信息
    triggered_rules = rule_evaluation.get('triggered_rules', [])
    is_allowed = rule_evaluation.get('is_allowed', True)
    warnings = rule_evaluation.get('warnings', [])
    optimizations = rule_evaluation.get('optimizations', [])
    
    prompt = f"""你是一位专业的股票投资顾问。请根据以下信息，为股票{stock_code}生成一份综合决策分析报告。

【决策建议】
操作：{action}
当前价格：￥{current_price:.2f}
建议：{suggestion}

【预测信息】
预测趋势：{prediction.get('trend', '未知')}
预期收益率：{prediction.get('predicted_return_pct', 0):.2f}%
置信度：{prediction.get('confidence', 0)*100:.1f}%
风险等级：{prediction.get('risk_level', '中')}

【规则评估】
规则评估结果：{'允许交易' if is_allowed else '禁止交易'}
触发规则数量：{len(triggered_rules)}条
"""
    
    if triggered_rules:
        prompt += "\n触发的规则：\n"
        for rule in triggered_rules:
            prompt += f"- {rule.get('rule_name', '')}：{rule.get('description', '')}\n"
    
    if warnings:
        prompt += "\n警告信息：\n"
        for warning in warnings:
            prompt += f"- {warning.get('message', '')}\n"
    
    if optimizations:
        prompt += "\n优化建议：\n"
        for opt in optimizations:
            prompt += f"- {opt.get('message', '')}\n"
    
    if technical_indicators:
        prompt += "\n【技术指标分析】\n"
        for indicator in technical_indicators:
            prompt += f"- {indicator}\n"
    
    prompt += """
请综合以上所有信息，生成一份专业的决策分析报告。报告应该：
1. 总结规则评估的关键发现
2. 分析技术指标的信号
3. 综合考虑预测、规则和技术指标，给出最终决策建议
4. 说明决策的理由和风险提示
5. 语言简洁专业，控制在300字以内

请直接输出分析报告，不要包含额外说明。
"""
    
    return prompt


def generate_default_explanation(decision, rule_evaluation, technical_indicators):
    """生成默认决策解释（当AI不可用时）"""
    parts = []
    
    action = decision.get('action', '未知')
    current_price = decision.get('current_price', 0)
    suggestion = decision.get('suggestion', '')
    
    parts.append(f"根据综合分析，建议对当前股票执行【{action}】操作。当前价格为￥{current_price:.2f}。")
    
    # 规则评估总结
    triggered_rules = rule_evaluation.get('triggered_rules', [])
    if triggered_rules:
        parts.append(f"规则评估显示触发了{len(triggered_rules)}条规则，{'允许' if rule_evaluation.get('is_allowed', True) else '禁止'}进行交易。")
    
    # 技术指标总结
    if technical_indicators:
        parts.append("技术指标分析显示多个指标提供了交易信号。")
    
    # 最终建议
    parts.append(f"综合建议：{suggestion}")
    
    if rule_evaluation.get('warnings'):
        parts.append("请注意相关警告信息，谨慎决策。")
    
    return "\n".join(parts)


@app.route('/api/stock-intent-recognition', methods=['POST'])
def stock_intent_recognition():
    """使用LLM识别股票名称并转换为股票代码"""
    try:
        import requests
        import json
        
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'message': '查询内容不能为空'
            }), 400
        
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
        
        # 调用Ollama API
        ollama_url = 'http://localhost:11434/api/generate'
        payload = {
            'model': 'qwen2.5:32b',
            'prompt': prompt,
            'stream': False
        }
        
        try:
            response = requests.post(ollama_url, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                stock_code = result.get('response', '').strip()
                
                # 清理返回结果，提取股票代码
                # 移除可能的markdown格式、换行符等
                stock_code = stock_code.replace('```', '').replace('\n', '').strip()
                
                # 验证是否为有效的6位股票代码
                import re
                code_match = re.search(r'([0-9]{6})', stock_code)
                if code_match:
                    extracted_code = code_match.group(1)
                    # 验证股票代码格式（以0、3、6开头）
                    if extracted_code.startswith(('0', '3', '6')):
                        return jsonify({
                            'success': True,
                            'stock_code': extracted_code,
                            'method': 'llm'
                        })
                
                # 如果返回的是UNKNOWN或无法识别
                if 'UNKNOWN' in stock_code.upper() or len(stock_code) < 6:
                    return jsonify({
                        'success': False,
                        'message': '无法识别股票代码或股票名称',
                        'stock_code': None
                    })
                
                # 如果返回了其他内容，尝试提取
                return jsonify({
                    'success': False,
                    'message': f'LLM返回格式不正确: {stock_code[:50]}',
                    'stock_code': None
                })
            else:
                error_msg = f'Ollama API返回错误: {response.status_code}'
                try:
                    error_detail = response.text[:100]
                    error_msg += f' ({error_detail})'
                except:
                    pass
                return jsonify({
                    'success': False,
                    'message': error_msg,
                    'stock_code': None
                })
        except requests.exceptions.ConnectionError:
            return jsonify({
                'success': False,
                'message': '无法连接到Ollama服务，请确保Ollama已启动并运行在 http://localhost:11434',
                'stock_code': None
            })
        except requests.exceptions.Timeout:
            return jsonify({
                'success': False,
                'message': 'LLM识别超时，请稍后重试',
                'stock_code': None
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'LLM识别出错: {str(e)}',
                'stock_code': None
            })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"股票意图识别错误: {error_trace}")
        return jsonify({
            'success': False,
            'message': f'处理过程中出错: {str(e)}',
            'stock_code': None
        }), 500


@app.route('/api/ai-analysis', methods=['POST'])
def ai_analysis():
    """使用Ollama进行技术指标AI分析"""
    try:
        import requests
        import json
        
        data = request.get_json()
        chart_type = data.get('chart_type', '')
        stock_code = data.get('stock_code', '')
        data_summary = data.get('data_summary', {})
        
        # 构建分析提示词
        prompt = build_analysis_prompt(chart_type, stock_code, data_summary)
        
        # 调用Ollama API
        ollama_url = 'http://localhost:11434/api/generate'
        payload = {
            'model': 'qwen2.5:32b',
            'prompt': prompt,
            'stream': False
        }
        
        try:
            response = requests.post(ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', '').strip()
                
                # 清理分析文本（移除可能的markdown格式）
                if analysis.startswith('```'):
                    lines = analysis.split('\n')
                    analysis = '\n'.join([l for l in lines if not l.startswith('```')])
                
                # 移除多余的换行和空格
                analysis = ' '.join(analysis.split())
                
                # 如果分析为空，返回默认分析
                if not analysis or len(analysis) < 5:
                    if chart_type == 'MACD指标':
                        macd_val = data_summary.get('macd_value', 'N/A')
                        signal_val = data_summary.get('signal_value', 'N/A')
                        cross = data_summary.get('cross_signal', 'N/A')
                        analysis = f"根据MACD指标数据，当前MACD值={macd_val}，信号线={signal_val}，{cross}。建议结合其他指标综合判断。"
                    else:
                        analysis = f"根据{chart_type}数据，当前指标状态正常。建议结合其他指标综合判断。"
                
                return jsonify({
                    'success': True,
                    'analysis': analysis
                })
            else:
                error_msg = f'Ollama API返回错误: {response.status_code}'
                try:
                    error_detail = response.text[:100]
                    error_msg += f' ({error_detail})'
                except:
                    pass
                return jsonify({
                    'success': False,
                    'analysis': error_msg
                })
        except requests.exceptions.ConnectionError:
            return jsonify({
                'success': False,
                'analysis': '无法连接到Ollama服务，请确保Ollama已启动并运行在 http://localhost:11434'
            })
        except requests.exceptions.Timeout:
            return jsonify({
                'success': False,
                'analysis': 'AI分析超时，请稍后重试'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'analysis': f'AI分析失败: {str(e)}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'analysis': f'服务器错误: {str(e)}'
        }), 500


def build_analysis_prompt(chart_type: str, stock_code: str, data_summary: dict) -> str:
    """构建AI分析提示词"""
    
    base_prompt = f"""你是一位专业的股票技术分析师。请分析股票{stock_code}的{chart_type}数据，给出专业、简洁的技术分析。

数据摘要：
"""
    
    if chart_type == 'K线图':
        prompt = base_prompt + f"""
- 当前价格: {data_summary.get('current_price', 'N/A')}元
- 价格变化: {data_summary.get('price_change', 'N/A')}%
- MA5趋势: {data_summary.get('ma5_trend', 'N/A')}
- MA10趋势: {data_summary.get('ma10_trend', 'N/A')}
- MA20趋势: {data_summary.get('ma20_trend', 'N/A')}

请分析K线形态、移动平均线排列、价格趋势，给出买入/卖出/持有的建议。回答要简洁专业，控制在100字以内。
"""
    elif chart_type == 'MACD指标':
        macd_val = data_summary.get('macd_value', 'N/A')
        signal_val = data_summary.get('signal_value', 'N/A')
        hist_val = data_summary.get('histogram_value', 'N/A')
        cross = data_summary.get('cross_signal', 'N/A')
        
        prompt = base_prompt + f"""
- MACD值: {macd_val}
- 信号线值: {signal_val}
- 柱状图值: {hist_val}
- 交叉信号: {cross}

请分析MACD指标的金叉/死叉、趋势强度，给出交易建议。回答要简洁专业，控制在100字以内。
"""
    elif chart_type == 'RSI指标':
        prompt = base_prompt + f"""
- RSI值: {data_summary.get('rsi_value', 'N/A')}
- 信号: {data_summary.get('signal', 'N/A')}

请分析RSI的超买超卖状态、趋势强度，给出交易建议。回答要简洁专业，控制在100字以内。
"""
    elif chart_type == '布林带指标':
        prompt = base_prompt + f"""
- 当前价格: {data_summary.get('current_price', 'N/A')}元
- 上轨: {data_summary.get('upper_band', 'N/A')}元
- 下轨: {data_summary.get('lower_band', 'N/A')}元
- 价格位置: {data_summary.get('position', 'N/A')}

请分析价格在布林带中的位置、突破情况，给出交易建议。回答要简洁专业，控制在100字以内。
"""
    elif chart_type == 'KDJ指标':
        prompt = base_prompt + f"""
- K值: {data_summary.get('k_value', 'N/A')}
- D值: {data_summary.get('d_value', 'N/A')}
- J值: {data_summary.get('j_value', 'N/A')}
- 交叉情况: {data_summary.get('cross', 'N/A')}

请分析KDJ的金叉/死叉、超买超卖状态，给出交易建议。回答要简洁专业，控制在100字以内。
"""
    elif chart_type == '成交量':
        prompt = base_prompt + f"""
- 当前成交量: {data_summary.get('current_volume', 'N/A')}
- 平均成交量: {data_summary.get('avg_volume', 'N/A')}
- 量比: {data_summary.get('volume_ratio', 'N/A')}

请分析成交量的变化、量价关系，给出交易建议。回答要简洁专业，控制在100字以内。
"""
    else:
        prompt = base_prompt + f"数据: {json.dumps(data_summary, ensure_ascii=False)}\n\n请给出专业的技术分析。回答要简洁专业，控制在100字以内。"
    
    return prompt


def _clean_nan_values(obj):
    """递归清理对象中的NaN值"""
    if isinstance(obj, dict):
        return {k: _clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nan_values(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        if pd.isna(obj) or np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return [_clean_nan_values(item) for item in obj.tolist()]
    elif pd.isna(obj):
        return None
    else:
        return obj


# ============================================
# 应用启动说明
# ============================================
# 本文件（app.py）只定义Flask应用，不直接启动服务
# 
# 正确的启动方式：
#   1. 使用 main.py 启动（推荐）：
#      python main.py
#   
#   2. 使用启动脚本（Windows）：
#      start.bat
#
# 禁止直接运行 app.py，如需直接运行，请取消下面的注释：
# ============================================

# 如果直接运行 app.py，显示提示信息
if __name__ == '__main__':
    print("=" * 60)
    print("  警告：请不要直接运行 app.py")
    print("=" * 60)
    print("  请使用以下方式启动应用：")
    print("  1. python main.py")
    print("  2. 双击运行 start.bat")
    print("=" * 60)
    print("\n如果您确实需要直接运行 app.py，请取消 app.py 底部的注释。")
    print("但建议使用 main.py 作为统一入口。\n")
    import sys
    sys.exit(1)
    
    # 如需直接运行，取消下面的注释：
    # from config.settings import HOST, PORT, DEBUG
    # app.run(host=HOST, port=PORT, debug=DEBUG)

