# 股票交易智能体系统

一个基于AI的股票交易决策系统，集成多种机器学习算法进行价格预测，结合复杂业务规则引擎生成智能交易决策。

## 📋 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [快速开始](#快速开始)
- [技术架构](#技术架构)
- [功能详解](#功能详解)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [API接口](#api接口)
- [常见问题](#常见问题)
- [项目结构](#项目结构)

## 项目简介

这是一个股票交易智能体系统，能够：
- 🤖 **智能预测**：使用多种机器学习算法（随机森林、XGBoost、LightGBM、LSTM等）进行股票价格预测
- 📊 **技术分析**：自动计算MACD、RSI、KDJ、布林带等技术指标，并提供AI分析
- 🔍 **意图识别**：使用本地LLM（qwen2.5:32b）识别股票名称并转换为股票代码
- ⚖️ **规则引擎**：支持复杂业务规则，包括风控规则、成本优化规则等
- 🎯 **智能决策**：整合预测结果和业务规则，生成买入/卖出/持有建议
- 💡 **可解释性**：提供详细的决策理由和模型解释
- 🖥️ **友好界面**：现代化Web界面，支持文本问答和Tab页展示

## 核心功能

### 1. 多模型预测系统
- **7种机器学习算法**：
  - 随机森林回归（Random Forest）
  - 梯度提升回归（Gradient Boosting）
  - 线性回归（Linear Regression）
  - Ridge回归
  - Lasso回归
  - XGBoost（可选，需安装）
  - LightGBM（可选，需安装）
- **深度学习模型**（可选）：
  - LSTM神经网络模型
  - 支持自动训练和早停机制
- **进度可视化**：训练进度条动画，不同模型以不同速度展示

### 2. 技术指标分析
系统自动计算以下技术指标：
- **K线图 + 移动平均线**（MA5、MA10、MA20、MA60）
- **MACD**（指数平滑异同移动平均线）
- **RSI**（相对强弱指标）
- **布林带**（Bollinger Bands）
- **KDJ指标**
- **成交量分析**
- **AI分析**：使用Ollama本地模型为每个技术指标提供专业分析

### 3. AI意图识别
- 使用本地LLM（qwen2.5:32b）进行智能识别
- 支持股票名称识别（如："常山北明" → "000158"）
- 支持股票简称识别（如："茅台" → "600519"）
- 支持上下文理解

### 4. 复杂业务规则引擎
支持多类型业务规则：
- **风控规则**：
  - 单只股票持仓限制（不超过总资产30%）
  - 止损规则（亏损超过5%自动卖出）
  - 止盈规则（盈利超过10%部分止盈）
  - 账户风险控制（总亏损不超过10%）
  - 资金不足限制
  - 单日限额规则
- **成本优化规则**：
  - 大额交易成本优化
  - 高置信度优化建议
- **其他规则**：
  - 时间窗口规则
  - 市场状态规则

### 5. 决策引擎
采用7阶段决策流程：
1. **预检查阶段**：价格合理性检查
2. **综合风险评估**：多维度风险计算
3. **动态规则参数调整**：根据市场状态调整阈值
4. **规则依赖链评估**：分组执行规则（风控→优化）
5. **交易策略选择**：激进/稳健/保守策略
6. **动态资金计算**：基于多因素计算交易金额
7. **最终决策生成**：综合所有因素生成决策

### 6. 账户管理
- **自动初始化**：启动时自动从真实接口获取股票价格，创建模拟持仓
- **实时价格获取**：优先使用实时行情，失败时回退到历史数据
- **持仓管理**：支持持仓查询、盈亏计算

### 7. 可解释性分析
- 特征重要性分析
- 预测因素分析
- 风险因素识别
- 趋势分析
- 自然语言解释

## 快速开始

### 环境要求

- **Python版本**：Python 3.8 或更高版本
- **操作系统**：Windows / Linux / macOS

### 安装步骤

#### 1. 安装基础依赖

```bash
pip install -r requirements.txt
```

如果安装速度慢，可以使用国内镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 2. 安装可选依赖（推荐）

**XGBoost和LightGBM**（提升预测性能）：
```bash
# 使用国内镜像源安装（推荐）
pip install xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lightgbm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**TensorFlow**（用于LSTM模型）：
```bash
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**SHAP**（用于模型可解释性）：
```bash
pip install "shap<0.44"
```

**注意事项**：
- NumPy必须使用1.x版本（`numpy<2.0.0`），因为某些依赖还不支持NumPy 2.0
- SHAP必须使用<0.44版本，以兼容NumPy 1.x
- TA-Lib需要先安装C库（Windows用户请访问：https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib 或使用 `conda install -c conda-forge ta-lib`）

#### 3. 安装Ollama（用于AI分析）

**安装Ollama**：
1. 访问 https://ollama.ai 下载并安装
2. 下载qwen2.5:32b模型：
```bash
ollama pull qwen2.5:32b
```
3. 确保Ollama服务运行在 `http://localhost:11434`

**注意**：如果不安装Ollama，AI分析功能将不可用，但不影响其他核心功能。

### 运行系统

**推荐方式**（使用main.py）：
```bash
python main.py
```

**或者使用启动脚本**：
- Windows: 双击运行 `script/start.bat`
- Linux/Mac: 运行 `bash script/start.sh` 或 `chmod +x script/start.sh && ./script/start.sh`

**注意**：请不要直接运行 `app.py`，系统会提示您使用正确的启动方式。

启动成功后，在浏览器中访问：**http://localhost:5000**

## 技术架构

### 系统架构

```
用户查询
    ↓
StockAgent（智能体）
    ├── 股票代码提取（LLM意图识别）
    ├── 数据获取（StockDataProvider）
    ├── 技术指标计算（TechnicalIndicators）
    ├── 价格预测（MultiModelPredictor）
    │   ├── 随机森林
    │   ├── XGBoost
    │   ├── LightGBM
    │   └── ...（其他模型）
    ├── 规则评估（AdvancedRuleEngine）
    ├── 决策生成（AdvancedDecisionEngine）
    └── 结果展示（可视化）
```

### 技术栈

- **后端框架**：Flask 3.0.0
- **数据获取**：akshare 1.12.0
- **机器学习**：
  - scikit-learn 1.3.2
  - XGBoost（可选）
  - LightGBM（可选）
  - TensorFlow（可选，用于LSTM）
- **技术指标**：TA-Lib 0.4.28
- **可视化**：
  - Plotly 5.18.0（交互式图表）
  - Matplotlib 3.8.2
- **AI模型**：Ollama + qwen2.5:32b
- **前端**：HTML + JavaScript + Bootstrap
- **数据处理**：Pandas 2.1.4, NumPy <2.0.0

### 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 智能体 | `src/agent/stock_agent.py` | 主入口，整合所有模块 |
| 数据获取 | `src/data/stock_data.py` | 股票数据获取和当前价格 |
| 技术指标 | `src/data/technical_indicators.py` | 计算MACD、RSI、KDJ等 |
| 多模型预测 | `src/prediction/multi_model_predictor.py` | 集成7种算法进行预测 |
| LSTM预测 | `src/prediction/lstm_predictor.py` | 深度学习预测（可选） |
| 规则引擎 | `src/rule_engine/advanced_rule_engine.py` | 复杂规则评估 |
| 决策引擎 | `src/decision/advanced_decision_engine.py` | 7阶段决策流程 |
| 模型解释 | `src/explainability/model_explainer.py` | 可解释性分析 |
| 账户管理 | `src/data/account_initializer.py` | 账户初始化和持仓管理 |

## 功能详解

### 1. 多模型预测

系统同时使用7种机器学习算法进行预测，训练进度以动画形式展示，不同模型以不同速度完成训练。

**支持的模型**：
- Random Forest（随机森林回归）
- Gradient Boosting（梯度提升回归）
- Linear Regression（线性回归）
- Ridge（Ridge回归）
- Lasso（Lasso回归）
- XGBoost（需安装）
- LightGBM（需安装）

**预测输出**：
- 未来5天的价格预测
- 预期收益率
- 置信度
- 风险等级（低/中/高）
- 训练指标（R²、MAE、RMSE）

### 2. 技术指标分析

系统自动计算6种技术指标，每个指标配有专业图表和AI分析：

1. **K线图 + 移动平均线**
   - 显示开盘价、收盘价、最高价、最低价
   - MA5、MA10、MA20、MA60移动平均线

2. **MACD指标**
   - DIF、DEA、MACD柱状图
   - 金叉/死叉信号识别

3. **RSI指标**
   - RSI6、RSI12
   - 超买（>70）/超卖（<30）区域标识

4. **布林带**
   - 上轨、中轨、下轨
   - 价格突破识别

5. **KDJ指标**
   - K、D、J三条线
   - 金叉/死叉信号

6. **成交量**
   - 成交量柱状图
   - 量价关系分析

**AI分析**：每个指标都配有Ollama生成的AI分析，提供专业的市场解读。

### 3. 规则引擎

#### 规则类型

**风控规则（Risk Control）**：
- **RC001**：单只股票持仓限制（持仓比例>30%禁止买入）
- **RC002**：止损规则（亏损>5%建议卖出）
- **RC003**：账户风险控制（总亏损>10%禁止交易）
- **RC004**：资金不足限制（可用资金<1000元禁止交易）
- **RC005**：止盈规则（盈利>10%且持仓>10%建议止盈）
- **RC006**：单日限额规则（交易金额>总资产20%禁止交易）

**成本优化规则（Cost Optimization）**：
- **CO001**：大额交易成本优化（交易金额>10万元）
- **CO002**：高置信度优化（收益>5%且置信度>80%）

**其他规则**：
- **TW001**：交易时间限制（9:00-15:59允许交易）
- **MS001**：高风险市场暂停（高风险+低置信度）

#### 规则条件

支持复杂条件表达式：
- 基本比较：`>`, `<`, `>=`, `<=`, `==`, `!=`
- 逻辑运算：`AND`, `OR`
- 括号嵌套：`(condition1 AND condition2) OR condition3`
- 数学表达式：`trade_amount > (total_assets * 0.2)`

### 4. 决策引擎

采用7阶段决策流程，确保决策的科学性和全面性：

1. **预检查阶段**：验证价格合理性、预测有效性
2. **综合风险评估**：计算价格波动、置信度、持仓集中度、市场状态、流动性、账户风险
3. **动态规则参数调整**：根据风险等级动态调整持仓上限、止损止盈比例
4. **规则依赖链评估**：分组执行规则，先风控后优化
5. **交易策略选择**：根据风险收益比选择激进/稳健/保守策略
6. **动态资金计算**：综合考虑风险、置信度、预期收益计算交易金额
7. **最终决策生成**：整合所有因素，生成买入/卖出/持有建议

### 5. 账户初始化

系统启动时自动：
1. 从真实股票接口获取5只股票的实时价格
2. 创建模拟持仓（5000-10000股）
3. 计算成本价（当前价格的85%-105%），产生盈亏差异
4. 计算账户状态（总资产、可用资金、持仓价值）

这样便于测试止损、止盈等规则。

## 配置说明

### 系统配置（`config/settings.py`）

```python
# Flask配置
HOST = '0.0.0.0'  # 允许外部访问
PORT = 5000
DEBUG = True

# 默认账户配置
DEFAULT_ACCOUNT = {
    'total_assets': 1000000,  # 总资产100万
    'available_cash': 300000,  # 可用资金30万
    'positions': {}  # 持仓（自动初始化）
}

# 预测参数
LOOKBACK_DAYS = 60  # 使用过去60天数据
PREDICTION_DAYS = 5  # 预测未来5天
```

### 规则配置（`config/rules.json`）

规则文件使用JSON格式，包含规则ID、类型、条件、动作、优先级等：

```json
{
  "rules": [
    {
      "rule_id": "RC001",
      "type": "风控规则",
      "name": "单只股票持仓限制",
      "condition": "position_ratio > 0.3",
      "action": {
        "type": "禁止",
        "message": "单只股票持仓不能超过总资产的30%"
      },
      "priority": 1
    }
  ]
}
```

### Ollama配置

AI分析功能需要Ollama服务：
- **默认地址**：`http://localhost:11434`
- **模型名称**：`qwen2.5:32b`
- **超时时间**：30秒（技术分析），15秒（意图识别）

如需修改，编辑 `app.py` 中的相关函数。

## 使用指南

### Web界面使用

1. **启动系统**：运行 `python main.py`
2. **访问界面**：浏览器打开 http://localhost:5000
3. **输入查询**：在输入框中输入自然语言查询，例如：
   - "帮我预测一个股票的走势，股票代码是 000560"
   - "000560这只股票可以买入吗？"
   - "请帮忙分析常山北明的股票趋势"
   - "分析一下600519的前景"

### 查询结果展示

系统会在两个Tab页中展示结果：

**Tab 1：量化分析**
- 技术指标图表（K线、MACD、RSI、布林带、KDJ、成交量）
- 每个指标的AI分析
- 决策建议和理由

**Tab 2：股价预测**
- 7个模型的训练进度条（动画展示）
- 模型训练完成后，显示7个模型的预测结果卡片
- 每个模型的预测图表和指标（预期收益率、置信度、趋势、风险等级）

### 命令行使用

**查看账户信息**：
```bash
curl http://localhost:5000/api/account
```

**运行场景演示**：
```bash
python testcase/demo_scenarios.py
```

**测试规则**：
```bash
python testcase/test_rules.py
```

**测试价格获取**：
```bash
python testcase/test_price.py
```

## API接口

### 核心接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/query` | POST | 处理用户查询，返回分析结果 |
| `/api/multi-model-predict` | POST | 启动多模型预测和训练 |
| `/api/training-progress/<progress_id>` | GET | 获取训练进度 |
| `/api/ai-analysis` | POST | 技术指标AI分析 |
| `/api/stock-intent-recognition` | POST | 股票意图识别 |
| `/api/account` | GET | 获取账户信息 |
| `/api/scenarios` | GET | 获取所有预设场景 |
| `/api/scenarios/<scenario_id>` | POST | 运行单个场景 |

### 请求示例

**查询股票**：
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我预测一个股票的走势，股票代码是 000560"}'
```

**多模型预测**：
```bash
curl -X POST http://localhost:5000/api/multi-model-predict \
  -H "Content-Type: application/json" \
  -d '{"stock_code": "000560"}'
```

## 常见问题

### 1. 端口被占用

**错误信息**：`Address already in use`

**解决方案**：
1. 修改 `config/settings.py` 中的 `PORT = 5000` 为其他端口（如 5001）
2. 或者关闭占用5000端口的程序

### 2. 模块导入错误

**错误信息**：`ModuleNotFoundError: No module named 'xxx'`

**解决方案**：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. akshare获取数据失败

**现象**：股票数据获取失败，使用模拟数据

**说明**：这是正常的，系统会自动使用模拟数据。如果需要真实数据：
1. 确保网络连接正常
2. 检查akshare版本是否最新：`pip install --upgrade akshare`
3. akshare可能有限流，稍后重试

### 4. XGBoost/LightGBM安装失败

**SSL连接错误**：
```bash
pip install xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lightgbm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**使用conda安装**：
```bash
conda install -c conda-forge xgboost lightgbm
```

### 5. Ollama未启动

**现象**：AI分析功能不可用

**解决方案**：
1. 确保Ollama已安装并运行
2. 检查模型是否已下载：`ollama list`
3. 如果没有qwen2.5:32b，运行：`ollama pull qwen2.5:32b`

### 6. NumPy版本冲突

**错误信息**：NumPy 2.0与某些依赖不兼容

**解决方案**：
```bash
pip install "numpy<2.0.0" --upgrade
pip install "shap<0.44" --upgrade
```

### 7. TA-Lib安装失败

**Windows用户**：
1. 访问 https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. 下载对应Python版本的whl文件
3. 使用pip安装：`pip install TA_Lib‑0.4.28‑cp39‑cp39‑win_amd64.whl`

**或使用conda**：
```bash
conda install -c conda-forge ta-lib
```

### 8. 进度条不显示

**现象**：模型训练时看不到进度条

**解决方案**：
1. 刷新页面
2. 检查浏览器控制台是否有错误
3. 确保JavaScript已加载

## 项目结构

```
stock-predict/
├── main.py                      # 项目主入口（推荐使用）
├── app.py                       # Flask应用定义
├── requirements.txt             # Python依赖包列表
├── requirements-optional.txt    # 可选依赖列表
├── script/                       # 启动脚本目录
│   ├── start.bat                # Windows启动脚本
│   └── start.sh                 # Linux/Mac启动脚本
├── testcase/                     # 测试脚本目录
│   ├── test_price.py            # 价格获取测试
│   └── test_rules.py            # 规则测试脚本
│
├── config/                      # 配置目录
│   ├── __init__.py
│   ├── settings.py             # 系统配置（账户、预测参数等）
│   └── rules.json              # 业务规则定义文件
│
├── src/                         # 源代码目录
│   ├── __init__.py
│   │
│   ├── agent/                  # 智能体模块
│   │   ├── __init__.py
│   │   └── stock_agent.py     # 股票交易智能体主类
│   │
│   ├── data/                   # 数据获取模块
│   │   ├── __init__.py
│   │   ├── stock_data.py      # 股票数据提供者类
│   │   ├── technical_indicators.py  # 技术指标计算
│   │   ├── account_initializer.py   # 账户初始化
│   │   └── database.py        # 数据持久化（可选）
│   │
│   ├── prediction/             # 股票预测模块
│   │   ├── __init__.py
│   │   ├── stock_predictor.py # 简单预测器
│   │   ├── multi_model_predictor.py  # 多模型预测器
│   │   ├── lstm_predictor.py  # LSTM预测器（可选）
│   │   └── enhanced_predictor.py     # 增强预测器
│   │
│   ├── rule_engine/            # 规则引擎模块
│   │   ├── __init__.py
│   │   ├── rule.py            # 业务规则类定义
│   │   ├── rule_engine.py     # 基础规则引擎
│   │   └── advanced_rule_engine.py   # 高级规则引擎
│   │
│   ├── decision/               # 决策引擎模块
│   │   ├── __init__.py
│   │   ├── decision_engine.py # 基础决策引擎
│   │   └── advanced_decision_engine.py  # 高级决策引擎
│   │
│   ├── explainability/         # 可解释性模块
│   │   ├── __init__.py
│   │   └── model_explainer.py # 模型解释器
│   │
│   ├── scenarios/              # 场景演示
│   │   ├── __init__.py
│   │   └── rule_scenarios.py  # 规则场景定义
│   │
│   └── visualization/          # 可视化模块
│       ├── __init__.py
│       └── rule_visualizer.py # 规则可视化器
│
├── templates/                   # HTML模板目录
│   └── index.html              # 主页面（支持文本问答的聊天界面）
│
├── script/                     # 启动脚本目录
│   ├── start.bat              # Windows启动脚本
│   └── start.sh               # Linux/Mac启动脚本
└── testcase/                   # 测试脚本目录
    ├── test_price.py          # 价格获取测试
    ├── test_rules.py          # 规则测试脚本
    └── demo_scenarios.py      # 场景演示脚本
```

## 开发指南

### 添加新规则

1. 编辑 `config/rules.json`
2. 添加规则定义：
```json
{
  "rule_id": "RC007",
  "type": "风控规则",
  "name": "自定义规则",
  "condition": "field > value",
  "action": {
    "type": "禁止",
    "message": "规则说明"
  },
  "priority": 5
}
```
3. 重启服务

### 添加新预测模型

1. 在 `src/prediction/multi_model_predictor.py` 中添加模型初始化代码
2. 在 `_init_models()` 方法中注册模型
3. 确保模型实现 `fit()` 和 `predict()` 方法

### 自定义决策策略

编辑 `src/decision/advanced_decision_engine.py` 中的策略选择逻辑。

## 版本兼容性

### NumPy版本

- **必须使用**：NumPy < 2.0.0
- **原因**：numba、shap等依赖还不支持NumPy 2.0
- **安装**：`pip install "numpy<2.0.0"`

### SHAP版本

- **必须使用**：SHAP < 0.44
- **原因**：0.44+版本需要NumPy 2.0+，但与numba不兼容
- **安装**：`pip install "shap<0.44"`

### Python版本

- **推荐**：Python 3.8 - 3.11
- **已测试**：Python 3.9, 3.10, 3.11

## 性能优化建议

1. **首次运行**：可能需要下载一些依赖包，稍等片刻
2. **数据获取**：akshare首次获取数据可能较慢，后续会使用缓存
3. **模型训练**：多模型训练可能需要1-3分钟，LSTM训练时间更长
4. **Ollama分析**：AI分析可能需要10-30秒，取决于硬件性能

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

本项目仅供学习和研究使用。

## 更新日志

### 最新版本特性

- ✅ 多模型预测系统（7种算法）
- ✅ 训练进度条动画展示
- ✅ 技术指标AI分析
- ✅ LLM股票意图识别
- ✅ 复杂业务规则引擎
- ✅ 7阶段决策流程
- ✅ 账户自动初始化
- ✅ 实时价格获取优化
- ✅ 可解释性分析
- ✅ 现代化Web界面

---

**如有问题或建议，请提交Issue或联系开发者。**
