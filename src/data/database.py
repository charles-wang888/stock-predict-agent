"""
数据持久化模块
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class Database:
    """数据库管理类"""
    
    def __init__(self, db_path: str = "data/stock_agent.db"):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 交易记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                action TEXT NOT NULL,
                shares INTEGER,
                price REAL,
                amount REAL,
                timestamp TEXT NOT NULL,
                prediction_data TEXT,
                decision_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 预测记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                actual_price REAL,
                predicted_price REAL,
                predicted_return REAL,
                confidence REAL,
                accuracy REAL,
                model_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 规则执行历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rule_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT NOT NULL,
                rule_name TEXT,
                triggered BOOLEAN,
                context_data TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 账户状态表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_assets REAL,
                available_cash REAL,
                positions TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, stock_code: str, action: str, shares: int, 
                   price: float, amount: float, prediction_data: Dict = None,
                   decision_data: Dict = None):
        """保存交易记录"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (stock_code, action, shares, price, amount, 
                              timestamp, prediction_data, decision_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stock_code,
            action,
            shares,
            price,
            amount,
            datetime.now().isoformat(),
            json.dumps(prediction_data) if prediction_data else None,
            json.dumps(decision_data) if decision_data else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_trades(self, stock_code: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取交易记录"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if stock_code:
            cursor.execute('''
                SELECT * FROM trades 
                WHERE stock_code = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (stock_code, limit))
        else:
            cursor.execute('''
                SELECT * FROM trades 
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def save_prediction(self, stock_code: str, prediction_date: str,
                       actual_price: float, predicted_price: float,
                       predicted_return: float, confidence: float,
                       accuracy: float = None, model_type: str = None):
        """保存预测记录"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (stock_code, prediction_date, actual_price,
                                   predicted_price, predicted_return, confidence,
                                   accuracy, model_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stock_code,
            prediction_date,
            actual_price,
            predicted_price,
            predicted_return,
            confidence,
            accuracy,
            model_type
        ))
        
        conn.commit()
        conn.close()
    
    def get_predictions(self, stock_code: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取预测记录"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if stock_code:
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE stock_code = ?
                ORDER BY prediction_date DESC
                LIMIT ?
            ''', (stock_code, limit))
        else:
            cursor.execute('''
                SELECT * FROM predictions 
                ORDER BY prediction_date DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def save_rule_execution(self, rule_id: str, rule_name: str, triggered: bool,
                           context_data: Dict = None):
        """保存规则执行记录"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO rule_executions (rule_id, rule_name, triggered, 
                                       context_data, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            rule_id,
            rule_name,
            triggered,
            json.dumps(context_data) if context_data else None,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def save_account_snapshot(self, total_assets: float, available_cash: float,
                             positions: Dict = None):
        """保存账户快照"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO account_snapshots (total_assets, available_cash, 
                                         positions, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (
            total_assets,
            available_cash,
            json.dumps(positions) if positions else None,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_trade_statistics(self, stock_code: str = None) -> Dict[str, Any]:
        """获取交易统计"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if stock_code:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN action = '买入' THEN amount ELSE 0 END) as total_buy,
                    SUM(CASE WHEN action = '卖出' THEN amount ELSE 0 END) as total_sell,
                    AVG(price) as avg_price
                FROM trades
                WHERE stock_code = ?
            ''', (stock_code,))
        else:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN action = '买入' THEN amount ELSE 0 END) as total_buy,
                    SUM(CASE WHEN action = '卖出' THEN amount ELSE 0 END) as total_sell,
                    AVG(price) as avg_price
                FROM trades
            ''')
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            'total_trades': row[0] or 0,
            'total_buy': row[1] or 0,
            'total_sell': row[2] or 0,
            'avg_price': row[3] or 0
        }


