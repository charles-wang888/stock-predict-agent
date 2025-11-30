"""
项目主入口文件
"""
from app import app
from config.settings import HOST, PORT, DEBUG

if __name__ == '__main__':
    print("=" * 50)
    print("  股票交易智能体系统")
    print("=" * 50)
    print(f"  服务地址: http://localhost:{PORT}")
    print(f"  调试模式: {'开启' if DEBUG else '关闭'}")
    print("=" * 50)
    print("\n正在启动服务...")
    print("按 Ctrl+C 停止服务\n")
    
    app.run(host=HOST, port=PORT, debug=DEBUG)


