#!/usr/bin/env python3
# File: sglang_server_wrapper.py

import os
import signal
import subprocess
import sys
import atexit

# 全局变量，用于信号处理
server_process = None

def start_sglang_server(model_path: str, log_file: str):
    global server_process

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "0.0.0.0",
        "--port", "30000",
        "--tensor-parallel-size", "4",
        "--log-level", "warning",
        "--context-length", "32768",
        "--max-prefill-tokens", "32768"
    ]

    print(f"🚀 Starting SGLang server with model: {model_path}")
    print(f"📝 Logging to: {log_file}")

    # 打开日志文件
    log_fp = open(log_file, "w")

    # 启动子进程，并创建新的进程组（便于后续统一 kill）
    server_process = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid  # 关键：创建新 session，使子进程独立成组
    )

    # 注册退出时的清理函数
    atexit.register(stop_sglang_server)

    print(f"✅ SGLang server started (PID: {server_process.pid})")


def stop_sglang_server():
    global server_process
    if server_process is None:
        return

    print("🛑 Shutting down SGLang server gracefully...")

    # 发送 SIGTERM 并无限等待进程退出
    pgid = os.getpgid(server_process.pid)
    os.killpg(pgid, signal.SIGTERM)
    print("⏳ Waiting for SGLang server to shut down gracefully (this may take a while)...")
    server_process.wait()  # ← 移除 timeout，永久阻塞直到退出
    print("✅ SGLang server stopped cleanly.")

def signal_handler(signum, frame):
    print(f"\n\nReceived signal {signum}. Triggering graceful shutdown...")
    stop_sglang_server()
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sglang_server_wrapper.py <model_path> <log_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    log_file = sys.argv[2]

    # 注册信号处理器（支持 Ctrl+C 和 kill -TERM）
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动服务
    start_sglang_server(model_path, log_file)

    # 保持主进程运行，等待信号
    try:
        print("💡 Server is running. Press Ctrl+C to stop gracefully.\n")
        server_process.wait()  # 阻塞直到子进程退出（比如崩溃）
    except KeyboardInterrupt:
        pass  # 已由 signal_handler 处理
    finally:
        stop_sglang_server()