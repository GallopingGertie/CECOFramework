#!/bin/bash

# 云边端推理框架演示脚本

echo "========================================"
echo "云边端推理框架演示"
echo "========================================"

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    exit 1
fi

# 检查依赖是否安装
echo "检查依赖..."
if ! python3 -c "import aiohttp" 2>/dev/null; then
    echo "安装依赖..."
    pip3 install -r requirements.txt
fi

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}依赖检查完成${NC}"

# 函数: 检查服务是否运行
check_service() {
    local endpoint=$1
    local name=$2
    
    if curl -s "$endpoint/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name 正在运行${NC}"
        return 0
    else
        echo -e "${RED}❌ $name 未运行${NC}"
        return 1
    fi
}

# 函数: 启动服务
start_service() {
    local script=$1
    local name=$2
    local log_file=$3
    
    echo -e "${YELLOW}启动 $name...${NC}"
    python3 "$script" --config config/config.yaml > "$log_file" 2>&1 &
    local pid=$!
    
    # 等待服务启动
    sleep 2
    
    # 检查是否成功启动
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name 启动成功 (PID: $pid)${NC}"
        echo $pid
    else
        echo -e "${RED}❌ $name 启动失败${NC}"
        echo "日志:"
        cat "$log_file"
        exit 1
    fi
}

# 检查服务是否已在运行
echo "检查服务状态..."
EDGE_RUNNING=0
CLOUD_RUNNING=0

if check_service "http://localhost:8080" "边端服务"; then
    EDGE_RUNNING=1
fi

if check_service "http://localhost:8081" "云端服务"; then
    CLOUD_RUNNING=1
fi

# 如果服务未运行，启动它们
if [ $EDGE_RUNNING -eq 0 ] && [ $CLOUD_RUNNING -eq 0 ]; then
    echo -e "${YELLOW}启动所有服务...${NC}"
    
    # 创建日志目录
    mkdir -p logs
    
    # 启动边端服务
    EDGE_PID=$(start_service "start_edge.py" "边端服务" "logs/edge.log")
    
    # 启动云端服务
    CLOUD_PID=$(start_service "start_cloud.py" "云端服务" "logs/cloud.log")
    
    echo -e "${GREEN}所有服务已启动${NC}"
    echo ""
    echo "日志文件:"
    echo "  边端: logs/edge.log"
    echo "  云端: logs/cloud.log"
    
    # 设置退出时清理
    trap "kill $EDGE_PID $CLOUD_PID 2>/dev/null; echo -e '${YELLOW}服务已停止${NC}'" EXIT
    
    # 等待服务完全启动
    echo "等待服务就绪..."
    sleep 3
    
elif [ $EDGE_RUNNING -eq 0 ]; then
    echo -e "${YELLOW}启动边端服务...${NC}"
    EDGE_PID=$(start_service "start_edge.py" "边端服务" "logs/edge.log")
    trap "kill $EDGE_PID 2>/dev/null" EXIT
    
elif [ $CLOUD_RUNNING -eq 0 ]; then
    echo -e "${YELLOW}启动云端服务...${NC}"
    CLOUD_PID=$(start_service "start_cloud.py" "云端服务" "logs/cloud.log")
    trap "kill $CLOUD_PID 2>/dev/null" EXIT
fi

# 再次检查服务
echo ""
echo "验证服务状态..."
check_service "http://localhost:8080" "边端服务"
check_service "http://localhost:8081" "云端服务"

echo ""
echo "========================================"
echo "运行示例..."
echo "========================================"

# 运行示例脚本
python3 examples/basic_usage.py

echo ""
echo "========================================"
echo "演示完成!"
echo "========================================"

# 如果我们是启动的服务，询问是否停止
if [ ${EDGE_PID:-0} -ne 0 ] || [ ${CLOUD_PID:-0} -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}按任意键停止服务...${NC}"
    read -n 1 -s
fi
