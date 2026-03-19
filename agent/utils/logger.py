import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional

class MCPLogger:
    def __init__(self):
        # 创建logger
        self.logger = logging.getLogger('mcp')
        self.logger.setLevel(logging.DEBUG)

        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # 创建文件处理器
        file_handler = logging.FileHandler('mcp.log')
        file_handler.setLevel(logging.DEBUG)

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 设置处理器的格式化器
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 添加处理器到logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _format_dict(self, data: Dict[str, Any]) -> str:
        """格式化字典数据为JSON字符串"""
        return json.dumps(data, ensure_ascii=False, indent=2)

    def log_server_selection(self, enabled_servers: Dict[str, Any], 
                           default_config: Dict[str, Any]) -> None:
        """记录服务器选择信息"""
        self.logger.info("MCP Server Selection:")
        self.logger.info(f"Enabled servers: {self._format_dict(enabled_servers)}")
        self.logger.info(f"Default config: {self._format_dict(default_config)}")

    def log_request(self, server_name: str, endpoint: str, 
                   method: str, headers: Dict[str, str], 
                   payload: Any) -> None:
        """记录请求信息"""
        request_info = {
            'timestamp': datetime.now().isoformat(),
            'server_name': server_name,
            'endpoint': endpoint,
            'method': method,
            'headers': headers,
            'payload': payload
        }
        self.logger.info(f"MCP Request: {self._format_dict(request_info)}")

    def log_response(self, server_name: str, status_code: int, 
                    headers: Dict[str, str], payload: Any, 
                    response_time: float) -> None:
        """记录响应信息"""
        response_info = {
            'timestamp': datetime.now().isoformat(),
            'server_name': server_name,
            'status_code': status_code,
            'headers': headers,
            'payload': payload,
            'response_time_ms': response_time
        }
        self.logger.info(f"MCP Response: {self._format_dict(response_info)}")

    def log_error(self, server_name: str, error_type: str, 
                 error_message: str, stack_trace: Optional[str] = None) -> None:
        """记录错误信息"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'server_name': server_name,
            'error_type': error_type,
            'error_message': error_message,
            'stack_trace': stack_trace
        }
        self.logger.error(f"MCP Error: {self._format_dict(error_info)}")

    def log_performance(self, server_name: str, operation: str, 
                       duration: float, success: bool) -> None:
        """记录性能信息"""
        performance_info = {
            'timestamp': datetime.now().isoformat(),
            'server_name': server_name,
            'operation': operation,
            'duration_ms': duration,
            'success': success
        }
        self.logger.info(f"MCP Performance: {self._format_dict(performance_info)}")

# 创建全局logger实例
mcp_logger = MCPLogger() 