"""
novelWriter – Local Tools Base Classes and Schema System
==========================================================

File History:
Created: 2025-09-25 [James - Dev Agent]

This file is a part of novelWriter
Copyright (C) 2025 Veronica Berglyd Olsen and novelWriter Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from enum import Enum
from functools import wraps

from pydantic import BaseModel, Field, ConfigDict

from novelwriter.api.exceptions import MCPExecutionError, APIPermissionError

logger = logging.getLogger(__name__)

# Type variable for tool result
T = TypeVar('T')


class ToolPermission(str, Enum):
    """工具权限枚举"""
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    ADMIN = "admin"


class ToolExecutionResult(BaseModel, Generic[T]):
    """工具执行结果标准格式"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    call_id: str = Field(description="唯一调用标识符")
    success: bool = Field(description="执行是否成功")
    result: Optional[T] = Field(default=None, description="执行结果数据")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    execution_time_ms: int = Field(description="执行时间（毫秒）")
    
    @classmethod
    def success_result(cls, call_id: str, result: T, execution_time_ms: int) -> "ToolExecutionResult[T]":
        """创建成功结果"""
        return cls(
            call_id=call_id,
            success=True,
            result=result,
            error_message=None,
            execution_time_ms=execution_time_ms
        )
    
    @classmethod
    def error_result(cls, call_id: str, error_message: str, execution_time_ms: int) -> "ToolExecutionResult[T]":
        """创建错误结果"""
        return cls(
            call_id=call_id,
            success=False,
            result=None,
            error_message=error_message,
            execution_time_ms=execution_time_ms
        )


class ToolMetadata(BaseModel):
    """工具元数据"""
    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述")
    version: str = Field(default="1.0.0", description="工具版本")
    author: str = Field(default="novelWriter", description="工具作者")
    parameters_schema: Dict[str, Any] = Field(description="参数Schema定义")
    required_permissions: List[ToolPermission] = Field(default_factory=list, description="所需权限列表")
    tags: List[str] = Field(default_factory=list, description="工具标签")


class BaseTool(ABC):
    """统一工具基类"""
    
    def __init__(self, api_instance: Any):
        """
        初始化工具
        
        Args:
            api_instance: NovelWriterAPI实例
        """
        self._api = api_instance
        self._metadata = self._build_metadata()
        self._call_counter = 0
        
    @abstractmethod
    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据"""
        pass
    
    @abstractmethod
    async def _execute_impl(self, **parameters) -> Any:
        """
        工具执行的具体实现
        
        Args:
            **parameters: 工具参数
            
        Returns:
            执行结果
        """
        pass
    
    @property
    def name(self) -> str:
        """获取工具名称"""
        return self._metadata.name
    
    @property
    def description(self) -> str:
        """获取工具描述"""
        return self._metadata.description
    
    @property
    def metadata(self) -> ToolMetadata:
        """获取完整元数据"""
        return self._metadata
    
    def _generate_call_id(self) -> str:
        """生成唯一调用ID"""
        self._call_counter += 1
        return f"{self.name}_{int(time.time())}_{self._call_counter}"
    
    def _check_permissions(self, required_permissions: List[ToolPermission]) -> bool:
        """
        检查权限
        
        Args:
            required_permissions: 所需权限列表
            
        Returns:
            是否有权限
        """
        # TODO: 实际权限检查逻辑，当前默认返回True
        return True
    
    async def execute(self, **parameters) -> ToolExecutionResult:
        """
        执行工具
        
        Args:
            **parameters: 工具参数
            
        Returns:
            ToolExecutionResult: 执行结果
        """
        call_id = self._generate_call_id()
        start_time = time.perf_counter()
        
        try:
            # 权限检查
            if not self._check_permissions(self._metadata.required_permissions):
                raise APIPermissionError(f"Insufficient permissions for tool: {self.name}")
            
            # 执行工具
            logger.debug(f"Executing tool {self.name} with call_id: {call_id}")
            result = await self._execute_impl(**parameters)
            
            # 计算执行时间
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            
            # 返回成功结果
            return ToolExecutionResult.success_result(
                call_id=call_id,
                result=result,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            # 计算执行时间
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            
            # 记录错误
            logger.error(f"Tool {self.name} execution failed: {str(e)}", exc_info=True)
            
            # 返回错误结果
            return ToolExecutionResult.error_result(
                call_id=call_id,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )


def monitor_performance(func):
    """性能监控装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # 记录性能数据
            if execution_time > 10:
                logger.warning(
                    f"Tool execution exceeded 10ms threshold: {func.__name__} took {execution_time:.2f}ms"
                )
            else:
                logger.debug(f"Tool {func.__name__} executed in {execution_time:.2f}ms")
                
            return result
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Tool {func.__name__} failed after {execution_time:.2f}ms: {str(e)}")
            raise
    
    return wrapper


def requires_permission(*permissions: ToolPermission):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # 检查权限
            if not self._check_permissions(list(permissions)):
                raise APIPermissionError(
                    f"Insufficient permissions. Required: {', '.join(p.value for p in permissions)}"
                )
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator


# Schema定义辅助类
class ParameterSchema(BaseModel):
    """参数Schema基类"""
    model_config = ConfigDict(extra='forbid')


class ProjectInfoParams(ParameterSchema):
    """项目信息工具参数"""
    include_settings: bool = Field(default=True, description="是否包含项目设置")
    include_stats: bool = Field(default=True, description="是否包含统计信息")


class DocumentListParams(ParameterSchema):
    """文档列表工具参数"""
    scope: str = Field(default="all", description="文档范围: all, novel, notes, trash")
    include_content: bool = Field(default=False, description="是否包含文档内容预览")


class DocumentReadParams(ParameterSchema):
    """文档读取工具参数"""
    handle: str = Field(description="文档句柄")
    include_metadata: bool = Field(default=True, description="是否包含元数据")


class DocumentWriteParams(ParameterSchema):
    """文档写入工具参数"""
    handle: str = Field(description="文档句柄")
    content: str = Field(description="文档内容")
    create_backup: bool = Field(default=True, description="是否创建备份")


class CreateDocumentParams(ParameterSchema):
    """创建文档工具参数"""
    title: str = Field(description="文档标题")
    parent_handle: Optional[str] = Field(default=None, description="父节点句柄")
    doc_type: str = Field(default="DOCUMENT", description="文档类型")
    content: Optional[str] = Field(default=None, description="初始内容")


class ProjectTreeParams(ParameterSchema):
    """项目树工具参数"""
    include_stats: bool = Field(default=True, description="是否包含统计信息")
    filter_type: Optional[str] = Field(default=None, description="过滤类型")
    max_depth: Optional[int] = Field(default=None, description="最大深度")


class GlobalSearchParams(ParameterSchema):
    """全局搜索工具参数"""
    query: str = Field(description="搜索查询")
    search_type: str = Field(default="content", description="搜索类型: content, title, tag")
    case_sensitive: bool = Field(default=False, description="是否区分大小写")
    whole_word: bool = Field(default=False, description="是否全词匹配")
    max_results: int = Field(default=100, description="最大结果数")


class TagListParams(ParameterSchema):
    """标签列表工具参数"""
    include_counts: bool = Field(default=True, description="是否包含使用计数")
    sort_by: str = Field(default="name", description="排序方式: name, count")
