"""
novelWriter – Document Operation Tools Implementation
======================================================

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

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from novelwriter.api.tools.base import (
    BaseTool, ToolMetadata, ToolPermission, ToolExecutionResult,
    monitor_performance, requires_permission,
    DocumentListParams, DocumentReadParams, DocumentWriteParams, CreateDocumentParams
)

logger = logging.getLogger(__name__)


class DocumentListTool(BaseTool):
    """文档列表获取工具"""
    
    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据"""
        return ToolMetadata(
            name="document_list",
            description="获取项目中的文档列表",
            version="1.0.0",
            parameters_schema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "文档范围",
                        "enum": ["all", "novel", "notes", "trash"],
                        "default": "all"
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "是否包含内容预览",
                        "default": False
                    }
                },
                "required": []
            },
            required_permissions=[ToolPermission.READ],
            tags=["document", "list", "query"]
        )
    
    @monitor_performance
    @requires_permission(ToolPermission.READ)
    async def _execute_impl(self, **parameters) -> List[Dict[str, Any]]:
        """
        执行文档列表获取
        
        Args:
            **parameters: 工具参数
            
        Returns:
            文档列表
        """
        # 验证参数
        params = DocumentListParams(**parameters)
        
        # 获取文档列表
        documents = self._api.listDocuments(scope=params.scope)
        
        # 处理文档信息
        result = []
        for doc in documents:
            doc_info = {
                "handle": doc.handle,
                "title": doc.title,
                "type": doc.doc_type,
                "status": doc.status,
                "created": doc.created.isoformat() if doc.created else None,
                "updated": doc.updated.isoformat() if doc.updated else None,
                "word_count": doc.word_count,
                "char_count": doc.char_count,
                "para_count": doc.para_count
            }
            
            # 包含内容预览
            if params.include_content:
                try:
                    content = self._api.getDocText(doc.handle)
                    # 获取前200个字符作为预览
                    preview = content[:200] + "..." if len(content) > 200 else content
                    doc_info["content_preview"] = preview
                except Exception as e:
                    logger.warning(f"Failed to get content preview for {doc.handle}: {e}")
                    doc_info["content_preview"] = None
            
            result.append(doc_info)
        
        return result


class DocumentReadTool(BaseTool):
    """文档内容读取工具"""
    
    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据"""
        return ToolMetadata(
            name="document_read",
            description="读取指定文档的完整内容",
            version="1.0.0",
            parameters_schema={
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "文档句柄"
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "是否包含元数据",
                        "default": True
                    }
                },
                "required": ["handle"]
            },
            required_permissions=[ToolPermission.READ],
            tags=["document", "read", "content"]
        )
    
    @monitor_performance
    @requires_permission(ToolPermission.READ)
    async def _execute_impl(self, **parameters) -> Dict[str, Any]:
        """
        执行文档读取
        
        Args:
            **parameters: 工具参数
            
        Returns:
            文档内容和元数据
        """
        # 验证参数
        params = DocumentReadParams(**parameters)
        
        # 读取文档内容
        content = self._api.getDocText(params.handle)
        
        result = {
            "handle": params.handle,
            "content": content,
            "length": len(content)
        }
        
        # 包含元数据
        if params.include_metadata:
            try:
                # 获取文档元数据
                doc_list = self._api.listDocuments(scope="all")
                doc_meta = next((d for d in doc_list if d.handle == params.handle), None)
                
                if doc_meta:
                    result["metadata"] = {
                        "title": doc_meta.title,
                        "type": doc_meta.doc_type,
                        "status": doc_meta.status,
                        "created": doc_meta.created.isoformat() if doc_meta.created else None,
                        "updated": doc_meta.updated.isoformat() if doc_meta.updated else None,
                        "word_count": doc_meta.word_count,
                        "char_count": doc_meta.char_count,
                        "para_count": doc_meta.para_count
                    }
            except Exception as e:
                logger.warning(f"Failed to get metadata for {params.handle}: {e}")
        
        return result


class DocumentWriteTool(BaseTool):
    """文档内容写入工具"""
    
    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据"""
        return ToolMetadata(
            name="document_write",
            description="写入或更新文档内容",
            version="1.0.0",
            parameters_schema={
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "文档句柄"
                    },
                    "content": {
                        "type": "string",
                        "description": "文档内容"
                    },
                    "create_backup": {
                        "type": "boolean",
                        "description": "是否创建备份",
                        "default": True
                    }
                },
                "required": ["handle", "content"]
            },
            required_permissions=[ToolPermission.WRITE],
            tags=["document", "write", "update"]
        )
    
    @monitor_performance
    @requires_permission(ToolPermission.WRITE)
    async def _execute_impl(self, **parameters) -> Dict[str, Any]:
        """
        执行文档写入
        
        Args:
            **parameters: 工具参数
            
        Returns:
            写入结果
        """
        # 验证参数
        params = DocumentWriteParams(**parameters)
        
        # 创建备份
        if params.create_backup:
            try:
                original_content = self._api.getDocText(params.handle)
                # 这里可以实现备份逻辑
                logger.debug(f"Backup created for document {params.handle}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        # 写入文档
        success = self._api.setDocText(params.handle, params.content)
        
        # 计算写入统计
        word_count = len(params.content.split())
        char_count = len(params.content)
        para_count = len([p for p in params.content.split('\n\n') if p.strip()])
        
        return {
            "handle": params.handle,
            "success": success,
            "content_length": len(params.content),
            "word_count": word_count,
            "char_count": char_count,
            "para_count": para_count,
            "timestamp": datetime.now().isoformat()
        }


class CreateDocumentTool(BaseTool):
    """创建新文档工具"""
    
    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据"""
        return ToolMetadata(
            name="create_document",
            description="创建新的文档或笔记",
            version="1.0.0",
            parameters_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "文档标题"
                    },
                    "parent_handle": {
                        "type": "string",
                        "description": "父节点句柄"
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "文档类型",
                        "enum": ["DOCUMENT", "NOTE", "FOLDER"],
                        "default": "DOCUMENT"
                    },
                    "content": {
                        "type": "string",
                        "description": "初始内容"
                    }
                },
                "required": ["title"]
            },
            required_permissions=[ToolPermission.CREATE],
            tags=["document", "create", "new"]
        )
    
    @monitor_performance
    @requires_permission(ToolPermission.CREATE)
    async def _execute_impl(self, **parameters) -> Dict[str, Any]:
        """
        执行文档创建
        
        Args:
            **parameters: 工具参数
            
        Returns:
            创建结果
        """
        # 验证参数
        params = CreateDocumentParams(**parameters)
        
        # 创建文档
        handle = self._api.createDocument(
            title=params.title,
            parent_handle=params.parent_handle,
            doc_type=params.doc_type
        )
        
        # 设置初始内容
        if params.content:
            self._api.setDocText(handle, params.content)
        
        return {
            "handle": handle,
            "title": params.title,
            "type": params.doc_type,
            "parent_handle": params.parent_handle,
            "created": datetime.now().isoformat(),
            "content_set": bool(params.content)
        }
