"""novelWriter – Project and Document Tools Implementation.
========================================================

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
from typing import Any

from novelwriter.api.tools.base import (
    BaseTool, ToolMetadata, ToolPermission, monitor_performance, requires_permission,
    ProjectInfoParams, ProjectTreeParams
)

logger = logging.getLogger(__name__)


class ProjectInfoTool(BaseTool):
    """项目信息获取工具."""

    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据."""
        return ToolMetadata(
            name="project_info",
            description="获取项目基本信息、设置和统计数据",
            version="1.0.0",
            parameters_schema={
                "type": "object",
                "properties": {
                    "include_settings": {
                        "type": "boolean",
                        "description": "是否包含项目设置",
                        "default": True
                    },
                    "include_stats": {
                        "type": "boolean",
                        "description": "是否包含统计信息",
                        "default": True
                    }
                },
                "required": []
            },
            required_permissions=[ToolPermission.READ],
            tags=["project", "metadata", "info"]
        )

    @monitor_performance
    @requires_permission(ToolPermission.READ)
    async def _execute_impl(self, **parameters) -> dict[str, Any]:
        """执行项目信息获取.

        Args:
            **parameters: 工具参数

        Returns:
            项目信息字典

        """
        # 验证参数
        params = ProjectInfoParams(**parameters)

        # 获取项目元数据
        project_meta = self._api.getProjectMeta()

        result = {
            "title": project_meta.get("title", "Untitled"),
            "author": project_meta.get("author", "Unknown"),
            "created": project_meta.get("created"),
            "updated": project_meta.get("updated"),
            "word_count": project_meta.get("wordCount", 0),
            "chapter_count": project_meta.get("chapterCount", 0),
            "scene_count": project_meta.get("sceneCount", 0)
        }

        # 包含设置信息
        if params.include_settings:
            result["settings"] = {
                "language": project_meta.get("language", "en"),
                "spell_check": project_meta.get("spellCheck", False),
                "auto_save": project_meta.get("autoSave", True),
                "backup_on_close": project_meta.get("backupOnClose", True)
            }

        # 包含统计信息
        if params.include_stats:
            stats = self._api.getProjectStats()
            result["statistics"] = {
                "total_words": stats.get("totalWords", 0),
                "total_characters": stats.get("totalChars", 0),
                "total_paragraphs": stats.get("totalParagraphs", 0),
                "average_chapter_length": stats.get("avgChapterLength", 0),
                "average_scene_length": stats.get("avgSceneLength", 0),
                "document_count": stats.get("documentCount", 0),
                "note_count": stats.get("noteCount", 0)
            }

        return result


class ProjectTreeTool(BaseTool):
    """项目树结构获取工具."""

    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据."""
        return ToolMetadata(
            name="project_tree",
            description="获取完整项目树结构，包含文档层级关系",
            version="1.0.0",
            parameters_schema={
                "type": "object",
                "properties": {
                    "include_stats": {
                        "type": "boolean",
                        "description": "是否包含统计信息",
                        "default": True
                    },
                    "filter_type": {
                        "type": "string",
                        "description": "过滤文档类型",
                        "enum": ["all", "novel", "notes", "trash"],
                        "default": "all"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "最大遍历深度",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": []
            },
            required_permissions=[ToolPermission.READ],
            tags=["project", "tree", "structure"]
        )

    @monitor_performance
    @requires_permission(ToolPermission.READ)
    async def _execute_impl(self, **parameters) -> dict[str, Any]:
        """执行项目树获取.

        Args:
            **parameters: 工具参数

        Returns:
            项目树结构

        """
        # 验证参数
        params = ProjectTreeParams(**parameters)

        # 获取项目树
        tree_data = self._api.getProjectTree()

        # 处理树结构
        def process_node(node: dict[str, Any], depth: int = 0) -> dict[str, Any] | None:
            """递归处理树节点."""
            # 检查深度限制
            if params.max_depth and depth >= params.max_depth:
                return None

            # 检查类型过滤
            node_type = node.get("type", "").lower()
            if params.filter_type != "all":
                if params.filter_type == "novel" and node_type not in ["root", "folder", "document"]:
                    return None
                elif params.filter_type == "notes" and node_type not in ["root", "folder", "note"]:
                    return None
                elif params.filter_type == "trash" and node_type != "trash":
                    return None

            # 构建节点信息
            result_node = {
                "handle": node.get("handle"),
                "title": node.get("title", "Untitled"),
                "type": node_type,
                "level": depth,
                "status": node.get("status"),
                "expanded": node.get("expanded", False)
            }

            # 添加统计信息
            if params.include_stats:
                result_node["stats"] = {
                    "word_count": node.get("wordCount", 0),
                    "char_count": node.get("charCount", 0),
                    "para_count": node.get("paraCount", 0)
                }

            # 处理子节点
            children = node.get("children", [])
            if children:
                processed_children = []
                for child in children:
                    processed_child = process_node(child, depth + 1)
                    if processed_child:
                        processed_children.append(processed_child)

                if processed_children:
                    result_node["children"] = processed_children

            return result_node

        # 处理根节点
        processed_tree = process_node(tree_data)

        return {
            "tree": processed_tree,
            "total_nodes": self._count_nodes(processed_tree) if processed_tree else 0,
            "max_depth": self._get_max_depth(processed_tree) if processed_tree else 0
        }

    def _count_nodes(self, node: dict[str, Any]) -> int:
        """统计节点数量."""
        count = 1
        children = node.get("children", [])
        for child in children:
            count += self._count_nodes(child)
        return count

    def _get_max_depth(self, node: dict[str, Any], current_depth: int = 0) -> int:
        """获取最大深度."""
        children = node.get("children", [])
        if not children:
            return current_depth

        max_child_depth = current_depth
        for child in children:
            child_depth = self._get_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth
