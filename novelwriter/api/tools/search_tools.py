"""novelWriter – Search and Tag Tools Implementation.
==================================================

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
from collections import Counter

from novelwriter.api.tools.base import (
    BaseTool, ToolMetadata, ToolPermission, monitor_performance, requires_permission,
    GlobalSearchParams, TagListParams
)

logger = logging.getLogger(__name__)


class GlobalSearchTool(BaseTool):
    """全局搜索工具."""

    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据."""
        return ToolMetadata(
            name="global_search",
            description="在项目中进行全文搜索、标题搜索或标签搜索",
            version="1.0.0",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询字符串"
                    },
                    "search_type": {
                        "type": "string",
                        "description": "搜索类型",
                        "enum": ["content", "title", "tag"],
                        "default": "content"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "是否区分大小写",
                        "default": False
                    },
                    "whole_word": {
                        "type": "boolean",
                        "description": "是否全词匹配",
                        "default": False
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最大结果数",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 1000
                    }
                },
                "required": ["query"]
            },
            required_permissions=[ToolPermission.READ],
            tags=["search", "query", "find"]
        )

    @monitor_performance
    @requires_permission(ToolPermission.READ)
    async def _execute_impl(self, **parameters) -> dict[str, Any]:
        """执行全局搜索.

        Args:
            **parameters: 工具参数

        Returns:
            搜索结果

        """
        # 验证参数
        params = GlobalSearchParams(**parameters)

        # 执行搜索
        search_results = self._api.searchProject(
            query=params.query,
            search_type=params.search_type,
            case_sensitive=params.case_sensitive,
            whole_word=params.whole_word
        )

        # 限制结果数量
        limited_results = search_results[:params.max_results]

        # 格式化搜索结果
        formatted_results = []
        for result in limited_results:
            formatted_result = {
                "handle": result.get("handle"),
                "title": result.get("title"),
                "match_type": result.get("match_type"),
                "line_number": result.get("line_number"),
                "context": result.get("context"),
                "score": result.get("score", 1.0)
            }

            # 添加高亮信息
            if params.search_type == "content":
                formatted_result["highlighted_text"] = self._highlight_match(
                    result.get("context", ""),
                    params.query,
                    params.case_sensitive
                )

            formatted_results.append(formatted_result)

        # 统计信息
        stats = {
            "total_matches": len(search_results),
            "returned_matches": len(formatted_results),
            "truncated": len(search_results) > params.max_results,
            "search_type": params.search_type,
            "query_length": len(params.query)
        }

        return {
            "query": params.query,
            "results": formatted_results,
            "statistics": stats
        }

    def _highlight_match(self, text: str, query: str, case_sensitive: bool) -> str:
        """高亮匹配文本.

        Args:
            text: 原始文本
            query: 查询字符串
            case_sensitive: 是否区分大小写

        Returns:
            带高亮标记的文本

        """
        if not case_sensitive:
            # 不区分大小写的替换
            import re
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            return pattern.sub(f"**{query}**", text)
        else:
            # 区分大小写的替换
            return text.replace(query, f"**{query}**")


class TagListTool(BaseTool):
    """标签列表获取工具."""

    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据."""
        return ToolMetadata(
            name="tag_list",
            description="获取项目中所有标签及其使用统计",
            version="1.0.0",
            parameters_schema={
                "type": "object",
                "properties": {
                    "include_counts": {
                        "type": "boolean",
                        "description": "是否包含使用计数",
                        "default": True
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "排序方式",
                        "enum": ["name", "count"],
                        "default": "name"
                    }
                },
                "required": []
            },
            required_permissions=[ToolPermission.READ],
            tags=["tag", "list", "metadata"]
        )

    @monitor_performance
    @requires_permission(ToolPermission.READ)
    async def _execute_impl(self, **parameters) -> dict[str, Any]:
        """执行标签列表获取.

        Args:
            **parameters: 工具参数

        Returns:
            标签列表和统计

        """
        # 验证参数
        params = TagListParams(**parameters)

        # 获取标签列表
        tag_list = self._api.getTagList()

        # 处理标签信息
        tags_data = []
        total_usage = 0

        for tag in tag_list:
            tag_info = {
                "name": tag.get("name"),
                "type": tag.get("type", "general"),
                "color": tag.get("color"),
                "description": tag.get("description")
            }

            # 包含使用计数
            if params.include_counts:
                usage_count = tag.get("usage_count", 0)
                tag_info["usage_count"] = usage_count
                total_usage += usage_count

            tags_data.append(tag_info)

        # 排序
        if params.sort_by == "name":
            tags_data.sort(key=lambda x: x["name"].lower())
        elif params.sort_by == "count" and params.include_counts:
            tags_data.sort(key=lambda x: x["usage_count"], reverse=True)

        # 分类统计
        tag_categories = Counter(tag.get("type", "general") for tag in tag_list)

        return {
            "tags": tags_data,
            "total_tags": len(tags_data),
            "total_usage": total_usage if params.include_counts else None,
            "categories": dict(tag_categories),
            "sort_by": params.sort_by
        }


class ProjectStatsTool(BaseTool):
    """项目统计信息工具."""

    def _build_metadata(self) -> ToolMetadata:
        """构建工具元数据."""
        return ToolMetadata(
            name="project_stats",
            description="获取项目的详细统计信息",
            version="1.0.0",
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            required_permissions=[ToolPermission.READ],
            tags=["stats", "analytics", "metrics"]
        )

    @monitor_performance
    @requires_permission(ToolPermission.READ)
    async def _execute_impl(self, **parameters) -> dict[str, Any]:
        """执行项目统计获取.

        Args:
            **parameters: 工具参数

        Returns:
            项目统计信息

        """
        # 获取项目统计
        stats = self._api.getProjectStats()

        # 格式化统计信息
        return {
            "content_statistics": {
                "total_words": stats.get("totalWords", 0),
                "total_characters": stats.get("totalChars", 0),
                "total_characters_no_spaces": stats.get("totalCharsNoSpaces", 0),
                "total_paragraphs": stats.get("totalParagraphs", 0),
                "total_sentences": stats.get("totalSentences", 0)
            },
            "document_statistics": {
                "total_documents": stats.get("documentCount", 0),
                "novel_documents": stats.get("novelDocCount", 0),
                "note_documents": stats.get("noteDocCount", 0),
                "trash_documents": stats.get("trashDocCount", 0),
                "folder_count": stats.get("folderCount", 0)
            },
            "structure_statistics": {
                "chapter_count": stats.get("chapterCount", 0),
                "scene_count": stats.get("sceneCount", 0),
                "average_chapter_length": stats.get("avgChapterLength", 0),
                "average_scene_length": stats.get("avgSceneLength", 0),
                "longest_chapter": stats.get("longestChapter", 0),
                "shortest_chapter": stats.get("shortestChapter", 0)
            },
            "writing_progress": {
                "daily_word_count": stats.get("dailyWordCount", 0),
                "weekly_word_count": stats.get("weeklyWordCount", 0),
                "monthly_word_count": stats.get("monthlyWordCount", 0),
                "completion_percentage": stats.get("completionPercentage", 0),
                "estimated_reading_time_minutes": stats.get("estimatedReadingTime", 0)
            },
            "metadata": {
                "last_updated": stats.get("lastUpdated"),
                "project_created": stats.get("projectCreated"),
                "total_editing_time_hours": (
                    stats.get("totalEditingTime", 0) / 3600 
                    if stats.get("totalEditingTime") else 0
                )
            }
        }
