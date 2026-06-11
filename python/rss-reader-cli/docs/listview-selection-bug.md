# Textual ListView 选中高亮问题排查

## 问题描述

在 RSS 阅读器中，从 Feed 列表进入 Article 列表后，第一篇文章虽然 `index = 0` 被设置，但没有视觉高亮。按 `j` 向下移动后，第二项才显示高亮。

## 排查过程

1. **初步尝试**：在 `on_mount` 中添加 `focus()` 调用 - 无效
2. **尝试 `call_later`**：延迟 focus 执行 - 无效
3. **尝试 `asyncio.sleep(0)`**：让出事件循环 - 无效
4. **尝试添加 CSS 样式**：为 `ListItem:focus` 设置背景色 - Textual 不支持 `:selected` 伪类
5. **创建测试文件**：发现简单的测试用例能正常工作

## 关键发现

对比正常工作的 `test_selection.py` 和不正常工作的 `test_diff.py` 以及原始 app，发现问题出在 `_refresh_and_select_article` 方法：

```python
# 有问题的代码
def _refresh_and_select_article(self, article_id: int | None):
    self._load_articles()  # 这里会清空并重建列表
    # ... 设置 index 和 focus
```

当通过 `call_later` 延迟调用时，`_load_articles()` 会重建列表，但在列表渲染完成前就调用了 `focus()`，导致焦点虽然设置在正确的 index 上，但视觉高亮没有正确显示。

## 解决方案

1. `on_mount` 中直接同步调用 `focus()`
2. `_refresh_and_select_article` 不再重新加载数据，只负责设置 index 和 focus

```python
def on_mount(self) -> None:
    self._load_articles()
    self.query_one("#article_list", ListView).focus()

def _refresh_and_select_article(self, article_id: int | None):
    article_list = self.query_one("#article_list", ListView)
    if article_list.children:
        if article_id is not None:
            for i, item in enumerate(article_list.children):
                if hasattr(item, 'key') and int(item.key) == article_id:
                    article_list.index = i
                    article_list.focus()
                    return
        if article_list.index is None:
            article_list.index = 0
        article_list.focus()
    else:
        article_list.focus()
```

## 经验教训

当通过 `call_later` 或其他延迟机制执行 UI 操作时，要避免在回调中重新构建正在被渲染的 UI 组件。列表的重建和焦点的设置应该分开进行，或者在同步上下文中完成。
