// 任务管理器
document.addEventListener('DOMContentLoaded', function() {
    // 等待数据库初始化
    setTimeout(async function() {
        if (!window.taskDB) {
            console.error('任务数据库未初始化');
            return;
        }
        
        // DOM元素
        const taskInput = document.getElementById('task-input');
        const addTaskBtn = document.getElementById('add-task-btn');
        const taskLists = {
            today: document.getElementById('today-tasks'),
            all: document.getElementById('all-tasks'),
            deleted: document.getElementById('deleted-tasks')
        };
        const taskTabs = document.querySelectorAll('.task-tab');
        const taskContainers = document.querySelectorAll('.task-container');
        const dateFilter = document.getElementById('date-filter');
        const clearDeletedBtn = document.getElementById('clear-deleted-btn');
        const exportTasksBtn = document.getElementById('export-tasks-btn');
        const importTasksBtn = document.getElementById('import-tasks-btn');
        const taskStats = document.getElementById('task-stats');
        
        // 当前活动标签
        let activeTab = 'today';
        
        // 初始化
        await loadTasks();
        
        // 添加事件监听器
        if (addTaskBtn) {
            addTaskBtn.addEventListener('click', addTask);
        }
        
        if (taskInput) {
            taskInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    addTask();
                }
            });
        }
        
        // 标签切换
        taskTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.getAttribute('data-tab');
                switchTab(tabId);
            });
        });
        
        // 日期过滤
        if (dateFilter) {
            dateFilter.addEventListener('change', async () => {
                await loadTasksByDate(dateFilter.value);
            });
            
            // 设置默认日期为今天
            const today = new Date();
            const year = today.getFullYear();
            const month = String(today.getMonth() + 1).padStart(2, '0');
            const day = String(today.getDate()).padStart(2, '0');
            dateFilter.value = `${year}-${month}-${day}`;
        }
        
        // 清空已删除任务
        if (clearDeletedBtn) {
            clearDeletedBtn.addEventListener('click', async () => {
                if (confirm('确定要清空所有已删除的任务吗？此操作不可恢复。')) {
                    await window.taskDB.clearDeletedTasks();
                    await loadDeletedTasks();
                    updateTaskStats();
                }
            });
        }
        
        // 导出任务
        if (exportTasksBtn) {
            exportTasksBtn.addEventListener('click', async () => {
                try {
                    const tasksJson = await window.taskDB.exportTasks();
                    const blob = new Blob([tasksJson], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `pomodoro-tasks-${new Date().toISOString().split('T')[0]}.json`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    
                    if (window.showInAppNotification) {
                        window.showInAppNotification('导出成功', '任务数据已成功导出');
                    }
                } catch (error) {
                    console.error('导出任务失败:', error);
                    alert('导出任务失败: ' + error.message);
                }
            });
        }
        
        // 导入任务
        if (importTasksBtn) {
            importTasksBtn.addEventListener('click', () => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.json';
                
                input.addEventListener('change', async (e) => {
                    const file = e.target.files[0];
                    if (!file) return;
                    
                    const reader = new FileReader();
                    reader.onload = async (event) => {
                        try {
                            const tasksJson = event.target.result;
                            await window.taskDB.importTasks(tasksJson);
                            await loadTasks();
                            
                            if (window.showInAppNotification) {
                                window.showInAppNotification('导入成功', '任务数据已成功导入');
                            }
                        } catch (error) {
                            console.error('导入任务失败:', error);
                            alert('导入任务失败: ' + error.message);
                        }
                    };
                    
                    reader.readAsText(file);
                });
                
                input.click();
            });
        }
        
        // 添加任务
        async function addTask() {
            const text = taskInput.value.trim();
            if (!text) return;
            
            try {
                const task = {
                    text: text,
                    completed: false,
                    createdAt: new Date().toISOString(),
                    status: 'active'
                };
                
                await window.taskDB.addTask(task);
                taskInput.value = '';
                
                // 重新加载任务
                await loadTasks();
                
                // 如果不在今日标签，切换到今日标签
                if (activeTab !== 'today') {
                    switchTab('today');
                }
            } catch (error) {
                console.error('添加任务失败:', error);
            }
        }
        
        // 切换标签
        function switchTab(tabId) {
            // 更新活动标签
            activeTab = tabId;
            
            // 更新标签样式
            taskTabs.forEach(tab => {
                if (tab.getAttribute('data-tab') === tabId) {
                    tab.classList.add('active');
                } else {
                    tab.classList.remove('active');
                }
            });
            
            // 更新任务容器显示
            taskContainers.forEach(container => {
                if (container.getAttribute('data-tab') === tabId) {
                    container.style.display = 'block';
                } else {
                    container.style.display = 'none';
                }
            });
            
            // 显示/隐藏日期过滤器
            if (dateFilter) {
                const dateFilterContainer = document.querySelector('.date-filter');
                if (dateFilterContainer) {
                    dateFilterContainer.style.display = tabId === 'all' ? 'flex' : 'none';
                }
            }
        }
        
        // 加载所有任务
        async function loadTasks() {
            await Promise.all([
                loadTodayTasks(),
                loadAllTasks(),
                loadDeletedTasks()
            ]);
            
            updateTaskStats();
        }
        
        // 加载今日任务
        async function loadTodayTasks() {
            if (!taskLists.today) return;
            
            try {
                const tasks = await window.taskDB.getTodayTasks();
                renderTasks(tasks, taskLists.today, 'today');
            } catch (error) {
                console.error('加载今日任务失败:', error);
            }
        }
        
        // 加载所有任务
        async function loadAllTasks() {
            if (!taskLists.all) return;
            
            try {
                const tasks = await window.taskDB.getActiveTasks();
                renderTasks(tasks, taskLists.all, 'all');
            } catch (error) {
                console.error('加载所有任务失败:', error);
            }
        }
        
        // 加载已删除任务
        async function loadDeletedTasks() {
            if (!taskLists.deleted) return;
            
            try {
                const tasks = await window.taskDB.getDeletedTasks();
                renderTasks(tasks, taskLists.deleted, 'deleted');
            } catch (error) {
                console.error('加载已删除任务失败:', error);
            }
        }
        
        // 按日期加载任务
        async function loadTasksByDate(dateStr) {
            if (!taskLists.all) return;
            
            try {
                const tasks = await window.taskDB.getTasksByDate(dateStr);
                renderTasks(tasks, taskLists.all, 'all');
            } catch (error) {
                console.error('按日期加载任务失败:', error);
            }
        }
        
        // 渲染任务列表
        function renderTasks(tasks, container, tabId) {
            container.innerHTML = '';
            
            if (tasks.length === 0) {
                const noTasks = document.createElement('div');
                noTasks.className = 'no-tasks';
                noTasks.textContent = tabId === 'deleted' ? '没有已删除的任务' : '没有任务';
                container.appendChild(noTasks);
                return;
            }
            
            // 按创建时间排序，最新的在前面
            tasks.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
            
            tasks.forEach(task => {
                const taskItem = document.createElement('li');
                taskItem.className = 'task-item';
                taskItem.dataset.id = task.id;
                
                const taskContent = document.createElement('div');
                taskContent.className = 'task-content';
                
                const taskText = document.createElement('span');
                taskText.className = 'task-text';
                if (task.completed) {
                    taskText.classList.add('completed');
                }
                taskText.textContent = task.text;
                
                const taskDate = document.createElement('div');
                taskDate.className = 'task-date';
                taskDate.textContent = formatDate(task.createdAt);
                
                taskContent.appendChild(taskText);
                taskContent.appendChild(taskDate);
                
                const taskActions = document.createElement('div');
                taskActions.className = 'task-actions';
                
                if (tabId === 'deleted') {
                    // 已删除任务的操作按钮
                    const restoreBtn = document.createElement('button');
                    restoreBtn.className = 'task-btn restore-btn';
                    restoreBtn.innerHTML = ICONS.restore;
                    restoreBtn.title = '恢复任务';
                    restoreBtn.addEventListener('click', () => restoreTask(task.id));
                    
                    taskActions.appendChild(restoreBtn);
                } else {
                    // 普通任务的操作按钮
                    const completeBtn = document.createElement('button');
                    completeBtn.className = 'task-btn complete-btn';
                    completeBtn.innerHTML = ICONS.check;
                    completeBtn.title = task.completed ? '标记为未完成' : '标记为已完成';
                    completeBtn.addEventListener('click', () => toggleTaskComplete(task.id));
                    
                    const deleteBtn = document.createElement('button');
                    deleteBtn.className = 'task-btn delete-btn';
                    deleteBtn.innerHTML = ICONS.delete;
                    deleteBtn.title = '删除任务';
                    deleteBtn.addEventListener('click', () => deleteTask(task.id));
                    
                    taskActions.appendChild(completeBtn);
                    taskActions.appendChild(deleteBtn);
                }
                
                taskItem.appendChild(taskContent);
                taskItem.appendChild(taskActions);
                
                container.appendChild(taskItem);
            });
        }
        
        // 切换任务完成状态
        async function toggleTaskComplete(taskId) {
            try {
                await window.taskDB.completeTask(taskId);

                await loadTasks();
            } catch (error) {
                console.error('切换任务状态失败:', error);
            }
        }
        
        // 删除任务
        async function deleteTask(taskId) {
            try {
                await window.taskDB.deleteTask(taskId);
                await loadTasks();
            } catch (error) {
                console.error('删除任务失败:', error);
            }
        }
        
        // 恢复任务
        async function restoreTask(taskId) {
            try {
                await window.taskDB.restoreTask(taskId);
                await loadTasks();
            } catch (error) {
                console.error('恢复任务失败:', error);
            }
        }
        
        // 更新任务统计
        async function updateTaskStats() {
            if (!taskStats) return;
            
            try {
                const allTasks = await window.taskDB.getActiveTasks();
                const completedTasks = allTasks.filter(task => task.completed);
                const todayTasks = await window.taskDB.getTodayTasks();
                const todayCompleted = todayTasks.filter(task => task.completed);
                
                taskStats.innerHTML = `
                    总计: ${allTasks.length} | 已完成: ${completedTasks.length} | 
                    今日: ${todayTasks.length} | 今日已完成: ${todayCompleted.length}
                `;
            } catch (error) {
                console.error('更新任务统计失败:', error);
            }
        }
        
        // 格式化日期
        function formatDate(dateStr) {
            const date = new Date(dateStr);
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            const hours = String(date.getHours()).padStart(2, '0');
            const minutes = String(date.getMinutes()).padStart(2, '0');
            
            return `${year}-${month}-${day} ${hours}:${minutes}`;
        }
    }, 500); // 给数据库初始化一些时间
});