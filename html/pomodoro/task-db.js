// 任务数据库管理
class TaskDB {
    constructor() {
        this.dbName = 'pomodoroTasksDB';
        this.storeName = 'tasks';
        this.db = null;
        this.initDB();
    }
    
    // 初始化数据库
    async initDB() {
        return new Promise((resolve, reject) => {
            if (!window.indexedDB) {
                console.error('您的浏览器不支持IndexedDB');
                reject('浏览器不支持IndexedDB');
                return;
            }
            
            const request = indexedDB.open(this.dbName, 1);
            
            request.onerror = (event) => {
                console.error('打开数据库失败:', event.target.error);
                reject(event.target.error);
            };
            
            request.onsuccess = (event) => {
                this.db = event.target.result;
                console.log('数据库连接成功');
                resolve();
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // 创建任务存储对象
                if (!db.objectStoreNames.contains(this.storeName)) {
                    const store = db.createObjectStore(this.storeName, { keyPath: 'id' });
                    
                    // 创建索引
                    store.createIndex('status', 'status', { unique: false });
                    store.createIndex('createdAt', 'createdAt', { unique: false });
                    store.createIndex('completedAt', 'completedAt', { unique: false });
                    
                    console.log('任务存储对象创建成功');
                }
            };
        });
    }
    
    // 添加任务
    async addTask(task) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            
            // 确保任务有必要的字段
            if (!task.id) task.id = Date.now().toString();
            if (!task.createdAt) task.createdAt = new Date().toISOString();
            if (!task.status) task.status = 'active';
            
            const request = store.add(task);
            
            request.onsuccess = () => {
                console.log('任务添加成功');
                resolve(task);
            };
            
            request.onerror = (event) => {
                console.error('添加任务失败:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    // 获取所有任务
    async getAllTasks() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.getAll();
            
            request.onsuccess = () => {
                resolve(request.result);
            };
            
            request.onerror = (event) => {
                console.error('获取任务失败:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    // 获取今日任务
    async getTodayTasks() {
        const allTasks = await this.getAllTasks();
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        return allTasks.filter(task => {
            const taskDate = new Date(task.createdAt);
            taskDate.setHours(0, 0, 0, 0);
            return taskDate.getTime() === today.getTime() && task.status !== 'deleted';
        });
    }
    
    // 获取活动任务（非删除状态）
    async getActiveTasks() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const index = store.index('status');
            const request = index.getAll('active');
            
            request.onsuccess = () => {
                resolve(request.result);
            };
            
            request.onerror = (event) => {
                console.error('获取活动任务失败:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    // 获取已删除任务
    async getDeletedTasks() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const index = store.index('status');
            const request = index.getAll('deleted');
            
            request.onsuccess = () => {
                resolve(request.result);
            };
            
            request.onerror = (event) => {
                console.error('获取已删除任务失败:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    // 获取指定日期的任务
    async getTasksByDate(dateStr) {
        const allTasks = await this.getActiveTasks();
        const targetDate = new Date(dateStr);
        targetDate.setHours(0, 0, 0, 0);
        
        return allTasks.filter(task => {
            const taskDate = new Date(task.createdAt);
            taskDate.setHours(0, 0, 0, 0);
            return taskDate.getTime() === targetDate.getTime();
        });
    }
    
    // 更新任务
    async updateTask(task) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            
            const request = store.put(task);
            
            request.onsuccess = () => {
                console.log('任务更新成功');
                resolve(task);
            };
            
            request.onerror = (event) => {
                console.error('更新任务失败:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    // 完成任务
    async completeTask(taskId) {
        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const request = store.get(taskId);
        
        return new Promise((resolve, reject) => {
            request.onsuccess = async () => {
                const task = request.result;
                if (task) {
                    task.completed = !task.completed;
                    if (task.completed) {
                        task.completedAt = new Date().toISOString();
                    } else {
                        task.completedAt = null;
                    }
                    
                    try {
                        await this.updateTask(task);
                        resolve(task);
                    } catch (error) {
                        reject(error);
                    }
                } else {
                    reject(new Error('任务不存在'));
                }
            };
            
            request.onerror = (event) => {
                console.error('获取任务失败:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    // 删除任务（标记为已删除）
    async deleteTask(taskId) {
        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const request = store.get(taskId);
        
        return new Promise((resolve, reject) => {
            request.onsuccess = async () => {
                const task = request.result;
                if (task) {
                    task.status = 'deleted';
                    task.deletedAt = new Date().toISOString();
                    
                    try {
                        await this.updateTask(task);
                        resolve(task);
                    } catch (error) {
                        reject(error);
                    }
                } else {
                    reject(new Error('任务不存在'));
                }
            };
            
            request.onerror = (event) => {
                console.error('获取任务失败:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    // 恢复已删除的任务
    async restoreTask(taskId) {
        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const request = store.get(taskId);
        
        return new Promise((resolve, reject) => {
            request.onsuccess = async () => {
                const task = request.result;
                if (task) {
                    task.status = 'active';
                    task.deletedAt = null;
                    
                    try {
                        await this.updateTask(task);
                        resolve(task);
                    } catch (error) {
                        reject(error);
                    }
                } else {
                    reject(new Error('任务不存在'));
                }
            };
            
            request.onerror = (event) => {
                console.error('获取任务失败:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    // 永久删除任务
    async permanentlyDeleteTask(taskId) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            
            const request = store.delete(taskId);
            
            request.onsuccess = () => {
                console.log('任务永久删除成功');
                resolve();
            };
            
            request.onerror = (event) => {
                console.error('永久删除任务失败:', event.target.error);
                reject(event.target.error);
            };
        });
    }
    
    // 清空已删除的任务
    async clearDeletedTasks() {
        const deletedTasks = await this.getDeletedTasks();
        const promises = deletedTasks.map(task => this.permanentlyDeleteTask(task.id));
        
        return Promise.all(promises);
    }
    
    // 导出任务数据
    async exportTasks() {
        const tasks = await this.getAllTasks();
        return JSON.stringify(tasks);
    }
    
    // 导入任务数据
    async importTasks(tasksJson) {
        try {
            const tasks = JSON.parse(tasksJson);
            if (!Array.isArray(tasks)) {
                throw new Error('无效的任务数据格式');
            }
            
            const promises = tasks.map(task => this.addTask(task));
            await Promise.all(promises);
            
            return true;
        } catch (error) {
            console.error('导入任务失败:', error);
            throw error;
        }
    }
}

// 创建并导出任务数据库实例
window.taskDB = new TaskDB();