SELECT
    NOW() AS check_time, -- 获取当前时间，作为检查时间点
    TIMESTAMPDIFF(SECOND, trx.trx_wait_started, NOW()) AS wait_time_seconds, -- 计算事务等待时间（秒）

    -- 等待锁的事务/线程信息
    r.ENGINE_TRANSACTION_ID AS waiting_trx_id, -- 等待锁的事务ID
    r.THREAD_ID AS waiting_thread_id, -- 等待锁的线程ID (Performance Schema中的)
    wp.ID AS waiting_process_id, -- 等待锁的进程ID (information_schema.processlist中的，与SHOW PROCESSLIST的Id一致)
    wp.USER AS waiting_user, -- 等待锁的连接用户
    wp.HOST AS waiting_host_port, -- 等待锁的客户端主机和端口（如 172.31.6.246:59532）
    SUBSTRING_INDEX(wp.HOST, ':', 1) AS waiting_ip, -- 从主机信息中提取IP地址
    SUBSTRING_INDEX(wp.HOST, ':', -1) AS waiting_port, -- 从主机信息中提取端口
    wes.SQL_TEXT AS waiting_current_sql, -- 等待锁的线程当前正在执行的SQL语句 (来自events_statements_current)
    wesh.SQL_TEXT AS waiting_last_executed_sql, -- 等待锁的线程最后执行完成的SQL语句 (来自events_statements_history)
    r.OBJECT_SCHEMA AS waiting_schema, -- 等待锁的表所在的数据库名
    r.OBJECT_NAME AS waiting_table, -- 等待锁的表名
    r.INDEX_NAME AS waiting_index, -- 等待锁的索引名
    r.LOCK_MODE AS waiting_lock_mode, -- 等待的锁模式 (如 S, X, IS, IX)
    r.LOCK_TYPE AS waiting_lock_type, -- 等待的锁类型 (如 RECORD, TABLE)
    r.LOCK_DATA AS waiting_lock_data, -- 等待锁的数据（对于行锁是主键值或索引值）

    -- 阻塞锁的事务/线程信息
    b.ENGINE_TRANSACTION_ID AS blocking_trx_id, -- 阻塞锁的事务ID
    b.THREAD_ID AS blocking_thread_id, -- 阻塞锁的线程ID (Performance Schema中的)
    bp.ID AS blocking_process_id, -- 阻塞锁的进程ID (information_schema.processlist中的，与SHOW PROCESSLIST的Id一致)
    bp.USER AS blocking_user, -- 阻塞锁的连接用户
    bp.HOST AS blocking_host_port, -- 阻塞锁的客户端主机和端口
    SUBSTRING_INDEX(bp.HOST, ':', 1) AS blocking_ip, -- 从主机信息中提取IP地址
    SUBSTRING_INDEX(bp.HOST, ':', -1) AS blocking_port, -- 从主机信息中提取端口
    bes.SQL_TEXT AS blocking_current_sql, -- 阻塞锁的线程当前正在执行的SQL语句 (来自events_statements_current)
    besh.SQL_TEXT AS blocking_last_executed_sql, -- 阻塞锁的线程最后执行完成的SQL语句 (来自events_statements_history)
    b.OBJECT_SCHEMA AS blocking_schema, -- 阻塞锁的表所在的数据库名
    b.OBJECT_NAME AS blocking_table, -- 阻塞锁的表名
    b.INDEX_NAME AS blocking_index, -- 阻塞锁的索引名
    b.LOCK_MODE AS blocking_lock_mode, -- 持有的锁模式 (如 S, X, IS, IX)
    b.LOCK_TYPE AS blocking_lock_type, -- 持有的锁类型 (如 RECORD, TABLE)
    b.LOCK_DATA AS blocking_lock_data -- 持有的锁数据（对于行锁是主键值或索引值）

FROM performance_schema.data_lock_waits w -- 从data_lock_waits表开始，显示哪些锁在等待
JOIN performance_schema.data_locks r ON w.REQUESTING_ENGINE_LOCK_ID = r.ENGINE_LOCK_ID -- 连接请求锁的信息 (等待者)
JOIN performance_schema.data_locks b ON w.BLOCKING_ENGINE_LOCK_ID = b.ENGINE_LOCK_ID -- 连接持有锁的信息 (阻塞者)
JOIN information_schema.innodb_trx trx ON r.ENGINE_TRANSACTION_ID = trx.trx_id -- 连接innodb_trx表获取事务状态和等待开始时间

LEFT JOIN performance_schema.threads wt ON r.THREAD_ID = wt.THREAD_ID -- 连接等待线程的详细信息
LEFT JOIN performance_schema.threads bt ON b.THREAD_ID = bt.THREAD_ID -- 连接阻塞线程的详细信息

LEFT JOIN performance_schema.processlist wp ON wt.PROCESSLIST_ID = wp.ID -- 连接等待线程的processlist信息（获取用户、主机等）
LEFT JOIN performance_schema.processlist bp ON bt.PROCESSLIST_ID = bp.ID -- 连接阻塞线程的processlist信息（获取用户、主机等）

LEFT JOIN performance_schema.events_statements_current wes ON wp.ID = wes.THREAD_ID -- 连接等待线程当前执行的SQL
LEFT JOIN performance_schema.events_statements_current bes ON bp.ID = bes.THREAD_ID -- 连接阻塞线程当前执行的SQL

-- 子查询：获取等待线程最近一次执行完成的SQL
LEFT JOIN (
    SELECT
        THREAD_ID,
        SQL_TEXT,
        ROW_NUMBER() OVER (PARTITION BY THREAD_ID ORDER BY EVENT_ID DESC) as rn -- 为每个线程的SQL历史记录按最新排序并编号
    FROM performance_schema.events_statements_history -- 从SQL历史事件表中查询
) wesh ON wt.THREAD_ID = wesh.THREAD_ID AND wesh.rn = 1 -- 连接等待线程ID，并只取最新的SQL

-- 子查询：获取阻塞线程最近一次执行完成的SQL
LEFT JOIN (
    SELECT
        THREAD_ID,
        SQL_TEXT,
        ROW_NUMBER() OVER (PARTITION BY THREAD_ID ORDER BY EVENT_ID DESC) as rn -- 为每个线程的SQL历史记录按最新排序并编号
    FROM performance_schema.events_statements_history -- 从SQL历史事件表中查询
) besh ON bt.THREAD_ID = besh.THREAD_ID AND besh.rn = 1 -- 连接阻塞线程ID，并只取最新的SQL

WHERE trx.trx_wait_started IS NOT NULL -- 过滤掉等待时间尚未开始的事务
  AND TIMESTAMPDIFF(SECOND, trx.trx_wait_started, NOW()) > 1; -- 只显示等待时间超过1秒的事务
