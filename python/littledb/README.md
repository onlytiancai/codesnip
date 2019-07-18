# 简单数据库

一个简单的数据库

# Quick Start

    # 建库建表
    db = littledb.create_db('./mydb')
    table = db.create_table('test')

    # 插入数据
    table.upinsert(1, {'a': 1, 'b': 2, 'c': 'test'})
    table.upinsert(2, {'a': 3, 'b': 4})

    # 查找数据
    row = table.find(2)
    print(row['a'])

    # 更新数据
    table.upinsert(1, {'a': 5, 'b': 6})
    row = table.find(1)
    print(row['a'])

    # 加载库，加载表
    db = littledb.load_db('./mydb')
    table = db.load_table('test')
    row = table.find(1)
    print(row['a'])

# Feature List

基本功能

- 按主键插入，更新，查找数据
- 更新数据，会在数据文件末尾增加数据以覆盖旧数据
- 查找数据，不会遍历整个数据文件，是根据主键索引定位到偏移量后直接读取

# Todo List 

- 删除功能
- key 重复支持 
- 二级索引
- 范围查找
- sql 语法支持
- 多线程支持
- 替换掉 JSON 的序列化，提高性能
