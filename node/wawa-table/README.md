DEBUG=wawa-table npm start

supervisor bin/www

forever start bin/www
forever stop bin/www

使用forever运行nodejs应用
http://tcrct.iteye.com/blog/2043644

## todo
- 日志记录request_id, file, line_no, client_ip, 
- API测试

## sql

    CREATE TABLE `users` (
        `id` int(11) NOT NULL auto_increment,
        `username` varchar(128) not null comment '用户名',
        `status` tinyint not null default '0' comment '状态',
        `registerTime` datetime not null comment '注册时间',
        `registerIp` varchar(32) not null comment '注册IP',
        `lastLoginTime` datetime not null comment '最后登录时间',
        `lastLoginIp` varchar(32) not null comment '最后登录IP',
        PRIMARY KEY (`id`)
    );

    CREATE TABLE `workbooks` (
        `id` int(11) NOT NULL auto_increment,
        `userId` int(11) NOT NULL default '0' comment '用户ID',
        `title` varchar(128) not null comment '标题',
        `content` text not null comment '内容',
        `shareTo` varchar(1024) not null default '' comment '共享给谁',
        `status` tinyint not null default '0' comment '状态',
        `createTime` datetime not null comment '创建时间',
        `updateTime` datetime not null comment '修改时间',
        PRIMARY KEY (`id`),
        key(`userid`)
    );

