# Wawa Framework

迷你 php 框架

- 自动路由: `get`, `post`
- 数据库操作封装: `fetch`, `execute`
- 配置文件支持: `config`
- view 层封装：`render`
- 其它工具函数：`redirect`, `e`, `send404`

Apache 配置

```
<IfModule alias_module>
    Alias /me "D:/xampp/me/public"
    <Directory "D:/xampp/me/public">
        Options FollowSymLinks
        AllowOverride All       
        Require all granted
        
        RewriteEngine on  
        RewriteCond %{REQUEST_FILENAME} !-f  
        RewriteCond %{REQUEST_FILENAME} !-d  
        RewriteRule ^(.*)$ index.php/$1 [L]
    </Directory>
</IfModule>    
```