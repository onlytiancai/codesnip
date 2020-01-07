<!DOCTYPE html>
<html>
    <head>
    </head>
    <body>
        <h1><?= $this->config['site_name'] ?></h1>
        <h2>删除用户</h2>
        <form action="<?= $this->e($this->sitePrefix . 'index/remove')?>" method="post">
            <input type="hidden" name="id" value="<?= $this->e($data['user']['id'])?>">
            
            <p>用户名：<?= $this->e($data['user']['username']) ?></p>
            
            <p>昵称：<?= $this->e($data['user']['nickname']) ?> </p>     
            
            <p><input type="submit" value="确认删除"></p>
        </form>
    </body>
</html>