<!DOCTYPE html>
<html>
    <head>
    </head>
    <body>
        <h1><?= $this->config['site_name'] ?></h1>
        <h2><?= $data['action'] === 'new' ? '新增用户' : '修改用户' ?></h2>
        <form action="<?= $this->e($this->sitePrefix) ?>index/<?= $data['action'] === 'new' ? 'add' : 'update' ?>" method="post">
            <?= isset($data['user']) ? '<input type="hidden" name="id" value="' .$data['user']['id'] .'">' : ''?>
            
            <p>用户名：<input type="text" name="username" 
            <?= $data['action'] === 'new' ? '' : 'disabled' ?> 
            value="<?= $this->e(isset($data['user'])) ? $data['user']['username'] : '' ?>"
            ></p>
            
            <p>昵称：<input type="text" name="nickname" 
            value="<?= $this->e(isset($data['user'])) ? $data['user']['nickname'] : '' ?>"
            ></p>
            
            <p>密码：<input type="text" name="password" 
            value="<?= $this->e(isset($data['user'])) ? $data['user']['password'] : '' ?>"
            ></p>
            <p><input type="submit" value="提交"> <input type="reset" value="重新填写"></p>
        </form>
    </body>
</html>