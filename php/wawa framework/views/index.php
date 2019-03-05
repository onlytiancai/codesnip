<!DOCTYPE html>
<html>
    <head>
    </head>
    <body>
        <h1><?= $this->config['site_name'] ?></h1>
        <p><a href="<?= $this->e($this->sitePrefix . 'index/new' ) ?>">新建</a></p>
        <table border="1">
            <?php foreach ($data['rows'] as $row): ?>
            <tr>
                <td><?= $this->e($row['id']) ?></td>
                <td><?= $this->e($row['username']) ?></td>
                <td><?= $this->e($row['nickname']) ?></td>
                <td><?= $this->e($row['created_at']) ?></td>
                <td>
                    <a href="<?= $this->e($this->sitePrefix . 'index/modify?id=' . $row['id']) ?>">修改</a>
                    <a href="<?= $this->e($this->sitePrefix . 'index/remove?id=' . $row['id']) ?>">删除</a>
                </td>
            </tr>
            <?php endforeach;?>
        </table>
    </body>
</html>