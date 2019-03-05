<?php
$w = new Wawa();

$w->get('index', function($w) {
    $rows = $w->fetch('SELECT * from users');
    $w->render('views/index.php', ['rows' => $rows]);   
});

$w->get('new', function($w) {
    $w->render('views/user_form.php', ['action' => 'new']);
});

$w->get('modify', function($w) {
    $rows = $w->fetch('SELECT * from users where id = ?', [$_GET['id']]);
    $w->render('views/user_form.php', ['action' => 'modify', 'user' => $rows[0]]);
});

$w->post('add', function($w) {   
    $sql = 'INSERT INTO `users`(`username`, `nickname`, `password`, `created_at`) VALUES (?, ?, ?, ?)';
    $w->execute($sql, [$_POST['username'], $_POST['nickname'], $_POST['password'], date('Y-m-d H:i:s')]);
    $w->redirect('index');
});

$w->post('update', function($w) {
    $sql = 'update users set nickname=?, password=? where id = ?';
    $w->execute($sql, [$_POST['nickname'], $_POST['password'], $_POST['id']]);
    $w->redirect('index');
});

$w->get('remove', function($w) {
    $rows = $w->fetch('SELECT * from users where id = ?', [$_GET['id']]);
    $w->render('views/user_delete.php', ['user' => $rows[0]]);
});

$w->post('remove', function($w) { 
    $w->execute('delete from users where id = ?', [$_POST['id']]);
    $w->redirect('index');
});

$w->run();