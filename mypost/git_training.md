# git training

- 为什么要用 git
- add, commit, push 三板斧
- 查看操作
- 撤销操作
- 分支操作
- 常见任务及问题
- 好习惯

---

# 为什么要用代码管理工具

- 防止代码丢失
	- 硬盘炸掉，地震，办公室失火	
- 团队协作
	- 合并多个人的工作成果
- 版本管理
	- 查看历史，排查问题，代码回滚

---
# 为什么要用 git

因为它是 Linus 写的

---

# git 的基本原理

两类代码仓库管理方式：

- 集中式：CVS, SVN, VSS
	- 提交代码必须联网
	- 对文件修改需要加锁（某些）
- 分布式：git
	- 本地代码提交
	- 联网后同步代码（推送及拉取）

仓库文件格式：

- 只保存对文件改动的增量部分，而不是整个副本
- 大大节省保存文件各个版本所需要的磁盘空间
- 切分支只是改动一个指针

---
# 安装

- ubuntu：apt-get install git
- centos: yum install git-core
- windows：https://gitforwindows.org/

配置：
```
git config --global user.name "username"
git config --global user.email "email"
git config --global core.quotepath false
git config --global core.editor vim
git config --global core.autocrlf true
git config --global core.safecrlf warn
```

---
# 创建和克隆仓库

自己新建 git 项目： 
 
`git init`

拉取别人 git 项目： 

`git clone https://github.com/521xueweihan/git-tips`

git 认证
- http 方式：输入用户名和密码
- ssh 方式：把公钥配置在 github, gitlab 上
	- windows：使用 git gui 生成
	- linux：`ssh-keygen -t rsa` `cat ~/.ssh/id_rsa.pub` 
---

# add, commit, push 三板斧

- `git add file`：提交`工作区`改动到`索引区`
- `git commit -m 'message'`：提交`索引区`改动到`本地仓库`
- `git push`：提交`本地仓库`改动到`远程仓库`

> 如果 `git push` 失败，通常是因为本地代码太旧，需要 `git pull` 更新远程代码到本地。

理解概念：
- 工作区
- 索引区（缓存区）
- 本地仓库（本地分支，HEAD）
- 远程仓库（远程分支）

---

# 查看操作

---

# 查看改动文件

```
$ git status
On branch master

No commits yet

Changes to be committed:
        new file:   readme.md
Untracked files:
        changelist.md
```

- 索引区文件
- 未跟踪文件

---
# 查看文件改动

```
$ git diff
diff --git a/readme.md b/readme.md
index 641d574..a60db1a 100644
--- a/readme.md
+++ b/readme.md
@@ -1,3 +1,3 @@
 111
-222
 333
+444

```

- `git diff -w：忽略空白
- `git diff --cached`：查看索引区和本地仓库的变更
- `git diff master dev`：查看 dev 对 master 进行的变更

---
# 查看改动历史

- `git log`：查看提交历史
- `git log --stat`: 查看文件改动
- `git log -p`：查看具体改动
- ` git log --oneline --graph`：查看分支合并图


--- 
# 查看某个文件的历史快照

查看某次提交的文件了列表
```
$ git show 0b2988eb^{tree}

changelist.md
readme.md
```
查看当时这个文件的快照
```
$ git show 0b2988eb:readme.md
111
333
444
```

---
# 按行查看某个文件的历史改动

```
$ git blame readme.md
^531b20a (haohu 2019-04-20 09:42:46 +0800 1) 111
^531b20a (haohu 2019-04-20 09:42:46 +0800 2) 333
30db61a0 (haohu 2019-04-20 09:51:07 +0800 3) 444
```

---

# 撤销操作

---
# 撤销掉已 `push` 的提交

如果某次已经 `push` 的提交发现问题，需要撤销，则

```
git revert e56ef5ff
```

- 撤销掉某一次提交，相当于 `git commit` 的反操作，但会提交一次新的 `commit`
- 这是最安全，最基本的撤销场景，`revert` 后可以继续进行 `push` 操作

---
# 修改已 `commit` 的消息

修改最后一次提交的消息

```
git commit --amend
```

修改之前的提交信息

```
# 执行 rebase
git rebase -i 24b347f
# 在弹出的编辑器里最上面插入如下代码后保存退出
r 24b347f
# 修改 24b347f 的提交信息
git commit --amend
# 继续 rebase
git rebase --continue
```
---
# 撤销未 `commit` 的改动

- 如果本地修改还未 `add` 到索引区，可以用 `checkout` 到原始状态
- 撤销到 `HEAD` 版本： `git checkout readme.md`
- 撤销到某个提交点版本：`git checkout 4b7f7c5e readme.md`

>你用这种方法“撤销”的任何修改真的会完全消失。因为它们从来没有被提交过，所以之后 Git 也无法帮助我们恢复它们

---

# 撤销已 `commit` 未 `push` 的提交

```
git reset 0b2988eb
git status
cat readme.md
```
- 直接回到某个提交点，最近的提交就跟没发生一样
- 默认情况会保留工作目录，即改动还存在，只是未提交，这样比较安全一些
- 如果想完全丢弃改动，可加 `--hard` 参数
- 如果 reset 到已 push 的提交点，则 push 会拒绝，不要这样做

---
# 任意恢复文件

如果 `git reset` 后又后悔了，可以再次进行 `reset`

```
git reflog
$ git reflog
  a363770 (HEAD -> master) HEAD@{0}: aaa
  e56ef5f HEAD@{1}: bbb
  1741752 HEAD@{2}: ccc
git reset 'HEAD@{1}'
```

- `reflog` 可以恢复任何提交过的东西，但 `reflog` 只记录 HEAD 的改动，不是永久的，且是本地的
- 可以用 `git checkout <SHA> -- <filename>` 恢复其中一个文件
- 重新执行某一次提交：`git cherry-pick 174175260`

---

# 分支操作

---
# 理解分支

- 分支是 git 的精髓，将功能分离到不同的分支对于任何严肃的开发人员来说都是至关重要的。
- 通过分离每个功能，错误修复或正在运行的实验，您将避免很多问题并保持开发分支的清洁。
- 永久分支：
  - master: 用于生产环境部署 
  - testing: 用于测试环境测试
  - dev: 用于日常开发
  - user-xxx：保存一些个人提交信息
- 临时分支：
	- feature-xxx
	- hotfix-xxx
- 理解 git-flow

---

# 分支基本操作

- 查看当前所在分支：`git branch`
- 查看所有分支，包括远程分支：`git branch -a`
- 从当前分支创建新分支并切换过去：`git checkout -b user-xxx`
- 切换到另一个分支：`git checkout master`
- 删除分支：`git branch -d user-xxx`
- 新建远程分支：`git push origin user-xxx:user-xxx`
- 删除远程分支：`git push origin :user-xxx`


---
# 合并分支

比如要把 `dev` 合并到 `master`

```
# 切换到 master 分支
git checkout master
# 确认已切换到 master 分支
git branch
# 确认要合并内容
git diff dev --stat
# 合并分支，弹出的编辑器直接保存，不要改动
git merge dev --no-ff
# 查看分支合并记录
git log --graph --oneline --simplify-by-decoration --all
```

---

# --no-ff 和 --squash

- 默认情况下 git 合并会使用快进式合并，即把 HEAD 直接指向 `dev` ，查看历史看不到分支信息
- 使用 `--no-ff` 后，则会在 `master` 会生成新的提交，并“开叉”，可以让我们的提交历史更加的清晰！
- 如果临时分支里提交信息比较乱，合并过来的时候想重新设置提交信息，则可以使用 `--squash` 参数

```
git merge dev --squash --no-ff
git commit -m 'dev to master: some message'
```
---
# 解决冲突

合并时冲突
```
$ git merge --no-ff dev
Auto-merging 2.txt
CONFLICT (content): Merge conflict in 2.txt
```

- 解决冲突：`vi 2.txt`，去掉 `<<<<` , `====`, `>>>>`
	- 使用本分支版本，使用被合并分支版本，或两者都要 
- 添加修改：`git add 2.txt`
- 提交修改：`git commit`，弹出的编辑器不要修改，直接保存

---

# 远程分支合并到本地分支

```
# 拉取并合并
git pull origin master
# 先拉取
git fetch origin master
# 查看改动范围
git diff HEAD origin/master --stat
# 合并
git merge origin/master
```

---
# 常见任务及问题

---
# 暂存代码

在对工作区代码进行一些改动，但未提交时，如果需要紧急切到别的分支做修改，或需要合并其它分支时会报错：

```
$ git checkout dev
error: Your local changes to the following 
files would be overwritten by checkout:
    2.txt
```

需要先暂存代码，再切到另一个分支，切回本分支后再从暂存区取出改动

```
git stash save 'some message' # 暂存代码
git checkout dev    # 切到另一个分支做改动
git checkout master # 切回本分支
git stash list      # 查看暂存区
git stash pop       # 从暂存区取出代码
git status          # 查看是否取出成功
```

---
# 删除和重命名文件

- 删除：`git rm 1.txt`
- 重命名：`git mv 1.txt 2.txt`

---
# 忽略某些文件

```
vi .gitignore
	*.pyc
    *.swp
    *.class
    *.jar
```
---

#  禁用 `git push -f`

如果 `rebase` 了已经在远程仓库存在的提交，则 `push` 时会拒绝，虽然加 `-f` 可以提交，但千万不要这么做，因为会覆盖别人的改动。

- 只 rebase 未 push 的部分
- 想维持树的整洁，方法就是：在git push之前，先git fetch，再git rebase


```
git fetch origin master
git rebase origin/master
git push
```

https://segmentfault.com/q/1010000000430041

---

# 好习惯

- 每天上班 `git pull`
- 每天下班 `git push`
- 提交代码前一定要 `git diff` 确认修改内容，防止夹带