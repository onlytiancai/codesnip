### 分支管理

最少有三个长期分支

- master: 用于生产环境部署 
- testing: 用于测试环境测试
- dev: 用于日常开发

有一些临时分支

- feture-xxx: 用于增加一个新功能
- hotfix-xxx: 用于修复一个紧急bug

### 常见任务

#### 增加一个新功能

    (dev)$: git checkout -b feture-xxx            # 从dev建立特性分支
    (feture-xxx)$: blabla                         # 开发
    (feture-xxx)$: git add xxx
    (feture-xxx)$: git commit -m 'commit comment'
    (dev)$: git merge feture-xxx --no-ff          # 把特性分支合并到dev

#### 修复一个紧急bug

    (master)$: git checkout -b hotfix-xxx         # 从master建立hotfix分支
    (hotfix-xxx)$: blabla                         # 开发
    (hotfix-xxx)$: git add xxx
    (hotfix-xxx)$: git commit -m 'commit comment'
    (master)$: git merge hotfix-xxx --no-ff       # 把hotfix分支合并到master，并上线到生产环境
    (master)$: git merge hotfix-xxx --no-ff       # 把hotfix分支合并到dev，同步代码

#### 测试环境测试代码

    (testing)$: git merge dev --no-ff             # 把dev分支合并到testing，然后在测试环境拉取并测试，配置管理人员操作

#### 生产环境大版本上线

    (master)$: git merge testing --no-ff          # 把testing测试好的代码合并到master，运维人员操作
    (master)$: git tag -a v0.1 -m '部署包版本名'  #给大版本命名，打Tag
