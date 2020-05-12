### 常用脚本

```
npm run build
npm i --save uuid
npm run build

npm i -D webpack-dev-server
npm run dev
```

### 功能

- 打开 Workflow
- 保存 Workflow
- 拖动 Widget 到 WorkSpace
- 双击 Node 打开属性设置
- 运行 Workspace


### 面向对象设计

Widget
- type
- text

Node
- type
- left
- top
- text

Toolbox
- init

Workflow
- load
- save
- run

### todo

- 获取 node 列表及连线
- 添加连接线
- 删除连接线
- 设置节点属性
- Perimeter 锚点不能删除
- 禁止自我连接
- 删除节点
- 节点重命名
- npm 安装指定版本 bootstrap