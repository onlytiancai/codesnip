# 前台

## Web 发展历史

Web设计初衷是一个静态信息资源发布媒介，通过超文本标记语言（HTML）描述信息资源，通过统一资源标识符（URI）定位信息资源，通过超文本转移协议（HTTP）请求信息资源。

用通俗的一点的话来说，客户端（一般为浏览器）通过URL找到网站(如www.google.com)，发出HTTP请求，服务器收到请求后返回HTML页面。

阶段

- Web 诞生
- 动态内容出现：CGI
- Web 编程脚本语言：PHP / ASP / JSP
- 分布式企业计算平台：J2EE / .NET
- 框架横飞的年代：MVC，ORM
- 回归 Web 本质：REST
- 浏览器的魔术：Ajax
- 前端MVC：Angular / Backbone
- Javascript 在服务端的逆袭：NodeJs

时间线

1. 1991年8月6日，Tim Berners Lee在alt.hypertext新闻组贴出了一份关于World Wide Web的简单摘要，标志了Web页面在Internet上的首次登场。
2. Berners Lee在1993年建立了万维网联盟（World Wide Web Consortium，W3C），负责Web相关标准的制定。
3. 1993年CGI（Common Gateway Interface）出现了，Web上的动态信息服务开始蓬勃兴起。
4. 于是1994年的时候，PHP诞生了，PHP可以把程序（动态内容）嵌入到HTML（模版）中去执行，不仅能更好的组织Web应用的内容，而且执行效率比CGI还更高。
5. 1995年NetScape公司设计的JavaScript被用作浏览器上运行脚本语言为网页增加动态性。
6. 之后96年出现的ASP和98年出现的JSP本质上也都可以看成是一种支持某种脚本语言编程（分别是VB和Java）的模版引擎。
7. 96年W3C发布了CSS1.0规范。CSS允许开发者用外联的样式表来取代难以维护的内嵌样式，而不需要逐个去修改HTML元素，这让HTML页面更加容易创建和维护。
8. Web开始广泛用于构建大型应用时，在分布式、安全性、事务性等方面的要求催生了J2EE(现在已更名为Java EE)平台在1999年的诞生，
9. 2000年随之而来的.net平台，其ASP.net构件化的Web开发方式以及Visual Stidio.net开发环境的强大支持，大大降低了开发企业应用的复杂度。
10. 2001年出现的Hibernate就是其中的佼佼者，已经成为Java持久层的规范JPA的主要参考和实现。
11. 比如2003年出现的Java开发框架Spring
12. 2004年出现的Struts就是当时非常流行的Java Web开发的MVC框架。
13. 2005年出现的AJAX这个概念使得JavaScript再次大放异彩。
14. 2004年出现的Ruby开发框架Rails
15. 2005出现的Python开发框架Django

参考链接

- [Web开发技术发展历史](https://www.tianmaying.com/tutorial/web-history)
- [Web开发技术的演变](http://blog.jobbole.com/45170/)


## 切图

- 取色：可以用 FSCapture 来取
- 字号大小：PS里一般用 pt 单位，要转换成网页上的px
- 字体设置：  font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
- 测量间距
	- "编辑"-"首选项"-"单位与标尺", 把单位设置为像素 px
	- "视图"-"标尺"（Ctrl+R） 把标尺显示出来
	- 按住 Shift 键盘，从标尺拖出参考线，会自动对齐物体的边缘
	- 工具栏中选择标尺工具，按住 Shift 键（会自动垂直或水平），先点击第一条辅助线，再拖动到第二条辅助线，然后看信息栏里的 W 和 H，表示宽和高的信息
	- 量完尺寸后可以"视图"-"清除参考线"
- 矢量 Logo 导出：
	- 通过图层的隐藏显示，找到 Logo 对应的矢量智能对象，
	- 右键点击图层，转换为智能对象	
	- 按住 Ctrl 单击智能对象图层，再 Ctrl + C 复制选中的内容
	- Ctrl + N 新建文件，图像宽高会根据复制内容大小自动选定，背景内容选择 "透明" 
	- Ctrl + V 粘贴内容，"文件"-"存储为 Web 所用格式"，格式选择 PNG-24, 存储即可，如选 PNG-8 可能会有毛刺
- 图片导出：
	- 打开移动工具，选项里把"自动选择"的勾打上，
	- 点击要切割的图片，图层面板一般会自动选择图片所在的图层，即使定位不到具体图层，也能定位到图层组
	- 如果该图片是组合图形，或者有图层效果，或者有一个背景图一个遮罩，则按住 Ctrl 选中多个图层，右键合并图层
	- 按住 Ctrl 单击图层，Ctrl + C， Ctrl + N, Ctrl + V， Ctrl + Shif + Alt + S
	

字体大小的设置单位，常用的有2种：px、pt。这两个有什么区别呢？
先搞清基本概念：px就是表示pixel，像素，是屏幕上显示数据的最基本的点；
pt就是point，是印刷行业常用单位，等于1/72英寸。

合并拷贝，背景导出


参考链接：

- [PT与PX区别](https://www.douban.com/note/155032221/)
- [ps标尺和参考线知识点及快捷键](http://www.ittribalwo.com/article/1625.html)


## 经典页面

- [Alloy Timer](http://alloyteam.github.io/AlloyTimer/)
- http://www.raiseai.com/

## 常见任务
垂直居中

```
parentElement{
	position:relative;
}

childElement{
	position: absolute;
	top: 50%;
	transform: translateY(-50%);
}
```

### 左右等高

- [4 Methods For Creating Equal Height Columns In CSS](http://vanseodesign.com/css/equal-height-columns/)
- [Fluid Width Equal Height Columns](https://css-tricks.com/fluid-width-equal-height-columns/)
- [CSS布局——左定宽度右自适应宽度并且等高布局](https://www.w3cplus.com/css/two-cloumn-width-one-fixed-width-one-fluid-width)
- [Equal Height Column Layouts with Borders and Negative](https://www.smashingmagazine.com/2010/11/equal-height-columns-using-borders-and-negative-margins-with-css/)


### 背景图全屏

```
html,body{
    width:100%;
    height:100%
}

body{
	width: 100%;
  	height:auto;
	background:#343434 url("../../assets/img/bg.jpg") no-repeat;  
	background-size: cover;
}
```

## 参考链接

- [CSS Stacking Context里那些鲜为人知的坑](http://blog.angular.in/css-stacking-contextli-na-xie-xian-wei-ren-zhi-de-keng/)
- [HTML 和 Body 在 CSS 中的区别](https://csspod.com/html-vs-body-in-css/)
- [等宽列背后的表格布局算法](https://csspod.com/table-width-algorithms/)
- [Appendix D. Default style sheet for HTML 4](https://www.w3.org/TR/CSS2/sample.html)
- [学习CSS布局](https://www.w3cplus.com/css/learn-css-layout.html)
- [w3school HTML 系列教程](http://www.w3school.com.cn/h.asp)
- [CSS参考手册](http://www.css88.com/book/css/)
- [10 个最常见的 JavaScript 错误（以及如何避免它们）](http://www.css88.com/archives/9184)

## todo

- rem



# 后台


## 算法

[编程之法：面试和算法心得](https://github.com/julycoding/The-Art-Of-Programming-By-July)



InfraredCounterParser InfraredCounterHandler InfraredCounterSender 等字符串，取出最后一个单词，如Parser, Handler, Sender 等

中英文混排，如何在英文和数字两边增加空格

# 数据库
# 网络


# 软件工程
## 代码风格

[编程之法：面试和算法心得 Code Style](https://github.com/julycoding/The-Art-Of-Programming-By-July)

## UML

### 类图

- 泛化: 泛化(generalization)：表示is-a的关系，是对象之间耦合度最大的一种关系，子类继承父类的所有细节。直接使用语言中的继承表达。在类图中使用带三角箭头的实线表示，箭头从子类指向父类。
- 实现（Realization）:在类图中就是接口和实现的关系。在类图中使用带三角箭头的虚线表示，箭头从实现类指向接口。
- 依赖(Dependency)：对象之间最弱的一种关联方式，是临时性的关联。代码中一般指由局部变量、函数参数、返回值建立的对于其他对象的调用关系。一个类调用被依赖类中的某些方法而得以完成这个类的一些职责。在类图使用带箭头的虚线表示，箭头从使用类指向被依赖的类。
- 关联(Association) : 对象之间一种引用关系，比如客户类与订单类之间的关系。这种关系通常使用类的属性表达。关联又分为一般关联、聚合关联与组合关联。在类图使用带箭头的实线表示，箭头从使用类指向被关联的类。可以是单向和双向。
- 聚合(Aggregation) : 表示has-a的关系，是一种不稳定的包含关系。较强于一般关联,有整体与局部的关系,并且没有了整体,局部也可单独存在。如公司和员工的关系，公司包含员工，但如果公司倒闭，员工依然可以换公司。在类图使用空心的菱形表示，菱形从局部指向整体。
- 组合(Composition) : 表示contains-a的关系，是一种强烈的包含关系。组合类负责被组合类的生命周期。是一种更强的聚合关系。部分不能脱离整体存在。如公司和部门的关系，没有了公司，部门也不能存在了；调查问卷中问题和选项的关系；订单和订单选项的关系。在类图使用实心的菱形表示，菱形从局部指向整体。

参考

- [UML类图与类的关系详解](http://www.uml.org.cn/oobject/201104212.asp)

## 需求工程



### 需求分类

软件需求包括3个不同的层次――业务需求、用户需求和功能需求。除此之外，每个系统还有各种非功能需求。 

- **业务需求**（Business requirement）表示组织或客户高层次的目标。业务需求描述了组织为什么要开发一个系统，即组织希望达到的目标。
- **用户需求**（user requirement）描述的是用户的目标，或用户要求系统必须能完成的任务,用户能使用系统来做些什么。用例、场景描述和事件都是表达用户需求的有效途径。 
- **功能需求**（functional requirement）规定开发人员必须在产品中实现的软件功能，用户利用这些功能来完成任务，满足业务需求。如“系统应该发送电子邮件来通知用户已接受其预定”。
- **系统需求**（system requirement）用于描述包含多个子系统的产品（即系统）的顶级需求。系统可以只包含软件系统，也可以既包含软件又包含硬件子系统
- **非功能需求** 为满足用户业务需求而必须具有且除功能需求以外的特性
  - 安全性
  - 可靠性
  - 易用性
  - 可维护性
  - 可移植性

### 需求开发



需求开发活动包括以下几个方面：

1. 确定用户分类
2. 获取每类用户的需求
3. 了解实际用户任务和目标
4. 分析源于用户的信息以获取用户任务需求、功能需求、业务规则、质量属性、建议解决方法和附加信息
5. 将系统级的需求分为几个子系统
6. 了解相关质量属性的重要性
7. 商讨实施优先级的划分
8. 编写成规格说明和模型。
9. 评审需求规格说明


### 用例描述

用例图描述了参与者要求系统能“做什么”，但是缺乏描述系统该“怎么做”的细节。一般情况下，每个用例应具有一个用例描述。

#### 用例描述说明



- **用例名称**：用例名称应该表明用户的意图或用例的用途，例如：借阅图书、归还图书、预定图书等。
- **用例编号**： UC 0001
- **简要说明**：对用例进行简要说明，描述该用例的作用，说明应当简明扼要。
- **参与者**：与此用例相关的参与者列表
- **前置条件**：执行用例之前系统必须满足的条件，例如：当学生借阅图书时，借阅图书用例需要获取学生的借阅证信息，如果学生使用了一个已经被注销的借阅证，那么借阅图书用例就不能执行。
- **后置条件**：后置条件将在用例成功完成以后得到满足，它提供了系统的部分描述。例如：当学生借阅图书成功后，借阅图书用例应该提供该学生的所有借阅信息。
- **基本操作流程**：指参与者在用例中所遵循的主逻辑路径。例如，借阅图书用例的基本操作流程如下：
  - (1) 图书管理员输入借阅证信息
  - (2) 系统检查读者是否有超期的借阅信息
  - (3) 系统检查读者的借书数量是否已经达到借书限额
  - (4) 图书管理员输入要借阅的图书信息
  - (5) 系统将读者的借阅信息保存到数据库中
- **可选操作流程**：指参与者在用例中所遵循的次逻辑路径，通常是指出现异常或发生错误的情况下所遵循的路径。
- **涉及数据**：填写该用例涉及的相关信息，如图书名字，价格，ISBN号，出版日期等

#### 完整用例示例



| 主执行者      | 请求者                                      |
| --------- | ---------------------------------------- |
| 语境中的目标    | 请求者通过系统买东西，并得到说买的东西。不包括付款方面的内容。          |
| 范围        | 业务——整个购买机制，包括电子的和非电子的，正如人们在公司中说见到的一样。    |
| 层次        | 概要                                       |
| 项目相关人员和利益 | 请求者：希望得到她订购的东西，并且操作要简单。公司：希望控制花费，但允许必要的购买。供货商：希望得到任何已发货物的货款。 |
| 前置条件      | 无                                        |
| 最小保证      | 每一个发出的订单都已经获得有效认证者的许可。订单具有可跟踪性，以便公司只对收到的有效货物开账单。 |
| 成功保证      | 求者得到货物，修改预算，记入借方。                        |
| 触发事件      | 请求者决定买东西。                                |
| 主成功场景     | 1. 请求者：发起一个请求。2.   批准者：检查预算中的资金，检查货物的价格，完成提交请求。3. 买者：检查仓库的存货，找出最好的供货商。4. 认证者：验证批准者的签名。5. 买者：完成订购请求，向供货商发出PO（订单）。6. 供货商：把货物发送给接收者，得到发货收据（这一点超出了本系统的设计范围）。7. 接收者：记录发货情况；向请求者发送货物。8. 请求者：设置请求已被满足标志。 |
| 扩展        | 1a）请求者不知道供货商和货物价格：不填写这些内容，然后继续。1b）在收到货物之前的任意时刻，请求者都可以修改或取消请求：如果取消，则把这个请求从执行处理中取消。（从系统中删除吗？）如果降低价格，则不影响其处理过程。如果提高价格，则把请求送回批准者。2a）批准者不知道供货商或货物价格：不填写这些内容，留待买者填写或返回。2b）批准者不是请求者的经理：只是批准者签名仍然可行。2c）批准者拒绝申请：送回给请求者，要其修改或删除。3a）买者在仓库中找到货物：将存货先发出，并从申请者要求的总购买者中减去已经发出的这部分货物量，然后继续。3b）买者填写在前面活动中没有填写的供货商和价格信息：请求重新发回给批准者。4a）认证者拒绝批准者：发回请求者，并将此请求从执行处理中取消。5a）请求涉及到多个供货商：买者创建多个PO5b）买者将多个请求合并：相同的过程，但是用被合并的请求标记PO6a）供货商没有按时发货：系统发出没有发货警告。7a）部分发货：接收者在PO上做部分发货标记，然后继续。7b）多个请求PO的部分发货：接收者给每个请求分配货物数量，然后继续。8a）货物不对或质量不合格：请求者拒绝接收所发送的货物。8b）请求者已经离开公司：买者同请求者的经理进行核实，或者重新指派申请者，或者返还货物并取消请求。 |
| 技术和数据变动列表 | 无                                        |
| 优先级       | 多种                                       |
| 发行版本      | 几个                                       |
| 响应时间      | 多样                                       |
| 使用频率      | 3/天                                      |
| 主执行者的渠道   | 网络浏览器、邮件系统或类似系统                          |
| 次要执行者     | 供货商                                      |
| 次要执行者的渠道  | 传真、电话或汽车                                 |
| 未解决的问题    | 什么时候从系统中删除被取消的请求？要取消一个请求需要那些权限？谁能修改一个请求的内容？请求中需要保留哪些修改历史记录？当请求者拒绝已经发送的货物时，会发生什么情况？申请和订货在运作上有什么不同？ 订购如何参考和使用内部存货？ |

#### 用例执行步骤的10大准则

（1）使用简单的语法；
句子结构应该非常简单：主语……谓语动词……直接宾语……前置短语
例如                  系统……从帐户余额中扣除……一定数量……

（2）明确地写出“谁控制球”；
作者举了踢足球的场景的例子，说明了不管步骤的执行者如何变化，都要遵循（1）描述的格式。

（3）从俯视的角度来编写用例；
从用户的角度来写用例，而不是从系统内部来描述系统

~~（4）显示过程向前推移；~~

（5）显示执行者的意图而不是动作；
通过操纵系统的用户界面来描述用户的动作，这是在编写用例时常见的一种严重错误，它使得编写的目标处于一个很低的层次。我把它叫做“界面细节描述（interface detail description）”。在需求文档中，我们只关心界面所要达到的意图，总结在执行者之间传递的信息。可将这些低层次的步骤合并成一个步骤。

~~（6）包含“合理”的活动集；~~

（7）“确认”而不是“检查是否”
这是一个经常犯的错误，写用例不是写程序流程，不需要用选择语法，需要选择的时候，在扩展场景里体现

（8）可选择地提及时间限制；

（9）习惯用语：“用户让系统A与系统B交互”；
要分开来写，用户与系统A怎么怎么样，然后系统A和系统B怎么怎么样，这样用户才能看的懂。

（10）习惯用语：“循环执行步骤X到Y，直到条件满足”；
同（7），但如果需要重复的话，可直接在重复的步骤的前面和后面说明即可。

总之，这10大原则，目的就是为了让用例成为用户和开发人员沟通的桥梁，所以语言要简单易懂，而且要逻辑清晰。

### 参考链接



- [需求入门： 需求工程＝需求开发＋需求管理](http://www.uml.org.cn/RequirementProject/201005285.asp)
- [软件需求规格说明书模板](https://jingyan.baidu.com/article/6dad5075eae10da123e36e80.html)
- [描述用例](http://blog.csdn.net/wlanye/article/details/7445676)
- [软件需求3个层次――业务需求、用户需求和功能需求](https://www.cnblogs.com/litian/articles/2047981.html)
- [非功能性六大点](https://jingyan.baidu.com/article/90bc8fc80960f1f653640ce0.html)
- [《编写有效用例》学习笔记](http://lib.csdn.net/article/softwaretest/24322)





## 文档

### 文档列表

1. 可行性分析报告
2. 项目开发计划
3. **软件需求说明书**
4. **概要设计说明书**
5. 详细设计说明书
6. **用户操作手册、运维部署文档**
7. 测试计划
8. 测试分析报告
9. 开发进度月报
10. 项目开发总结报告
11. 软件维护手册
12. 软件问题报告
13. 软件修改报告 

### 需求分析文档

内容

- 项目背景
- 使用角色
- 功能需求
  - 功能划分
  - 用例图
  - 用例描述
- 性能需求
  - 用户数评估
  - 响应时间要求
  - 可靠性需求
- 用户界面

### 概要设计文档

对大部分的公司来说，概要设计文档是唯一的设计文档，对后面的开发、测试、实施、维护工作起到关键性的影响。

内容

- 功能介绍
  - 用例图
- 整体架构
  - 层次结构
  - 模块划分
  - 模块间调用关系
  - 包图、组件图
- 接口设计
  - 对外接口草案
  - 模块间接口草案
- 模块设计
  - 职责描述
  - 类图
  - 算法描述（如有）
- 数据流设计
  - 流程图
  - 序列图
  - 状态图
- 数据库概要设计
  - 表设计
  - 核心语句
- 主要界面
- 部署结构
  - 部署图
- 非功能需求
  - 性能需求
  - 安全需求
  - 扩展性需求
- 编码规范
  - 代码风格
  - 数据库命名规范
  - 接口规范

### 详细设计文档

同概要设计文档，只是需要更详细，比如

- 类图要精确到字段，方法的类型，
- 数据库设计要精确到字段类型，索引设计
- 核心算法要画出序列图及伪代码

### 运维部署文档

内容

- 硬件需求
- 软件需求
- 外部依赖
- 部署步骤
- 配置说明
- 维护命令：启动、停止、重启
- 监控指标
- 如何升级
- 常见问题处理
- 数据备份

### 参考链接

- [软件开发文档范例](http://zz563143188.iteye.com/blog/1835305)
- [概要设计文档编写规范](http://blog.csdn.net/nengyu/article/details/3758312)

# 工具

## 其它

- [f.lux - 全天候保护眼睛健康软件！自动调整屏幕色温减少蓝光防疲劳，长时间玩电脑必备！](https://www.iplaysoft.com/flux.html)
- [MarkDown 写 ppt](https://yhatt.github.io/marp/)
- [在线根据 markdown 生成 ppt](http://www.vmfor.com/ppt/index.html)

## 编辑器



### 功能需求



- 行号显示
- 语法高亮
- 自动识别编码
- 自动缩进
- 智能提示
- 列编辑
- 代码片段
- 代码折叠
- 新建文件模板
- 多标签
- 会话管理
- 分屏
- 书签
- 多次光标点跳转

### VS Code

快捷键

- 注释切换：ctrl +／
- 格式化代码： Alt + Shift + F
- 自动换行：Alt + Z
- 选区移动：Alt + ↑/↓
- 多选修改：Ctrl + D

### Eclipse

- 格式化代码：Ctrl + Shift + F

参考链接：

- [为什么我选择使用 VS Code进行前端开发?](https://zhuanlan.zhihu.com/p/28631442)
- [TextMate代码片段语法](https://manual.macromates.com/en/snippets)

# 物联网

## 参考链接

[关于RS232 和 RS485 的区别](http://blog.csdn.net/foreverhuylee/article/details/23375079)

## 硬件技术


### 串口

串行接口（Serial port）又称“串口”，主要用于串行式逐位数据传输。

串行接口按电气标准及协议来分，包括RS-232-C、RS-422、RS485、USB等。

- RS-232-C、RS-422与RS-485标准只对接口的电气特性做出规定，不涉及接插件、电缆或协议。
- USB是近几年发展起来的新型接口标准，主要应用于高速数据传输领域。

**RS-232-C ：也称标准串口**，是目前最常用的一种串行通讯接口。电脑一般有两个串行口：COM1和COM2，9针D形接口通常在计算机后面能看到。

**RS-485** ：为扩展应用范围，增加了多点、双向通信能力，即允许多个发送器连接到同一条总线上，同时增加了发送器的驱动能力和冲突保护特性，扩展了总线共模范围。

**Universal Serial Bus（通用串行总线） ：简称USB，** USB接口是电脑主板上的一种四针接口，其中中间两个针传输数据，两边两个针给外设供电。USB接口速度快、连接简单、不需要外接电源，传输速度12Mbps，新的USB 2.0可达480Mbps；

**RJ-45接口** ：是以太网最为常用的接口， 8个位置（8针）的模块化插孔或者插头



串口属性：

-  PortName 串口名
-  BaudRate 获取或设置串行波特率bit/s    默认值9600
-  DataBits 获取或设置每个字节的标准数据位长度    默认值8
-  StopBits 获取或设置每个字节的标准停止位数    默认值One
-  Parity 获取或设置奇偶校验检查协议    默认值None



参考

- [C#中的串口通信](http://www.cnblogs.com/51net/p/6050840.html)

### IC卡，ID卡，M1卡，射频卡的区别

IC卡全称集成电路卡（Integrated Circuit Card），又称智能卡（Smart Card）。可读写，容量大，有加密功能，数据记录可靠，使用更方便，如一卡通系统、消费系统等

ID卡全称身份识别卡（Identification Card），是一种不可写入的感应卡，含固定的编号，

[门禁卡是选择IC卡好还是ID卡好](https://jingyan.baidu.com/article/54b6b9c0d8056f2d593b474c.html)




### RFID

从概念上来讲，RFID类似于[条码扫描](https://baike.baidu.com/item/%E6%9D%A1%E7%A0%81%E6%89%AB%E6%8F%8F)，对于条码技术而言，它是将已编码的条形码附着于目标物并使用专用的扫描读写器利用光信号将信息由条形磁传送到扫描读写器；而RFID则使用专用的RFID读写器及专门的可附着于目标物的RFID标签，利用频率信号将信息由RFID标签传送至RFID读写器。

[射频识别系统](https://baike.baidu.com/item/%E5%B0%84%E9%A2%91%E8%AF%86%E5%88%AB%E7%B3%BB%E7%BB%9F)最重要的优点是非接触识别，它能穿透雪、雾、冰、[涂料](https://baike.baidu.com/item/%E6%B6%82%E6%96%99)、[尘垢](https://baike.baidu.com/item/%E5%B0%98%E5%9E%A2)和条形码无法使用的恶劣环境阅读标签，并且阅读速度极快，大多数情况下不到100毫秒。

一维条形码的容量是50Bytes，二维条形码最大的容量可储存2至3000字符，RFID最大的容量则有数MegaBytes.

现今的条形码印刷上去之后就无法更改，RFID标签则可以重复地新增、修改、删除RFID卷标内储存的数据，方便信息的更新。

### NFC

NFC近场通信技术是由非接触式[射频识别](https://baike.baidu.com/item/%E5%B0%84%E9%A2%91%E8%AF%86%E5%88%AB)（[RFID](https://baike.baidu.com/item/RFID)）及互联互通技术整合演变而来，在单一芯片上结合感应式[读卡器](https://baike.baidu.com/item/%E8%AF%BB%E5%8D%A1%E5%99%A8)、感应式卡片和点对点的功能，能在短距离内与兼容设备进行识别和[数据交换](https://baike.baidu.com/item/%E6%95%B0%E6%8D%AE%E4%BA%A4%E6%8D%A2)。

NFC手机内置NFC芯片，比原先仅作为标签使用的RFID更增加了数据双向传送的功能，这个进步使得其更加适合用于电子货币支付的；

NFC传输范围比RFID小，RFID的传输范围可以达到几米、甚至几十米，但由于NFC采取了独特的信号衰减技术，相对于RFID来说NFC具有距离近、带宽高、能耗低等特点。

应用方向不同。NFC看更多的是针对于消费类电子设备相互通讯，有源RFID则更擅长在长距离识别。

NFC的短距离通信特性正是其优点，由于耗电量低、一次只和一台机器链接，拥有较高的保密性与安全性，NFC有利于信用卡交易时避免被盗用。

### PLC

### 继电器

继电器是具有隔离功能的自动开关元件，广泛应用于遥控、遥测、通讯、自动控制、机电一体化及电力电子设备中，是最重要的控制元件之一。

继电器一般都有能反映一定输入变量（如电流、电压、功率、阻抗、频率、温度、压力、速度、光等）的感应机构（输入部分）；有能对被控电路实现“通”、“断”控制的执行机构（输出部分）；在继电器的输入部分和输出部分之间，还有对输入量进行耦合隔离，功能处理和对输出部分进行驱动的中间机构（驱动部分）。

### 二进制协议

二进制常用操作

```

	int value = 170;
	void print(int x) => Console.WriteLine(Convert.ToString(x, 2).PadLeft(8, '0'));
	int make_mask(int x) => 1 << x;
	int set(int x, int i) => x | make_mask(i);
	int unset(int x, int i) => x & ~make_mask(i);
	bool isset(int x, int i) => (x & make_mask(i)) != 0;

	//170的二进制显示：                    10101010
	print(value);

	// 右数第 5 位置 1 ：                  10111010
	print(set(value, 4));

	// 右数第 4 位置 0：                   10100010
	print(unset(value, 3));

	// 右数第 4 位是否为1:                 true
	Console.WriteLine(isset(value, 3));

	//右数第 3 位是否为1:                  false
	Console.WriteLine(isset(value, 2));
```

### 其它

TTL电平信号之所以被广泛使用，原因是：通常我们采用二进制来表示数据。而且规定，+5V等价于逻辑“1”，0V等价于逻辑“0”。这样的数据通信及电平规定方式，被称做TTL（晶体管-晶体管逻辑电平）信号系统。这是计算机处理器控制的设备内部各部分之间通信的标准技术。

GND是电线接地端的简写。代表地线或0线。这个地并不是真正意义上的地，是出于应用而假设的一个地，对于电源来说，它就是一个电源的负极。

VCC：电源电压(双极器件);电源电压(74系列数字电路);

RXD 为接收数据的引脚，TXD 为发送数据的引脚。

DTE提供或接收数据，连接到调制解调器上的计算机就是一种DTE。DTE提供或接收数据，连接到网络中的用户端机器，主要是计算机和终端设备。

在网络端的连接设备称为DCE（Data-Communication Equipment）。DTE与进行信令处理的DCE相连。
DTE通过DCE设备，例如，调制解调器，连接到数据网络。

RS232标准中的RTS与CTS：即请求发送/清除发送，用于半双工时的收发切换，属于辅助流控信号。半双工的意思是说，发的时候不收，收的时候不发。那么怎么区分收发呢？缺省时是DCE向DTE发送数据，当DTE决定向DCE发数据时，先有效RTS，表示DTE希望向DCE发送。一般DCE不能马上转换收发状态，DTE就通过监测CTS是否有效来判断可否发送，这样避免了DTE在DCE未准备好时发送所导致的数据丢失。

差分输入的是将两个输入端的差值作为信号，这样可以免去一些误差，比如你输入一个1V的信号电源有偏差，比实际输入要大0.1.就可以用差分输入1V和2V一减就把两端共有的那0.1误差剪掉了。单端输入无法去除这类误差。

在某些系统里，系统'地'被用作电压基准点。当'地'当作电压测量基准时，这种信号规划被称之为单端的。

差分信号的第一个好处是，因为你在控制'基准'电压，所以能够很容易地识别小信号。
差分信号的第二个主要好处是，它对外部电磁干扰（EMI）是高度免疫的。
差分信号提供的第三个好处是，在一个单电源系统，能够从容精确地处理'双极'信号。

上拉就是将不确定的信号通过一个电阻钳位在高电平，电阻同时起限流作用。下拉同理，也是将不确定的信号通过一个电阻钳位在低电平。

# 企业信息化

大多数OA产品功能集中在信息共享、行政办公领域，一些主流OA系统虽然引入了工作流，但相对比较封闭，开放性和扩展性不够。

BPM是一个开放性平台，不仅能实现所有OA的功能，还能满足企业内部系统之间集成的需求，在BPM引擎驱动下，企业的流程终会形成一个闭环。

ERP

WMS

MES

ESB

SOA

- [智能MES解决方案](http://www.rtdsoft.com/channels/57.html)
- [制造执行系统(MES)选型与实施指南简版](https://wenku.baidu.com/view/052b5ef4a32d7375a41780c8.html)
- [OpenMES架构说明书](https://wenku.baidu.com/view/2a98711ec281e53a5802ffc8.html)

## 快速开发框架

- 登录注册: Apache Shiro 
- 组织机构
- 权限管理: Apache Shiro 
- 增删改查
- 后台界面
- 菜单管理
- 工作流：Activity
- 报表：JasperReports

参考

- http://www.jeecg.org/
- [Java通用权限系统管理（Spring+springMVC+ibatis+Angularjs）](http://46aae4d1e2371e4aa769798941cef698.devproxy.yunshipei.com/liaodehong/article/details/53100313)
- [组织机构对象模型设计及实现](http://blog.csdn.net/wangpeng047/article/details/7280800)
- [LigerUI 快速开发UI框架](http://www.ligerui.com/)
- https://github.com/thinkgem/jeesite

jeesite应用实战（数据增删改查），认真读完后10分钟就能开发一个模块
http://blog.csdn.net/qing_gee/article/details/76223064


# 未整理

[Avoid calling Invoke when the control is disposed](https://stackoverflow.com/questions/1874728/avoid-calling-invoke-when-the-control-is-disposed)



http://blog.csdn.net/pkueecser/article/details/50610796 时间序列数据库的秘密



https://github.com/justjavac/ReplaceGoogleCDN Replace Google CDN

https://stackoverflow.com/questions/31572580/how-covert-c-sharp-datetime-to-java-datetimeusing-joda-time
https://pine.fm/LearnToProgram/
http://www.qdaily.com/articles/42060.html
https://zh.wikihow.com/%E5%AD%A6%E4%B9%A0%E7%BC%96%E7%A8%8B

nginx waf

前端系列教程
https://chuanke.baidu.com/s5508922.html
http://growth.phodal.com/

关于PHP程序员解决问题的能力
http://www.cnblogs.com/phpworld/p/8038581.html
ZH奶酪：编程语言入门经典100例【Python版】
http://www.cnblogs.com/CheeseZH/archive/2012/11/05/2755107.html
https://www.coursera.org/learn/python

JavaScript专题之惰性函数 https://segmentfault.com/a/1190000010783034
关于尾递归的问题 https://segmentfault.com/q/1010000002705723

var pi = 3.14;
function area(r) { return 2 * pi * r;}
function iseven(n) {return n % 2 == 0;}

function range(n, m) { console.log(n); (n < m) ? range(n + 1, m) : null;}
function rangef(n, m, f) { f(n);  (n < m) ? rangef(n + 1, m, f) : null;}
function rangesum(n, m, sum) { return n < m ? n + rangesum(n + 1, m, sum) : n + sum; }
function rangesumf(n, m, sum, f) { return n < m ? f(n) + rangesumf(n + 1, m, sum, f) : f(n) + sum; }
function rangecond(n, m, cond) { cond(n) ? console.log(n) : null; (n < m) ? rangecond(n + 1, m, cond) : null;}

7 of the Best Code Playgrounds
https://www.sitepoint.com/7-code-playgrounds/

Beginning Programming For Dummies
https://www.amazon.com/Beginning-Programming-Dummies-Wallace-Wang/dp/0470088702
jQuery File Upload跨域上传
https://www.cnblogs.com/ZHF/p/5057416.html
Javascript知识点：IIFE - 立即调用函数
https://linghucong.js.org/2016/04/25/2016-04-08-Javascript-IIFE/
技术面试需要掌握的基础知识
https://github.com/CyC2018/Interview-Notebook

### 函数组合
function rangei(n) {
	var i = 0;
	return function() {
		var ret = i < n ? i : null;
		i = i + 1;
		return ret;
	};
}

function mapi(i, f) {
	return function () {
		var t = i();
		return t == null ? null : f(t);
	};
}

function filteri(i, c) {
	return function inner() {
		var t = i();
		return t == null ? null : c(t) ? t : inner(); 
	};
}

function reducei(i, f, init_val) {
	var t = i();
	return t == null ? init_val : reducei(i, f, f(init_val, t));
}
	
function iseven(x) {
	return x % 2 == 0; 
}

function sqr(x) {
	return x * x;
}

function add(x, y) {
	return x + y;
} 

//10 以内偶数的平方和
reducei(mapi(filteri(rangei(10), iseven), sqr), add, 0)

Jquery mobile change page
https://stackoverflow.com/questions/9738948/jquery-mobile-change-page
The Truth About Multiple H1 Tags in the HTML5 Era
https://webdesign.tutsplus.com/articles/the-truth-about-multiple-h1-tags-in-the-html5-era--webdesign-16824

How to set up Spark on Windows?
https://stackoverflow.com/questions/25481325/how-to-set-up-spark-on-windows


Git for windows 中文乱码解决方案
https://segmentfault.com/a/1190000000578037

git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
export LESSCHARSET=utf-8

Best way to find if an item is in a JavaScript array? [duplicate]
https://stackoverflow.com/questions/143847/best-way-to-find-if-an-item-is-in-a-javascript-array
后台管理UI的选择
https://www.cnblogs.com/webenh/p/5815732.html

配置 PHP 错误日志

mkdir /data/logs/php
chown apache:apache /data/logs/php

php-fpm.conf：

    [global]
    ; php-fpm pid文件
    pid = /usr/local/php/var/run/php-fpm.pid
    ; php-fpm 错误日志路径
    error_log = /data/logs/php/error.log
    ; php-fpm 记录错误日志等级
    log_level = notice
    [www]
    ; 记录错误到php-fpm的日志中
    ;catch_workers_output = yes
    ; 慢日志
    slowlog = /data/logs/php/www-slow.log
    ; 关闭打印日志
    php_flag[display_errors] = off
    ; 错误日志
    php_admin_value[error_log] = /data/logs/php/www-error.log
    ; 记录错误
    php_admin_flag[log_errors] = on
    ; 内存使用量
    php_admin_value[memory_limit] = 32M

php.ini：

    ; 错误日志
    log_errors = On
    ; 显示错误
    display_errors = Off
    ; 日志路径
    error_log = "/usr/local/lnmp/php/var/log/error_log"
    ; 错误等级
    error_reporting = E_ALL&~E_NOTICE
    
    
nginx 路径匹配测试

    location /test {
        add_header Content-Type text/html;
        return 200 'hello';
    }

nginx php 测试

    chmod o+x /root
    chmod o+x /root/helloworld
    chmod o+x /root/helloworld/phptest
    
    location ~ /test$ {
        fastcgi_pass   127.0.0.1:9000;
        include        fastcgi_params;
        fastcgi_param SCRIPT_FILENAME /root/helloworld/phptest/test.php;
    }
    
nginx 日志格式

    log_format  main  '$time_iso8601 $status $request_time $upstream_response_time $remote_addr '
                      '"$request" $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

nginx php 子目录
    location /turtle {
        # 一定要 / 结尾
        alias /data/release/helloworld-turtle/;

        location ~ \.php$ {
            try_files $uri = 404;
            fastcgi_pass   127.0.0.1:9000;
            fastcgi_index  index.php;
            include        fastcgi_params;
            # 需要用 $request_filename 而不是  $document_root$fastcgi_script_name;
            fastcgi_param SCRIPT_FILENAME $request_filename;
            fastcgi_param SERVER_NAME $http_host;
            fastcgi_param CI_ENV production;
            fastcgi_ignore_client_abort on;
            fastcgi_connect_timeout 600s;
            fastcgi_send_timeout 600s;
            fastcgi_read_timeout 600s;
        }
    }

                      
日志滚动

    /var/log/nginx/*.log
    /data/log/nginx/*.log
    /data/logs/helloworld/*.php
    /data/logs/php/*.log
    /var/log/php-fpm/*log
    {
        daily
        rotate 30
        missingok
        notifempty
        compress
        sharedscripts

        postrotate
            /bin/kill -USR1 `cat /run/nginx.pid 2>/dev/null` 2>/dev/null || true
                /bin/kill -SIGUSR1 `cat /var/run/php-fpm/php-fpm.pid 2>/dev/null` 2>/dev/null || true
        endscript
    }
                      
    
Linux日志文件总管——logrotate
https://linux.cn/article-4126-1.html
Linux中find常见用法示例
http://www.cnblogs.com/wanqieddy/archive/2011/06/09/2076785.html

找到 /data/logs 目录下所有 15 天之前修改过的 .php 和 .log 文件删除掉，删除前加确认，去掉确认的话，把 -ok 改成 -exec
find /data/logs/ -type f \( -name '*.php' -o -name '*.log' \)  -mtime +15 -print -ok rm  {} \;    



防止文件误删：http://www.cnblogs.com/lihaozy/archive/2012/08/17/2643784.html

    mkdir -p ~/.trash
    alias rm=trash  
    alias r=trash  
    alias rl='ls ~/.trash'
    alias ur=undelfile

    undelfile()
    {
      mv -i ~/.trash/$@ ./
    }

    trash()
    {
      mv $@ ~/.trash/
    }
    
    cleartrash()
    {
        read -p "clear sure?[n]" confirm
        [ $confirm == 'y' ] || [ $confirm == 'Y' ]  && /usr/bin/rm -rf ~/.trash/*
    }
    
[译] Node.js 8: util.promisify()
https://segmentfault.com/a/1190000009743481
Node.js Async Best Practices & Avoiding the Callback Hell
https://blog.risingstack.com/node-js-async-best-practices-avoiding-callback-hell-node-js-at-scale/
Nodejs 中使用 Async/Await
https://www.jianshu.com/p/0837dde8dcd5

promisify + async 

    (async () => {
      const fs = require('fs');
      const util = require('util');

      const readFile = util.promisify(fs.readFile);

      const txt = await readFile('./notes.txt');
      console.log(txt);
    })();
    
好吧，CSS3 3D transform变换，不过如此！    
http://www.zhangxinxu.com/wordpress/2012/09/css3-3d-transform-perspective-animate-transition/    
客栈说书：CSS遮罩CSS3 mask/masks详细介绍
http://www.zhangxinxu.com/wordpress/2017/11/css-css3-mask-masks/
新手应该如何学习 PHP 语言？
https://www.zhihu.com/question/20003635/answer/338733500
目前最详细的PHP培训课程安排
http://baijiahao.baidu.com/s?id=1563138980186749&wfr=spider&for=pc

CSS3 box-pack 属性
http://www.w3school.com.cn/cssref/pr_box-pack.asp
设计一个灵活的、可维护CSS和SVG饼图SVG
https://www.jianshu.com/p/f6526355de54
SVG中stroke-dasharray及stroke-dashoffset属性
https://blog.csdn.net/u014291497/article/details/78409350   
纯CSS实现帅气的SVG路径描边动画效果
http://www.zhangxinxu.com/wordpress/2014/04/animateion-line-drawing-svg-path-%E5%8A%A8%E7%94%BB-%E8%B7%AF%E5%BE%84/

https://secbone.com/
https://github.com/Secbone

# 快速学习一门语言
- 打字练习：
    - 数字，字母，下划线
    - 标点符号：+-*/\%(){};"',.!<>[]?^|
- 输出
    - 打印字符串
    - 打印数字
    - 混合打印
    - 输出复杂信息
    - 格式化输出    
- 字符串连接
- 数学运算: + - * / %
- 关系运算: > < == != >= <=
- 逻辑运算: && || !
- 变量和赋值: =
- 条件语句和关系操作符：两个数谁最大
- 多重条件语句和逻辑操作符：三个数谁最大
- 循环语句：打印 5 次 Hello
- for 循环：打印 10 以内的偶数
- 数组基本操作
    - 打印数组
    - 获取数组长度
    - 获取指定索引元素
    - 修改指定索引元素的值
    - 数组末尾增加元素
    - 某元素是否存在
    - 遍历数组
    - 删除指定索引元素
    - 重排数组
    - 任意位置插入元素
    - 连接成字符串
    - 练习
        - 找出一个数组中最大的数
        - 找出一个数组中最大的奇数
        - 求出数组中所有奇数的和
- 字符串基本操作
    - 字符串长度
    - 字符串连接
    - 转换大小写
    - 去掉首尾指定字符
    - 用某字符在首尾填充    
    - 分割成数组
    - 比较两个字符串
    - 获取某子串的位置
    - 是否存在某个子串
    - 替换某个子串
    - 获取子串
    - 是否以某子串开头或结尾
    - 练习
        - 判断某字符是否为大写
        - 把字符串中的每个单词首字母大写
        - 按大写字母分割字符串
- 字典基本操作
    - 根据键获取值
    - 添加键值对
    - 修改键值对
    - 删除指定键
    - 某键是否存在
    - 遍历字典
- 时间操作
    - 设置时区
    - 获取当前时间的时间戳
    - 获取当前时间格式化字符串
- 文件操作：打开并逐行读取文件
- 类型判断
    - 是否为数字
    - 是否为整型
    - 是否为浮点数
    - 是否为字符串
    - 是否为 null
    - 是否为 数组
    - 是否为空
- 类型转换
    - 数字转字符串
    - 字符串转数字
- 函数
    - 函数定义和使用
    - 匿名函数和闭包
    - 返回函数的函数
    - 参数为函数的函数
    - 函数的递归调用
- 面向对象
    - 类
    - 方法
    - 字段
    - 静态字段
    - 子类

数组:
- get set remove insert length 
- push pop shift unshit slice indexof
- startswitch endswitch contains
- map filter reduce
- removeCond groupby sortby countby

字典
- add set get remove
树

输入输出文件

Android Studio:

- 下载带 Sdk 版本
- 修改字体
- 修改编码
- 自动提示快捷键修改
- 去掉拼写检查
- 设置 git 目录
- 禁用不需要的插件
- 自动导入
- File –> Other Settings –> Default Project Structure
- idea.properties  disable.android.first.run=true
- ANDROID_SDK_HOME 环境变量指向 D:\android-home 把 .android 目录拷过去
- 禁止自动打开上次的工程
- 禁止代码折叠

Android:

- 加载中效果
- 短暂提示
- 网络访问要用AsyncTask
- 小图标
- 全局错误挂接
- 优雅退出
- 打日志
- 每页的标题
- 返回按钮
- 导航条一直存在


安卓异步任务类AsyncTask——突出一个简单、好用
https://blog.csdn.net/jerrycqu/article/details/49357191
Android OkHttp的基本用法
https://www.jianshu.com/p/c478d7a20d03
Toolbar的简单使用
https://blog.csdn.net/monalisatearr/article/details/78415585
ListView中自定义adapter的封装
https://www.cnblogs.com/smyhvae/p/4477079.html
Gson解析——从简单数据到复杂数据
https://www.jianshu.com/p/886f7b7fca7d
LoadingBar - 如何更优雅的使用Loading
https://blog.csdn.net/aa464971/article/details/70197394
NavigationView的使用
https://blog.csdn.net/bskfnvjtlyzmv867/article/details/70245826
Android中处理崩溃异常
https://blog.csdn.net/liuhe688/article/details/6584143




Android Studio 入门级教程（
https://www.cnblogs.com/abao0/archive/2017/06/02/6934023.html

详解 Android 的 Activity 组件
https://www.ibm.com/developerworks/cn/opensource/os-cn-android-actvt/
Android之自定义Adapter的ListView
http://www.cnblogs.com/topcoderliu/archive/2011/05/07/2039862.html
想写个app，在哪里可以找到icon素材？
https://www.zhihu.com/question/40639915?sort=created
https://github.com/konifar/android-material-design-icon-generator-plugin
Android学习 - 美化ListView
https://blog.csdn.net/wolflz/article/details/45078107
Android开发之漂亮Button样式
https://www.jianshu.com/p/e5e8a98fc5d9


图标制作
https://www.iconfinder.com/editor/
https://www.flaticon.com/free-icons/programming-language/2
https://glyphter.com/
https://www.shutterstock.com/zh/image-vector/workplace-programmer-coder-desktop-pc-laptop-705161689?src=Kd61QRsGtq1hI9vEL7LU2w-1-49
https://iconsflow.com/editor

APP第三方微信登录与公众号数据打通
https://www.jianshu.com/p/18b1288f4c41

Flex 布局教程：语法篇
http://www.ruanyifeng.com/blog/2015/07/flex-grammar.html
Flex 布局教程：实例篇
http://www.ruanyifeng.com/blog/2015/07/flex-examples.html

Programmed Lessons in QBasic
http://chortle.ccsu.edu/QBasic/index.html
small basic
https://smallbasic-publicwebsite.azurewebsites.net/Program/Editor.aspx
中等职业教育国家规划教材·编程语言基础：QBASIC语言（计算机及应用专业）
https://item.jd.com/10493424.html

wget https://npm.taobao.org/mirrors/node/v8.9.3/node-v8.9.3-linux-x64.tar.xz

tar -xzvf node-v8.9.3-linux-x64.tar.gz
tar -xvf node-v8.9.3-linux-x64.tar
ln -s /root/node-v8.9.3-linux-x64/bin/node /usr/local/bin/node
ln -s /root/node-v8.9.3-linux-x64/bin/npm /usr/local/bin/npm
npm -v
npm install -g cnpm --registry=https://registry.npm.taobao.org

npm config set registry http://registry.npm.taobao.org/

alias cnpm="npm --registry=https://registry.npm.taobao.org \
--cache=$HOME/.npm/.cache/cnpm \
--disturl=https://npm.taobao.org/dist \
--userconfig=$HOME/.cnpmrc"

$ sudo npm install forever -g   #安装
$ forever start app.js          #启动
$ forever stop app.js           #关闭
$ forever start -l forever.log -o out.log -e err.log app.js   #输出日志和错误

## nginx

## Apache backend for www.quancha.cn ##
upstream apachephp  {
    server ip:8080; #Apache
}
 
## Start www.quancha.cn ##
server {
    listen 80;
    server_name  www.quancha.cn;
 
    access_log  logs/quancha.access.log  main;
    error_log  logs/quancha.error.log;
    root   html;
    index  index.html index.htm index.php;
 
    ## send request back to apache ##
    location / {
        proxy_pass  http://apachephp;
 
        #Proxy Settings
        proxy_redirect     off;
        proxy_set_header   Host             $host;
        proxy_set_header   X-Real-IP        $remote_addr;
        proxy_set_header   X-Forwarded-For  $proxy_add_x_forwarded_for;
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
        proxy_max_temp_file_size 0;
        proxy_connect_timeout      90;
        proxy_send_timeout         90;
        proxy_read_timeout         90;
        proxy_buffer_size          4k;
        proxy_buffers              4 32k;
        proxy_busy_buffers_size    64k;
        proxy_temp_file_write_size 64k;
   }
}

How to easily convert utf8 tables to utf8mb4 in MySQL 5.5
https://dba.stackexchange.com/questions/8239/how-to-easily-convert-utf8-tables-to-utf8mb4-in-mysql-5-5

不翻墙也能找到免费优质素材，7个免费商用素材库墙裂推荐
https://zhuanlan.zhihu.com/p/28508804

微服务分布式事务Saga模式简介
http://www.jdon.com/49307

大数据告诉你：为啥近5年来Python如此火爆？
https://www.sohu.com/a/196290953_704222
kvm管理平台webvirtmgr的部署
https://www.jianshu.com/p/160272d81ac3
KVM 虚拟化环境搭建 - WebVirtMgr
https://zhuanlan.zhihu.com/p/49120559
KVM 虚拟化环境搭建 - ProxmoxVE
https://zhuanlan.zhihu.com/p/49118355
Nas 系统的虚拟化方案
https://zhuanlan.zhihu.com/p/55025102
Comfortable interface for KVM? (with PCIe Passthrough)
https://www.centos.org/forums/viewtopic.php?t=52640

CentOS之7与6的区别
https://www.cnblogs.com/Csir/p/6746667.html
开源虚拟化管理平台Ovirt简介和配置环境搭建
https://www.2cto.com/os/201202/120678.html
libvirt apps
https://libvirt.org/apps.html#web
十分钟带你理解Kubernetes核心概念
http://dockone.io/article/932
使用 Docker/LXC 迅速启动一个桌面系统
https://www.oschina.net/question/54100_137626
在Docker中运行桌面应用
https://yq.aliyun.com/articles/224645#
docker-desktop
https://github.com/rogaha/docker-desktop
图形化界面的 docker ?
https://www.zhihu.com/question/34493859
Windows Docker 安装
http://www.runoob.com/docker/windows-docker-install.html
https://blog.csdn.net/tina_ttl/article/details/51372604
后端架构师技术图谱
https://github.com/xingshaocheng/architect-awesome/blob/master/README.md#%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84

termtosvg
A Linux terminal recorder written in Python that renders your command line sessions as standalone SVG animations.
https://github.com/nbedos/termtosvg

如何编写技术解决方案
https://wenku.baidu.com/view/b42e31a1760bf78a6529647d27284b73f2423635.html?from=search

p5.js 
http://ofcourse.io/biao-ti-3/

腾讯云 使用DockerHub加速器
https://cloud.tencent.com/document/product/457/9113
理解Docker（7）：Docker 存储 - AUFS
https://www.cnblogs.com/sammyliu/p/5931383.html
［授权发表］基于 ssh + Xpra 构建 Docker 桌面系统
https://blog.csdn.net/tinylab/article/details/45443563
CentOS6下docker的安装和使用
https://www.cnblogs.com/zhangzhen894095789/p/6641981.html?utm_source=itdadao&utm_medium=referral

How to stop docker pull
https://stackoverflow.com/questions/29486032/how-to-stop-docker-pull
DOCKER_OPTS do not work in config file /etc/default/docker
https://stackoverflow.com/questions/27763340/docker-opts-do-not-work-in-config-file-etc-default-docker

Set Docker_Opts in centos
https://stackoverflow.com/questions/26166550/set-docker-opts-in-centos

vi /etc/sysconfig/docker
    OPTIONS=--registry-mirror=https://mirror.ccs.tencentyun.com
sudo service docker restart

netstat -tnpl
CONTAINER_ID=$(sudo docker run -d -p 2222:22 rogaha/docker-desktop)

echo $(sudo docker logs $CONTAINER_ID | sed -n 1p)
User: docker Password: eengoch3ooK5
docker port $CONTAINER_ID 22

/sbin/iptables -I INPUT -p tcp --dport 80 -j ACCEPT 
/sbin/iptables -I INPUT -p tcp --dport 22 -j ACCEPT 

然后保存： 

/etc/rc.d/init.d/iptables save 
centos 5.3，5.4以上的版本需要用 
service iptables save 

人人都该学编程？
https://www.jiemodui.com/Item/22041

How to Install PHP 7 in CentOS 7
https://www.tecmint.com/install-php-7-in-centos-7/


### .vimrc

set nocp
set ts=4
set sw=4
set smarttab
set et
set ambiwidth=double
colo torte
set nu

set encoding=UTF-8
set langmenu=zh_CN.UTF-8
language message zh_CN.UTF-8
set fileencodings=ucs-bom,utf-8,cp936,gb18030,big5,euc-jp,euc-kr,latin1
set fileencoding=utf-8


syntax on
filetype plugin indent on

### .bashrc

alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
alias vi='vim'

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

export LC_ALL='zh_CN.utf8'





Programmed Lessons in QBasic
http://chortle.ccsu.edu/QBasic/index.html  发现这个 qb 教程还挺好的

Shiro的三种授权(十二)
https://www.cnblogs.com/qlqwjy/p/7257616.html
走进Java（五）JSTL和EL表达式
https://blog.csdn.net/u010096526/article/details/50038365
JSTL获取Parameter参数
https://blog.csdn.net/kevinxxw/article/details/50884649
自定义页面方法：${fns:sayHelloToName('zxw')}
https://blog.csdn.net/zhengxiangwen/article/details/40652019
JSTL表达式的使用（c:if）以及在JS中使用
https://blog.csdn.net/weistin/article/details/80027218


eclipse ERMaster插件安装 利用ERMaster快速生成db维护文档
https://my.oschina.net/dajianguo/blog/1622944    


Eclipse启动时禁用不必要的验证。
https://blog.csdn.net/u012726702/article/details/51758596

单词补全 Alt + /


window---->preferences---->general----->content types----->Text------>java properties file---->UTF-8---->update ------>ok

Mybatis整合Spring -- typeAliasesPackage
https://blog.csdn.net/sky786905664/article/details/51801933

重启 PHP-FPM

sudo kill -usr2 $(cat /var/run/php-fpm/php-fpm.pid)
查看 PHP 配置

php --ini

解决Setting property 'source' to 'org.eclipse.jst.jee.server的问题
https://blog.csdn.net/z69183787/article/details/19911935

Tomcat中server.xml配置详解
https://www.cnblogs.com/yanghua1012/p/5869192.html
tomcat context元素属性介绍
http://outofmemory.cn/code-snippet/3035/tomcat-context-element-property-introduction
tomcat 8.0特性
https://blog.csdn.net/hmy1106/article/details/51270761
Tomcat 7 的七大新特性
http://www.iteye.com/news/17928/
Tomcat 5.5.x到Tomcat 6.0（tomcat6新特性及变化）
https://blog.csdn.net/wlanye/article/details/8570891

SpringMVC重要注解（四）@ModelAttribute
https://blog.csdn.net/lovesomnus/article/details/78873089
java怎么用一行代码初始化ArrayList
https://www.itstrike.cn/Question/e74b36fa-c01f-4254-87ec-e549df2abebe.html

JS 物理引擎
Physics engine in your JavaScript program
http://slicker.me/javascript/matter.htm
黑苹果
https://github.com/kholia/OSX-KVM
命令行下的电子表格
https://github.com/andmarti1424/sc-im
A*
http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html
寻路算法可视化
https://qiao.github.io/PathFinding.js/visual/
碰撞检测
http://mozdevs.github.io/gamedev-js-3d-aabb/raw_box.html
游戏算法
https://www.redblobgames.com/
4种方法让SpringMVC接收多个对象
https://blog.csdn.net/lutinghuan/article/details/46820023
Objects binding in Spring MVC form
https://stackoverflow.com/questions/10138715/objects-binding-in-spring-mvc-form
Spring 4 官方文档学习 Web MVC 框架
https://www.cnblogs.com/t3306/p/7244134.html
Allow binding=false on @ModelAttribute
https://github.com/spring-projects/spring-framework/commit/2e7470b27f0eaae042334cd86f212cd958676be0
SpringMVC<from:form>表单标签和<input>表单标签简介
https://blog.csdn.net/hp_yangpeng/article/details/51906654
Myths about /dev/urandom
https://www.2uo.de/myths-about-urandom
With Undefined Behavior, Anything is Possible
https://raphlinus.github.io/programming/rust/2018/08/17/undefined-behavior.html
How to Read 100s of Millions of Records per Second from a Single Disk
https://clemenswinter.com/2018/08/13/how-read-100s-of-millions-of-records-per-second-from-a-single-disk/
Introduction to Go Modules
https://roberto.selbach.ca/intro-to-go-modules/

轻量级前端框架
https://mithril.js.org/

Docker的web端管理平台对比（DockerUI 、Shipyard、Portainer、Daocloud）
https://blog.csdn.net/qq273681448/article/details/75007828
Docker可视化管理工具Portainer的安装配置及使用
https://blog.csdn.net/bbwangj/article/details/80973219
Swarm -- 搭建Docker集群
https://blog.csdn.net/a632189007/article/details/78756339

How to use UTF-8 in resource properties with ResourceBundle
https://stackoverflow.com/questions/4659929/how-to-use-utf-8-in-resource-properties-with-resourcebundle
Eclipse 在线安装properties编辑插件
https://www.cnblogs.com/DylanZ/p/6428709.html


稍等两分钟，就会出现插件列表，选择PropertiesEditor，然后Next.

Where does forever store console.log output?
https://stackoverflow.com/questions/21021186/where-does-forever-store-console-log-output

forever start -o out.log -e err.log my-script.js


Learn You a Haskell for Great Good!
http://learnyouahaskell.com/chapters

owncloud+collabora 实现网盘在线预览
https://blog.csdn.net/a295184686/article/details/78632706

Epigrams on Programming
http://pu.inf.uni-tuebingen.de/users/klaeren/epigrams.html


程序员不能把自己全部的时间都用来换钱，而是要拿出20%~40%的可支配时间来学习、尝试和应用新的技术，让自己的技术栈保持连续更新。


Guacamole 通过浏览器远程访问服务器
https://www.jianshu.com/p/ebaba8ca17de

vim buffer 操作

:bn -- buffer列表中下一个 buffer　　
:bp -- buffer列表中前一个 buffer　　
:b# -- 你之前所在的前一个 buffer
:bdelete num -- 删除第num编号buffer

JavaScript Space After Function [closed]
https://stackoverflow.com/questions/9300636/javascript-space-after-function
基于Docker快速搭建多节点Hadoop集群
http://dockone.io/article/395


How to Setup Single Node Hadoop Cluster Using Docker
https://linoxide.com/cluster/setup-single-node-hadoop-cluster-docker/

A Road to Common Lisp
http://stevelosh.com/blog/2018/08/a-road-to-common-lisp/#get-a-lisp

C compiler with support for structs written in Assembly
https://news.ycombinator.com/item?id=17851311

Oops, I Wrote a C++ Compiler
https://news.ycombinator.com/item?id=17851293

Double Buffer
http://gameprogrammingpatterns.com/double-buffer.html

JSCPP, a simple C++ interpreter written in JavaScript 
https://news.ycombinator.com/item?id=17843293

Ubuntu 16.04 硬盘安装
https://www.cnblogs.com/dzlixu/p/5e9475c3990d720ca22e18b730b01d57.html


## 编程挑战

### 挑战 1：

输出 1000 以内的斐波那契数列，用空格隔开，每行 5 个显示。

### 挑战 2：

假设公司年会抽奖，一等奖1名，二等奖3名，三等奖5名，共100人抽奖，写一个抽奖程序来抽奖。
奖品不要多发，不要漏发，考虑通用性和边界处理。以后大家工作后肯定会遇到类似场景的。

### 挑战 3：

写程序解析出一个网址的各个部分

比如 https://123.abc.com:8000/hello/world?x=111&y=222&z=333 这个网址要求输出如下

协议: https
域名: 123.abc.com
端口: 8000
路径: /hello/world
参数: x: 111, y: 222, z: 333
    
### 挑战 4 ：

随机生成 100 个 15 以内的随机正整数

1、统计出每个数字出现的次数
2、统计出出现次数最多和出现次数最少的前三个数字
3、统计出偶数和奇数各自出现的次数和它们的和
4、统计出出现次数最多的偶数和奇数




## docker 常用命令


    sudo docker build -t sshd:ubuntu2 .
    sudo docker run -d --name ssh_test -p 10122:22 sshd:ubuntu2
    sudo docker run -d --name ssh_test2 -P  sshd:ubuntu3
    sudo docker commit 161f67ccad50 sshd:ubuntu
    sudo  docker exec -it b1141d7b1937 bash


    sudo /home/helloworld/.local/bin/wssh --address='0.0.0.0' --port=80
    ssh root@localhost -p 10122
    
    docker run -it sequenceiq/hadoop-docker:2.7.0 /etc/bootstrap.sh -bash

    sudo docker ps -a    
    docker network create hadoop
    sudo docker-compose up -d
    sudo docker-compose stop
    sudo docker exec -it namenode bash

    hadoop fs -put /etc/issue /
    hadoop fs -ls /
    



    
    sudo docker run -d --name novnc -p 6080:80 dorowu/ubuntu-desktop-lxde-vnc
    
    sudo docker ps --filter ancestor=sshd:ubuntu2 \
        --format "table {{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}"
        
    sudo docker run -d -p 9000:9000     --restart=always     -v /var/run/docker.sock:/var/run/docker.sock     --name prtainer-test     docker.io/portainer/portainer
   
    docker run -m 512m --memory-swap 1G -it -p 58080:8080 --restart=always   
    --name bvrfis --volumes-from logdata mytomcat:4.0 /root/run.sh  
    docker update --restart=always xxx  
    sudo docker run --restart=on-failure:10 redis  
    
    sudo docker swarm init --advertise-addr 192.168.1.15
    docker swarm join-token worker
    docker swarm join --token SWMTKN-1-2nifhbsha6kzfu7pacy93z1niki425t2f3t807y1kzehxhsval-5zns821bezcby2k1o0v7y946z 192.168.1.15:2377
    sudo docker node ls
    
    sudo docker service create --replicas 1 --name hadoop-test01 sequenceiq/hadoop-docker /etc/bootstrap.sh
    sudo docker service ps hadoop-test01
    sudo docker service inspect --pretty hadoop-test01
    docker service scale helloworld=2
    sudo docker service rm hadoop-test01
    
    docker swarm leave
    
    sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ssh.1.uoyp9smajh939e98yc40ygm5z



sudo sed -i "s|EXTRA_ARGS='|EXTRA_ARGS='--registry-mirror=http://1d20ae13.m.daocloud.io |g" /var/lib/boot2docker/profile    

Docker(六)：Docker 三剑客之 Docker Swarm
https://baijiahao.baidu.com/s?id=1598047425152066542&wfr=spider&for=pc

Docker Swarm 部署Mysql/Mariadb高可用主从复制集群
http://www.chairis.cn/blog/article/90

docker 容器日志清理方案
http://www.chairis.cn/blog/article/94
    
docker应用-5（使用overlay 网络进行容器间跨物理主机通信）
https://www.jianshu.com/p/4bbaf761fad8

Docker Swarm集群部署实践
http://www.chairis.cn/blog/article/89    

        
在使用docker run启动容器时，使用--restart参数来设置：        
--restart具体参数值详细信息：
no -  容器退出时，不重启容器；
on-failure - 只有在非0状态退出时才从新启动容器；
always - 无论退出状态是如何，都重启容器；

Docker容器开机自动启动
https://blog.csdn.net/menghuanbeike/article/details/79261828

使用DockerToolbox自动创建Swarm集群+Portainer图形化管理的脚本
https://blog.csdn.net/CSDN_duomaomao/article/details/73430919

Docker-Hadoop ALL-IN-ONE
https://www.jianshu.com/p/6bcd72083c08

    
    
Docker Container Executor
https://hadoop.apache.org/docs/r2.7.2/hadoop-yarn/hadoop-yarn-site/DockerContainerExecutor.html
Docker image for Hadoop
https://github.com/bigdatafoundation/docker-hadoop    

使用Docker在本地搭建Hadoop分布式集群
https://www.cnblogs.com/onetwo/p/6419925.html
将Linux系统的home、var目录迁移到新分区
https://blog.csdn.net/davidhopper/article/details/79768073
How do I change the Docker image installation directory?
https://forums.docker.com/t/how-do-i-change-the-docker-image-installation-directory/1169
关于docker的15个小tip
http://www.cnblogs.com/elnino/p/3899136.html
运行 MapReduce 样例
https://blog.csdn.net/chengqiuming/article/details/78826143

Docker环境下Hadoop分布式集群搭建
https://blog.csdn.net/zhangfan1212/article/details/54791334

Docker-Hadoop ALL-IN-ONE
https://www.jianshu.com/p/6bcd72083c08
Docker创建私有仓库+SSL+AUTH+WEB
https://www.jianshu.com/p/d85398240f05
Docker-Compose入门
https://blog.csdn.net/chinrui/article/details/79155688
使用docker部署hadoop hdfs
http://fatkun.com/2017/12/deploy-hadoop-hdfs-use-docker.html

Docker Machine 是什么？
https://www.cnblogs.com/sparkdev/p/7044950.html





http://npm.taobao.org/mirrors/chromedriver/
v2.40 Supports Chrome v66-68
http://npm.taobao.org/mirrors/chromedriver/2.40/notes.txt

Windows下的Jupyter Notebook 安装与自定义启动（图文详解）
http://www.cnblogs.com/zlslch/p/6984403.html

pip install selenium

Python3.5+selenium操作Chrome浏览器
https://www.cnblogs.com/drake-guo/p/6188366.html

selenium 安装与 chromedriver安装
https://www.cnblogs.com/technologylife/p/5829944.html

最详尽使用指南：超快上手Jupyter Notebook
https://blog.csdn.net/DataCastle/article/details/78890469

Python+Selenium WebDriver API：浏览器及元素的常用函数及变量整理总结
https://www.cnblogs.com/yufeihlf/p/5764807.html

mkdir ~/.pip/

vi ~/.pip/pip.conf

    [global]
    index-url = https://pypi.tuna.tsinghua.edu.cn/simple
    [install]
    trusted-host=mirrors.aliyun.com
    
pip install virtualenv
virtualenv venv
echo venv >> .gitignore

pip install Flask
pip install docker
pip freeze > requirements.txt

mkdir static
mkdir templates

. venv/bin/activate
export FLASK_ENV=development
export FLASK_APP=dockerweb.py
flask run --host=0.0.0.0

wssh 目录
/home/helloworld/.local/lib/python2.7/site-packages/webssh
wssh debug 模式会去掉 csrf 
sudo /home/helloworld/.local/bin/wssh --address='0.0.0.0' --port=8000 --debug=true   

screen -dmS dockerweb
screen -r dockerweb


8天学会Hadoop
https://ke.qq.com/course/229879
https://ke.qq.com/course/276816
https://ke.qq.com/course/287048


GRANT ALL PRIVILEGES ON helloworld.* TO 'root'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON helloworld.* TO 'root'@'localhost' IDENTIFIED BY 'password';
GRANT SELECT ON *.* TO 'readonly'@'localhost' IDENTIFIED BY 'readonly';
flush privileges;

mysql添加用户、修改权限，修改登录权限ip
https://www.cnblogs.com/lemon-flm/p/7597879.html

An Introduction to Modern CMake
https://cliutils.gitlab.io/modern-cmake/

PDF 转换
https://www.ilovepdf.com/

sudo apt-get update
sudo apt-get upgrade

（实用）Ubuntu 、CentOS更换国内源
https://www.cnblogs.com/Security-Darren/p/3947952.html



curl -o /dev/null -s -w 'status=%{http_code}\ndns-time=%{time_namelookup}s\nconnec-time=%{time_connect}s\nserver-time=%{time_starttransfer}s\ntota-ltime=%{time_total}\ncontent-length=%{size_download}\n' http://helloworld.cn


function reserved_ip($ip)
{
    $reserved_ips = array( // not an exhaustive list
    '167772160'  => 184549375,  /*    10.0.0.0 -  10.255.255.255 */
    '3232235520' => 3232301055, /* 192.168.0.0 - 192.168.255.255 */
    '2130706432' => 2147483647, /*   127.0.0.0 - 127.255.255.255 */
    '2851995648' => 2852061183, /* 169.254.0.0 - 169.254.255.255 */
    '2886729728' => 2887778303, /*  172.16.0.0 -  172.31.255.255 */
    '3758096384' => 4026531839, /*   224.0.0.0 - 239.255.255.255 */
    );

    $ip_long = sprintf('%u', ip2long($ip));

    foreach ($reserved_ips as $ip_start => $ip_end)
    {
        if (($ip_long >= $ip_start) && ($ip_long <= $ip_end))
        {
            return TRUE;
        }
    }
    return FALSE;
}

var_dump(reserved_ip('127.0.0.1')); // reserved (localhost)
var_dump(reserved_ip('74.125.140.101')); // not reserved (Google)


iView 发布 3.0 版本，以及开发者社区等 5 款新产品
https://www.v2ex.com/t/475227#reply10




sudo su
mkdir /data/bakup
mysqldump -uroot -p --all-databases >/data/bakup/dump-`date +%Y%m%d%H%M%S`.sql
mysqldump -h 172.3.0.17 -uroot -p --database helloworld > helloworld.20190507.sql

Can I restore a single table from a full mysql mysqldump file?
https://stackoverflow.com/questions/1013852/can-i-restore-a-single-table-from-a-full-mysql-mysqldump-file

腾讯工程师带你深入解析 MySQL binlog
https://zhuanlan.zhihu.com/p/33504555


ubuntu开启crontab日志记录及解决No MTA installed, discarding output问题
http://www.pooy.net/ubuntu-open-crontab-logging-and-resolution-no-mta-installed-discarding-output-problem.html

ERROR: Failed to allocate directory watch: Too many open files
sysctl fs.inotify.max_user_instances=512


解决(CRON) info (No MTA installed, discarding output)
https://blog.csdn.net/win_turn/article/details/53000899


像素
https://art.pixlab.io/#picLoader

linux系统挂载U盘，中文文件名乱码解决方案
https://www.cnblogs.com/zhouqinxiong/p/3497293.html

Linux下添加新硬盘,分区及挂载
https://www.cnblogs.com/jiu0821/p/7209825.html

挂载 U 盘
mount -o iocharset=utf8 /dev/sdb4 /mnt

## 分区

    # 查看分区表，/dev/sda，/dev/sdb 表示硬盘，/dev/sda1, /dev/sda2 表示分区
    fdisk -l |grep '/dev/sd[abcd]'
    
    # 对 /dev/sdb 分区
    fdisk /dev/sdb

    # 添加新分区
    n
    
    # e 为扩展分区， p 为主分区
    p
    
    # 确认分区起始点，如果是新硬盘，输入 1，一般情况下直接回车
    1
    
    #输入分区大小，以下表示 1G，注意 m 是小写
    +1024m
    
    # 保存分区
    w
    
    # 查看新分好的区
    fdisk -l
    
    # 格式化
    mkfs -t ext4 -N 7000000 -c /dev/sdb1
    
    # 挂载
    mount /dev/sdb1 /www
    
    # 查看挂载的分区大小
    df -TH
    
    # 设置为开机自动挂载
    echo '/dev/sdb1 /www ext4 defaults 1 2' >> /etc/fstab
    

导出镜像
docker save -o vnc.tar vnc:0.01    
    
查看磁盘 IO    
iostat -d -x -k 1 | grep -E '(Device|sd[abcd])'


docker镜像的导入和导出
https://blog.csdn.net/x6_9x/article/details/72891404 


彻底禁用Chrome的“请停用以开发者模式运行的扩展程序”提示
提示
https://www.cnblogs.com/liuxianan/p/disable-chrome-extension-warning.html 真够复 真够复杂的






至强CPU多数都是可以配置多路的，就是多个CPU放在同一个板子上，这一点是i7无法做到的。
至强E5系列里，四位数字的第一位表示的就是CPU的个数，比如Xeon E5-2696 v3，表示此款CPU支持双路，
Xeon E5-4655 v3支持四路，那么我给的列表里E5最高配置是Xeon E5-4669 v3，四路，每路18核，每核双线程，
那么要组装到板子上，在Windows任务管理器里就可以看到壮观的144个核心的场面，

Intel超级大杀器发布:28核56线程桌面CPU,4.3G还不锁频
http://diy.pconline.com.cn/1180/11808584.html

ubuntu16.04开机启动字符界面
https://jingyan.baidu.com/article/948f5924ee2a5dd80ff5f9e4.html

非常实用的Linux 系统监控工具
https://www.cnblogs.com/mengdeep/p/5296991.html

了解VMware预订限制和共享
Understanding VMware Reservations Limits and Shares
http://www.vfrank.org/2013/09/19/understanding-vmware-reservations-limits-and-shares/

Docker: 限制容器可用的内存
http://www.cnblogs.com/sparkdev/p/8032330.html

Docker service Limits and Reservations
https://stackoverflow.com/questions/38200028/docker-service-limits-and-reservations

浮点算法
echo "scale=2;$a/$b" | bc


Swarm mode: Inspect worker node's container on manager node
https://stackoverflow.com/questions/39237998/swarm-mode-inspect-worker-nodes-container-on-manager-node


Error loading config file XXX.dockerconfig.json permission denied
https://blog.csdn.net/xukai0110/article/details/80637884

sudo chown "$USER":"$USER" /home/"$USER"/.docker -R
sudo chmod g+rwx "/home/$USER/.docker" -R

How can I use docker without sudo?
https://askubuntu.com/questions/477551/how-can-i-use-docker-without-sudo

sudo groupadd docker
sudo gpasswd -a $USER docker
newgrp docker
docker ps

How to determine where an environment variable came from
https://unix.stackexchange.com/questions/813/how-to-determine-where-an-environment-variable-came-from


ubuntu 死机，查看日志


Linux Kernel panic issue: How to fix hung_task_timeout_secs and blocked for more than 120 seconds problem
https://www.blackmoreops.com/2014/09/22/linux-kernel-panic-issue-fix-hung_task_timeout_secs-blocked-120-seconds-problem/


NO-BREAK SPACE
https://stackoverflow.com/questions/5237989/nonbreaking-space


VNC Tight Encoder - Comparison Results
http://tightvnc.com/archive/compare.html

Comparing 7 Monitoring Options for Docker
https://rancher.com/comparing-monitoring-options-for-docker-deployments/


一种基于负载预测的DockerSwarm集群资源调度优化方法与流程
http://www.xjishu.com/zhuanli/55/201710461892.html

运维之我的docker-集群的管理-swarm
http://blog.51cto.com/nginxs/1894432

Docker Stack Swarm - Service Replicas are not spread for Mutli Service Stack
https://stackoverflow.com/questions/44868123/docker-stack-swarm-service-replicas-are-not-spread-for-mutli-service-stack


https://www.cnblogs.com/aguncn/p/6904522.html

清理Docker占用的磁盘空间
https://blog.csdn.net/weixin_32820767/article/details/81196250

Docker网络和 overlay跨主机通讯
http://blog.51cto.com/littledevil/1922922

Docker Swarm 入门：Service Network 管理
https://www.jianshu.com/p/60bccbdb6af9


Battlefield: Calico, Flannel, Weave and Docker Overlay Network
http://xelatex.github.io/2015/11/15/Battlefield-Calico-Flannel-Weave-and-Docker-Overlay-Network/


Demystifying Docker overlay networking
http://blog.nigelpoulton.com/demystifying-docker-overlay-networking/

DOCKER_OPTS do not work in config file /etc/default/docker
https://stackoverflow.com/questions/27763340/docker-opts-do-not-work-in-config-file-etc-default-docker

journalctl -xe

金步国作品集 各种手册翻译
http://www.jinbuguo.com/

分页无截断查看日志
journalctl -rxn -u docker.service | less
查看错误日志
journalctl -p err..alert
查看内核日志
journalctl -k
显示本次启动后的所有日志：
journalctl -b

journalctl: how to prevent text from truncating in terminal
https://unix.stackexchange.com/questions/229188/journalctl-how-to-prevent-text-from-truncating-in-terminal

systemd 之 journalctl
https://www.cnblogs.com/itxdm/p/Systemd_log_system_journalctl.html

Systemd 常规操作与彩蛋
http://www.cnblogs.com/itxdm/p/Systemd_regular_operation_with_eggs.html


DOCKER图形页面管理工具--3种，shipyard最强大，其次是portainer
https://blog.csdn.net/xl_lx/article/details/81183956

Protect the Docker daemon socket
docker 进程认证
https://docs.docker.com/engine/security/https/

Docker security 安全
https://docs.docker.com/engine/security/security/#kernel-namespaces


 算机专业课一体化支撑平台 北航 CG OJ
http://www.cjudge.net/FAQs.html

linux开发神器--Tmux
https://www.cnblogs.com/ArsenalfanInECNU/p/5756763.html
更好利用 tmux 会话的 4 个技巧
http://ju.outofmemory.cn/entry/372410


tmux new -s monitor

暂时离开窗口

    ctrl + b, d 离开会话
    tmux ls 查看会话
    tmux attach -t 0

按下 Ctrl-b 后的快捷键如下：


% 创建一个水平窗格
" 创建一个竖直窗格
方向键切换窗格
q 显示窗格的编号
o 在窗格间切换
ctrl + 方向键， alt + 方向键，切换大小
d 离开会话，然后可以用 tmux attach 挂载
x 关闭当前窗格

node js 进程守护神forever
https://blog.csdn.net/jbboy/article/details/35281225

apt-cache show xserver-xorg | grep Version
https://superuser.com/questions/366505/how-to-find-out-xorg-version-or-whats-my-xorg-version

htop 快捷键，H 切换用户线程 K 切换内核线程

Why docker uses port numbers from 32768-65535?
https://stackoverflow.com/questions/40787524/why-docker-uses-port-numbers-from-32768-65535

容器删除后，主机映射给容器的端口为何并立即未回收利用？
https://segmentfault.com/q/1010000000496866

NGINX as a WebSocket Proxy
https://www.nginx.com/blog/websocket-nginx/


Ganglia 权威指南-安装Ganglia过程
http://www.cnblogs.com/chris-cp/p/4324392.html



第2章 rsync(一)：基本命令和用法
https://www.cnblogs.com/f-ck-need-u/p/7220009.html

rsync -avP /var/www/html/ helloworld@192.168.1.102:/var/www/html


Eclipse快捷键 10个最有用的快捷键
https://blog.csdn.net/lynn349x/article/details/56282704

How to convert an rtf string to text in C#
https://stackoverflow.com/questions/5634525/how-to-convert-an-rtf-string-to-text-in-c-sharp

grep 和 awk的buffer
https://blog.csdn.net/csCrazybing/article/details/78096301

How to fix stdio buffering
http://www.perkin.org.uk/posts/how-to-fix-stdio-buffering.html


Bash History Display Date And Time For Each Command
https://www.cyberciti.biz/faq/unix-linux-bash-history-display-date-time/


hadoop 集群 硬件配置
http://info.ipieuvre.com/article/201506/250.html
https://www.oschina.net/translate/how-to-select-the-right-hardware-for-your-new-hadoop-cluster

Docker 日志都在哪里？怎么收集？
https://www.cnblogs.com/YatHo/p/7866029.html

PHP file_get_contents fails to read zip on windows
https://stackoverflow.com/questions/27405430/php-file-get-contents-fails-to-read-zip-on-windows

baidu ai studio
http://aistudio.baidu.com/#/projectDetail/35620

百度实验平台
http://abcinstitute.baidu.com/lab/experiment/list;JSESSIONID=d8d1ce38-2afe-4324-b976-2f908f9dea15

机器学习——15分钟透彻理解感知机
https://blog.csdn.net/yxhlfx/article/details/79093456

自然语言处理 中文分词 词性标注 命名实体识别 依存句法分析 新词发现 关键词短语提取 自动摘要 文本分类聚类 拼音简繁
https://github.com/hankcs/HanLP#1-%E7%AC%AC%E4%B8%80%E4%B8%AAdemo

搜狗实验室
http://www.sogou.com/labs/

word2vec原理推导与代码分析
http://www.hankcs.com/nlp/word2vec.html

识别数字
http://www.paddlepaddle.org/documentation/docs/zh/develop/beginners_guide/quick_start/recognize_digits/README.cn.html

PaddlePaddle  新手入门
http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/index.html

Deep Learning（深度学习） 中文版
https://github.com/exacity/deeplearningbook-chinese

神经网络的四块积木：全连接，激活函数，卷积，池化
三种神经网络模型：Softmax回归模型，多层感知机模型，卷积神经网络模型
LeNet-5

三分钟带你对 Softmax 划重点
https://blog.csdn.net/red_stone1/article/details/80687921
小白都能看懂的softmax详解
https://blog.csdn.net/bitcarmanlee/article/details/82320853

Istio是啥？一文带你彻底了解！
https://www.sohu.com/a/270131876_463994

nginx配合modsecurity实现WAF功能
https://www.52os.net/articles/nginx-use-modsecurity-module-as-waf.html
https://www.techrepublic.com/article/how-to-install-and-enable-modsecurity-with-nginx-on-ubuntu-server/
CentOS Linux下用Nginx和Naxsi搭建Web应用防火墙
http://blog.cnwyhx.com/centos-nginx-naxsi-install/
nginx下安装配置naxsi waf防火墙（附完整编译、配置）
http://f2ex.cn/nginx-installed-configuration-naxsi-waf/

nginx的sql注入过滤配置
http://blog.itpub.net/30208512/viewspace-1578399/

Nginx 配置格式化小工具
https://github.com/fangpeishi/ngxfmt

轻松玩转OpenWAF之安装篇.md
https://github.com/titansec/OpenWAF/blob/master/doc/%E8%BD%BB%E6%9D%BE%E7%8E%A9%E8%BD%ACOpenWAF%E4%B9%8B%E5%AE%89%E8%A3%85%E7%AF%87.md

How to install and enable ModSecurity with NGINX on Ubuntu Server
https://www.techrepublic.com/article/how-to-install-and-enable-modsecurity-with-nginx-on-ubuntu-server/


Compatibility of ModSecurity with NginX waf
https://stackoverflow.com/questions/44978041/compatibility-of-modsecurity-with-nginx


2018-12-27T16:45:54+08:00 404 0.000 - 119.180.52.108 "POST /admin/login HTTP/1.1" 571 "http://helloworld.cn/admin/login" "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36" "-"


2018/12/27 16:46:52 [error] 9952#0: *132 open() "/data/release/ythelloworld/src/public/admin/login" failed (2: No such file or directory), client: 119.180.52.108, server: helloworld.cn, request: "POST /admin/login HTTP/1.1", host: "helloworld.cn", referrer: "http://helloworld.cn/admin/login"


How to Install Nginx with libModSecurity and OWASP core rule set on Ubuntu 16
https://www.hugeserver.com/kb/install-nginx-libmodsecurity-owasp-core-ruleset-ubuntu16/

Ubuntu 16.04上使用libmodsecurity的Nginx和OWASP ModSecurity核心规则
https://www.howtoing.com/nginx-with-libmodsecurity-and-owasp-modsecurity-core-rule-set-on-ubuntu-1604/

How to Install libmodsecurity + nginx on Ubuntu 14.04
https://help.dreamhost.com/hc/en-us/articles/223608748-How-to-Install-libmodsecurity-Nginx-on-Ubuntu-14-04


斯坦福大学公开课 ：机器学习课程
http://open.163.com/special/opencourse/machinelearning.html
百度 AI 课程
https://ai.baidu.com/paddlepaddle/player?id=13



Cleaning up docker to reclaim disk space
https://lebkowski.name/docker-volumes/

#!/bin/bash

    # remove exited containers:
    docker ps --filter status=dead --filter status=exited -aq | xargs -r docker rm -v

    # remove unused images:
    docker images --no-trunc | grep '<none>' | awk '{ print $3 }' | xargs -r docker rmi

    # remove unused volumes (needs to be ran as root):
    find '/var/lib/docker/volumes/' -mindepth 1 -maxdepth 1 -type d | grep -vFf <(
      docker ps -aq | xargs docker inspect | jq -r '.[]|.Mounts|.[]|.Name|select(.)'
    ) | xargs -r rm -fr
    
    
    
CREATE DATABASE IF NOT EXISTS helloworld default charset utf8 COLLATE utf8_general_ci; 

Windows远程数据同步工具cwRsync
https://www.cnblogs.com/l1pe1/p/4901031.html

Netplan – How To Configure Static IP Address in Ubuntu 18.04 using Netplan
https://www.itzgeek.com/how-tos/linux/ubuntu-how-tos/netplan-how-to-configure-static-ip-address-in-ubuntu-18-04-using-netplan.html

apt 下载
sudo apt-get install -d  network-manager
apt-offline - offline apt package manager
apt-zip - Update a non-networked computer using apt and removable media
apt download <package_name>

https://stackoverflow.com/questions/4419268/how-do-i-download-a-package-from-apt-get-without-installing-it

https://blog.csdn.net/thinktalk/article/details/83716617

Ubuntu中启用关闭Network-manager网络设置问题
https://blog.csdn.net/weixin_42155195/article/details/80683166

如何在 Linux 上使用网络配置工具 Netplan
https://linux.cn/article-10095-1.html?pr

ubuntu使用apt-get制作OfflinePackage
https://blog.csdn.net/ouyangziling/article/details/79056161


rsync也可以通过校验和比较文件。

--size-only这意味着rsync将跳过大小匹配的文件，即使时间戳不同。这意味着它将同步比默认行为更少的文件

--ignore-times 这意味着rsync将检查每个文件的和，即使时间戳和文件大小匹配。这意味着它将同步比默认行为更多的文件。


kvm vnc


vnc viewer连接kvm虚拟机连接不上的解决方法
https://blog.csdn.net/kepa520/article/details/50329505

video vga里变成了qxl，调回vga




rsync “Operation not permitted”
https://serverfault.com/questions/296587/rsync-operation-not-permitted

函数响应式编程（FRP）思想
https://blog.csdn.net/fly1183989782/article/details/62053973

漫谈FRP之 其实你早就学过FRP
http://insights.thoughtworkers.org/frp-series-0-not-new-concept/

函数编程中functor和monad的形象解释
https://www.jdon.com/idea/functor-monad.html

bootstrap dropdown hover menu
https://codepen.io/bsngr/pen/frDqh

How to create a collapsing tree table in html/css/js?  树形结构
https://stackoverflow.com/questions/5636375/how-to-create-a-collapsing-tree-table-in-html-css-js

chrome插件 —— 帮助高度还原设计稿
https://www.cnblogs.com/joyho/articles/5622760.html

OpenPAI：大规模人工智能集群管理平台
https://www.msra.cn/zh-cn/news/features/openpai


Cython入门到放弃（一）
https://blog.csdn.net/qtlyx/article/details/80614608

Best approach to encrypt big files with php
https://stackoverflow.com/questions/16175154/best-approach-to-encrypt-big-files-with-php

加密解密
Whole File Encryption/Decryption With PHP
http://monkeylogic.com/whole-file-encryptiondecryption-with-php/

All the crypto code you’ve ever written is probably broken
https://tonyarcieri.com/all-the-crypto-code-youve-ever-written-is-probably-broken

openssl -aes-128-ecb encryption doesn't match python Crypto.Cipher AES encryption
https://stackoverflow.com/questions/48410452/openssl-aes-128-ecb-encryption-doesnt-match-python-crypto-cipher-aes-encryptio

A Python-to-PHP compatible AES encryption with openssl_encrypt AES-CBC
https://stackoverflow.com/questions/40820661/a-python-to-php-compatible-aes-encryption-with-openssl-encrypt-aes-cbc

PHP 加密，Python 解密
A Python-to-PHP compatible AES encryption with openssl_encrypt AES-CBC
https://stackoverflow.com/questions/40820661/a-python-to-php-compatible-aes-encryption-with-openssl-encrypt-aes-cbc

云脑科技徐昊：AutoML 工程实践与大规模行业应用
http://www.lyweixiao.com/5314124/20190105A0YCBV00.html

https://serverfault.com/questions/870568/fatal-error-cant-open-and-lock-privilege-tables-table-storage-engine-for-use
Fatal error: Can't open and lock privilege tables: Table storage engine for 'user' doesn't have this option

Docker CPU 资源限制——CPU固定核功能测试
https://www.cnblogs.com/zhenyuyaodidiao/p/5061884.html

Linux下限制进程的CPU利用率
https://blog.csdn.net/bailyzheng/article/details/51355384

Linux---使用 nice、cpulimit 和 cgroups 限制 cpu 占用率
https://blog.csdn.net/loyachen/article/details/52167124

How do I install Ubuntu Server (step-by-step)?
https://askubuntu.com/questions/340965/how-do-i-install-ubuntu-server-step-by-step

H5 游戏引擎
 phaser，pixi，enchant.js

【blockly教程】第一章 Google Blockly教学应用手册
https://www.cnblogs.com/scratch8/archive/2018/09/12/9637214.html
https://www.cnblogs.com/scratch8/category/1299483.html
https://v.youku.com/v_show/id_XMTg3MTYzNjYzNg==.html
https://blog.csdn.net/oalevel/article/details/81834837


用友T3
http://www.3322.cc/soft/43731.html

Redis 设计与实现¶
http://redisbook.com/index.html

技术图书作译者的炼成方法
http://blog.huangz.me/2017/how-i-became-a-writer-and-translator.html

《Redis in Action》翻译记事
http://blog.huangz.me/diary/2015/memories-of-redis-in-action-translation.html

分布式监控工具Ganglia 介绍 与 集群部署.
https://www.cnblogs.com/yuki-lau/p/3201110.html 

数据可视化（三）基于 Graphviz 实现程序化绘图
https://riboseyim.com/2017/09/15/Visualization-Graphviz/

全面学习Prometheus
https://www.sohu.com/a/233111809_198222

Cacti：cacti是用php语言实现的一个软件，它的主要功能是用snmp服务获取数据，然后用rrdtool储存和更新数据，当用户需要查看数据的时候用rrdtool生成图表呈现给用户。
Nagios：Nagios 利用其众多的插件实现对本机和远端服务的监控，当被监控对象出现异常，Nagios 就会及时给管理人员告警。
Graphite：Graphite 是处理可视化和指标数据的优秀开源工具。它有强大的查询 API 和相当丰富的插件功能设置。

Kibana：Kibana是一个针对Elasticsearch的开源分析及可视化平台，用来搜索、查看交互存储在Elasticsearch索引中的数据
Grafana：Grafana是一个跨平台的开源的度量分析和可视化工具，可以通过将采集的数据查询然后可视化的展示，并及时通知

Zabbix：zabbix 是一个基于WEB界面的提供分布式系统监视以及网络监视功能的企业级的开源解决方案
Ganglia：Ganglia是UC Berkeley发起的一个开源集群监视项目，设计用于测量数以千计的节点。
Prometheus：Prometheus 将所有信息都存储为时间序列数据，实时分析系统运行的状态、执行时间、调用次数等，以找到系统的热点，为性能优化提供依据。


通过建立完善的监控体系，从而达到以下目的：

长期趋势分析：通过对监控样本数据的持续收集和统计，对监控指标进行长期趋势分析。例如，通过对磁盘空间增长率的判断，我们可以提前预测在未来什么时间节点上需要对资源进行扩容。
对照分析：两个版本的系统运行资源使用情况的差异如何？在不同容量情况下系统的并发和负载变化如何？通过监控能够方便的对系统进行跟踪和比较。
告警：当系统出现或者即将出现故障时，监控系统需要迅速反应并通知管理员，从而能够对问题进行快速的处理或者提前预防问题的发生，避免出现对业务的影响。
故障分析与定位：当问题发生后，需要对问题进行调查和处理。通过对不同监控指标以及历史数据的分析，能够找到并解决根源问题。
数据可视化：通过可视化仪表盘能够直接获取系统的运行状态、资源使用情况、以及服务运行状态等直观的信息。


elk
日志存几天，可以按天分区的，定时删
其它运维指标和运营指标存的时间比较久
单元测试和拨测成功率都扔es里，每天发报表到小组邮件里
还有每天的 Error 日志 loggerName， 慢速 api，慢速 sql 的top 10
每个规模指标，收入指标，LoggerName，API 性能，慢 SQL 都有对应负责人
日志组件是 error 日志 es, db，本机文本都记一份，warning 日志只记 es 和 本机文本，debug 只记录到本机
查问题先查 db，然后es，最后写了个分布式 grep  可以在web界面上对多台机器上同时进行 grep，前提是每台机器的日志的相对路径是一样的，而且使用者在cmdb里有这几台机器的权限。


【技能】小白耳机维修入门--各种耳机插头接线图--耳机维修汇总贴
https://www.cnblogs.com/tony-ning/p/7445761.html

科普帖：深度学习中GPU和显存分析
https://blog.csdn.net/liusandian/article/details/79069926
深度学习需要的显卡配置
https://blog.csdn.net/pjh23/article/details/83513066
【深度学习】为什么深度学习需要大内存？
https://blog.csdn.net/shenxiaolu1984/article/details/71522141

Docker环境下玩转GPU(一)
https://www.jianshu.com/p/2442394a934e?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation

ML之NN：BP solve XOR Problem
https://blog.csdn.net/qq_41185868/article/details/80789982
【机器学习】神经网络实现异或（XOR）
http://www.cnblogs.com/Belter/p/6711160.html

不用算术运算符实现两个数的加法(按位异或)
http://www.cnblogs.com/houjun/p/4908725.html


Codeup墓地 Online Judge FAQ
http://codeup.cn/faqs.php
OnlineJudge 2.0
https://github.com/QingdaoU/OnlineJudge/blob/master/README-CN.md
模型评估指标AUC（area under the curve）
https://blog.csdn.net/liweibin1994/article/details/79462554


获取 group by 后的序号，第几组的第几条数据：
        
SELECT 
@exp_number:=CASE WHEN @task_id=task_id THEN @exp_number+1 ELSE 1 END AS exp_number,
@task_number:=CASE WHEN @exp_number = 1 THEN @task_number+1 ELSE @task_number END AS task_number,
@task_id:=task_id AS task_id,
CONCAT(@task_number, '.', @exp_number) AS seq,
exp_id
FROM exp_course_class,(SELECT @task_number:=0, @exp_number:=0,@task_id:='') AS t
WHERE course_id = 41 AND class_id = 40 GROUP BY  task_id, pos, exp_id 
ORDER BY task_id, pos, task_number, exp_number;       

线性代数——向量、向量加法、向量数乘
https://blog.csdn.net/Fancy_Real/article/details/79944916
线性代数的本质学习笔记（1）：向量、线性组合、张成（SPAN）、线性变换
https://blog.csdn.net/dugudaibo/article/details/78714668

矩阵-DirectX与OpenGL的不同
http://www.cnblogs.com/graphics/archive/2012/08/02/2616017.html

## 数学
物理专业：向量是空间中的箭头，由长度和方向决定。可以自由移动；数学专业：向量可以是任何东西，只保证向量加法和数乘运算有意义；计算机专业：向量是有序的数字列表，起始点在原点上。

所有可以表示为给定向量线性组合的向量集合被称为给定向量张成的空间。

两个向量在三维空间中往往构成的是一个平面，有的时候也可能是一个直线。三个向量一般在三维空间中张成了整个三维空间，可以想象一下，两个向量已经张成了一个平面，而平面沿着第三个向量进行平移，张成了整个三维空间。新增的向量如果落在了原有向量张成的空间中，那么这个向量与原先的向量是线性相关的；另一方面如果所有向量都给张成的空间增添了新的维度，它们就是线性无关的。

向量在右 矩阵在左的时候

线性变换就是把矩阵里的值看成列向量

空间投影就是把矩阵看成行向量（每个行向量和右边向量点积，点积即投影）

首先，线性变换是把一个向量变为另一个向量，这里不涉及坐标变换。形式上，你可以把向量写成一行，然后右边乘一个矩阵，结果仍然写成了一行，成为新的向量。至于向量左乘矩阵，当这个向量的每个分量都表示一个基的时候，这就是基变换；而当这个向量的每个分量只是向量分量时，这就是线性变换。

线性组合 线性变换 线性相关 线性可分 空间投影

OpenGL中经常会使用的平移矩阵、缩放矩阵以及旋转矩阵，投影矩阵。

解析解，是指通过严格的公式所求得的解。即包含分式、三角函数、指数、对数甚至无限级数等基本函数的解的形式。给出解的具体函数形式，从解的表达式中就可以算出任何对应值。用来求得解析解的方法称为解析法，解析法是常见的微积分技巧，如分离变量法等。解析解为一封闭形式的函数，因此对任一独立变量，皆可将其代入解析函数求得正确的相依变量。因此，解析解也称为闭式解。

特别的，凸集，实数R上（或复数C上）的向量空间中，如果集合S中任两点的连线上的点都在S内，则称集合S为凸集。

凸函数是一个定义在某个向量空间的凸子集C（区间）上的实值函数f，而且对于凸子集C中任意两个向量  , f((x1+x2)/2)>=(f(x1)+f(x2))/2,则f(x)是定义在凸子集c中的凸函数（该定义与凸规划中凸函数的定义是一致的，下凸）。

牛顿法是一种在实数域和复数域上近似求解方程的方法。方法使用函数f(x)的泰勒级数的前面几项来寻找方程f(x) = 0的根。牛顿法最大的特点就在于它的收敛速度很快。

梯度下降法实现简单，当目标函数是凸函数时，梯度下降法的解是全局解。一般情况下，其解不保证是全局最优解，梯度下降法的速度也未必是最快的。梯度下降法的优化思想是用当前位置负梯度方向作为搜索方向，因为该方向为当前位置的最快下降方向，所以也被称为是”最速下降法“。

梯度的本意是一个向量（矢量），表示某一函数在该点处的沿着该方向取得最大值，即函数在该点处沿着该方向（此梯度的方向）变化最快，变化率最大（为该梯度的模）。

在函数定义域的内点，对某一方向求导得到的导数。一般为二元函数和三元函数的方向导数，方向导数可分为沿直线方向和沿曲线方向的方向导数。

启发式优化方法种类繁多，包括经典的模拟退火方法、遗传算法、蚁群算法以及粒子群算法等等。

共轭梯度法是介于最速下降法与牛顿法之间的一个方法，它仅需利用一阶导数信息，但克服了最速下降法收敛慢的缺点，又避免了牛顿法需要存储和计算Hesse矩阵并求逆的缺点，共轭梯度法不仅是解决大型线性方程组最有用的方法之一，也是解大型非线性最优化最有效的算法之一。

共轭在数学、物理、化学、地理等学科中都有出现。 本意：两头牛背上的架子称为轭，轭使两头牛同步行走。共轭即为按一定的规律相配的一对。通俗点说就是孪生。

作为一种优化算法，拉格朗日乘子法主要用于解决约束优化问题，它的基本思想就是通过引入拉格朗日乘子来将含有n个变量和k个约束条件的约束优化问题转化为含有（n+k）个变量的无约束优化问题。拉格朗日乘子背后的数学意义是其为约束方程梯度线性组合中每个向量的系数。

最优化问题的共同特点是：求满足一定条件的变量x1，x2，…，xn，使某函数f(x1，x2，…，xn)取得最大值或者最小值。

矩阵A为n阶方阵，若存在n阶矩阵B，使得矩阵A、B的乘积为单位阵，则称A为可逆阵，B为A的逆矩阵。若方阵的逆阵存在，则称为可逆矩阵或非奇异矩阵，且其逆矩阵唯一。
在矩阵的乘法中，有一种矩阵起着特殊的作用，如同数的乘法中的1，这种矩阵被称为单位矩阵。它是个方阵，从左上角到右下角的对角线（称为主对角线）上的元素均为1。除此以外全都为0。

矩阵求导（Matrix Derivative）也称作矩阵微分（Matrix Differential），在机器学习、图像处理、最优化等领域的公式推导中经常用到。

矩阵的微积分本质上是多元变量的微积分问题，只是应用在矩阵空间上而已

机器学习中的线性代数之矩阵求导
https://blog.csdn.net/u010976453/article/details/54381248

参数估计是机器学习里面的一个重要主题，而极大似然估计是最传统、使用最广泛的估计方法之一。

十分钟学习极大似然估计
https://endlesslethe.com/easy-to-learn-mle.html

极大似然估计其实是理想地认为，对于极少的样本观测，我们观测到的样本很可能就是发生概率最大的。

Latex数学公式简明教程
https://endlesslethe.com/latex-math-formula-tutorial.html

最小二乘法是勒让德( A. M. Legendre)于1805年在其著作《计算慧星轨道的新方法》中提出的。它的主要思想就是求解未知参数，使得理论值与观测值之差（即误差，或者说残差）的平方和达到最小：

把  矩阵的行列互换之后得到的矩阵，称为  的转置矩阵，记作 A^T
一个n阶方阵A称为可逆的，或非奇异的，如果存在一个n阶方阵B，使得AB=BA=E则称B是A的一个逆矩阵。A的逆矩阵记作A-1。


3D矩阵变换中，投影矩阵是最复杂的。位移和缩放变换一目了然，旋转变换只要基本的三角函数就能想象出来，投影矩阵则很难凭借直觉想象出来。

当样本量m很少，小于特征数n的时候，这时拟合方程是欠定的，需要使用LASSO。当m=n时，用方程组求解。当m>n时，拟合方程是超定的，我们可以使用最小二乘法。

投影矩阵推导(翻译)
https://www.cnblogs.com/davelink/p/5623760.html

损失函数（loss function）或代价函数（cost function）是将随机事件或其有关随机变量的取值映射为非负实数以表示该随机事件的“风险”或“损失”的函数。在应用中，损失函数通常作为学习准则与优化问题相联系，即通过最小化损失函数求解和评估模型。

机器学习：Python中如何使用最小二乘法
https://www.cnblogs.com/lc1217/p/6514734.html

机器学习:形如抛物线的散点图在python和R中的非线性回归拟合方法
http://www.cnblogs.com/lc1217/p/6519860.html


机器学习：Python实现单层Rosenblatt感知器
https://www.cnblogs.com/lc1217/p/6530177.html

Rosenblatt感知器详解
https://www.cnblogs.com/lanix/p/5003521.html

Python 机器学习
https://www.cnblogs.com/lc1217/category/960537.html

深度学习：Keras入门(一)之基础篇
https://www.cnblogs.com/lc1217/p/7132364.html

常见硬盘IOPS参考值
https://elf8848.iteye.com/blog/1731301

对机械硬盘和SSD固态硬盘IOPS、吞吐量的压测对比
http://blog.itpub.net/28916011/viewspace-2200027/

硬盘IOPS与读写速度
https://blog.csdn.net/star890124/article/details/52004138

磁盘性能指标--IOPS、吞吐量及测试
https://blog.51cto.com/wushank/1708168

centos7 df 命令卡死
https://www.cnblogs.com/John-2011/p/9577038.html

systemctl restart proc-sys-fs-binfmt_misc.automount;

调用谷歌翻译API
https://www.jianshu.com/p/6187d5915f70

解决pip的警告
pip install pyopenssl ndg-httpsclient pyasn1



禁用 root ssh，增加新用户

useradd helloworld
passwd helloworld
usermod -G wheel helloworld
usermod -G ssh helloworld
usermod -aG sudo username

vi /etc/ssh/sshd_config
    port 36000
    PermitRootLogin no
    
    
用sklearn做一个完整的机器学习工程——以波士顿房价预测为例。（一、用自定义转换器、Pipeline Feature_Union做特征工程）
https://blog.csdn.net/PythonstartL/article/details/82874932
https://blog.csdn.net/PythonstartL/article/details/82991548
https://blog.csdn.net/PythonstartL/article/details/83347173

https://blog.csdn.net/sdoddyjm68/article/details/78991699

香港大学深度学习课件笔记（1.5）
https://blog.csdn.net/sdoddyjm68/article/details/78409202

总结：sklearn机器学习之特征工程
https://www.jianshu.com/p/1c4ec02dd33f

Sklearn 笔记
https://www.cnblogs.com/yaoz/tag/Sklearn/

机器学习中的范数规则化之（一）L0、L1与L2范数
https://blog.csdn.net/zouxy09/article/details/24971995
笔记︱范数正则化L0、L1、L2-岭回归&Lasso回归（稀疏与特征工程）
https://blog.csdn.net/sinat_26917383/article/details/52092040

机器学习中的损失函数 （着重比较：hinge loss vs softmax loss）
https://blog.csdn.net/u010976453/article/details/78488279

zouxy09博客原创性博文导航：怀着对机器学习和计算机视觉
https://blog.csdn.net/zouxy09/article/details/14222605

Python机器学习库scikit-learn实践
https://blog.csdn.net/zouxy09/article/details/48903179    


iris-经典案例解析-机器学习
https://www.jianshu.com/p/da18f0cd7f60

机器学习训练数据
http://archive.ics.uci.edu/ml/index.php

tensorflow提示未编译使用SSE4.1，SSE4.2等问题的解决方法
https://blog.csdn.net/qq_36511757/article/details/77895316


Ubuntu 18.04 安装 nodejs
Ubuntu 18.04 LTS Server npm : Depends: node-gyp (>= 0.10.9) but it is not going to be installed [duplicate]
https://askubuntu.com/questions/1057737/ubuntu-18-04-lts-server-npm-depends-node-gyp-0-10-9-but-it-is-not-going

ubuntu无法修正错误，因为您要求某些软件包保持现状...解决办法
https://blog.csdn.net/buppt/article/details/78914234

sudo apt install npm

    npm : 依赖: node-gyp (>= 0.10.9) 但是它将不会被安装
    
sudo apt install node-gyp

    node-gyp : 依赖: nodejs-dev 但是它将不会被安装    
    
sudo apt install nodejs-dev

    nodejs-dev : 依赖: libssl1.0-dev (>= 1.0.2) 但是它将不会被安装    
    
sudo apt install libssl1.0-dev    
sudo apt -f install npm


How to install Node.js and npm on Ubuntu 18.04
https://linuxize.com/post/how-to-install-node-js-on-ubuntu-18.04/


## 小型团队工程化 CheckList



tar 压缩中文文件乱码
sudo apt install p7zip-full
7za a x.7z *.ipynb *.py


Issue in installing php7.2-mcrypt
https://stackoverflow.com/questions/48275494/issue-in-installing-php7-2-mcrypt
https://lukasmestan.com/install-mcrypt-extension-in-php7-2/


apt install php-pear
pecl version

apt-get -y install gcc make autoconf libc-dev pkg-config
apt-get -y install php7.2-dev libmcrypt-dev
pecl install mcrypt-1.0.1

bash -c "echo extension=/usr/lib/php/20170718/mcrypt.so > /etc/php/7.2/cli/conf.d/mcrypt.ini"
php -i | grep "mcrypt"

Encrypt/decrypt with XOR in PHP
https://stackoverflow.com/questions/14673551/encrypt-decrypt-with-xor-in-php


2018-04-27 《程序员的职业素养 - The Clean Coder》
https://blog.csdn.net/sunzhongyuan888/article/details/80396825

 好程序员与坏程序员的差别详细分析！
http://www.sohu.com/a/223781695_413876
 
 什么样的人当不好程序员？
 https://36kr.com/p/5042433.html
 
 一个十几年程序员给所有新老程序员的忠告
 https://www.jianshu.com/p/57fd54974d71
 
 好的程序员和差的程序员
 https://blog.csdn.net/dutao53486944/article/details/20009039
 
 金牌员工下班前必做的6件事
 https://jingyan.baidu.com/article/bad08e1ea2e3a709c9512168.html
 lib
 
 # 格式化大硬盘，大于 4 G

    # 查看硬盘数
    fdisk -l
    
    # 用 parted 进行大分区
    parted /dev/vdb
        mklable
            gpt
        print
        mkpart primary 0 -1   # 0表示分区的开始  -1表示分区的结尾  意思是划分整个硬盘空间为主分区        
        
    # 查看分区结果
    fdisk -l
    
    # 格式化
    mkfs.ext4 -T largefile /dev/vdb1
    
    # 挂载
    mkdir /data2
    mount -t ext4 /dev/vdb1 /data
    
    # 设置重启后也生效
    echo '/dev/vdb1   /data  ext4    defaults    0   0'  >>/etc/fstab
    
ganglia 不显示堆叠图
Missing stacked graphs on new ganglia-web 3.5.3 install
https://sourceforge.net/p/ganglia/mailman/ganglia-general/thread/alpine.DEB.2.02.1209191232500.4417@vukovar/
rm /var/lib/ganglia-web/conf/ganglia_metrics.cache    

Python Encrypting with PyCrypto AES
https://stackoverflow.com/questions/14179784/python-encrypting-with-pycrypto-aes

推荐 ：一文读懂最大似然估计(附R代码)
http://www.ijiandao.com/2b/baijia/171559.html

数据科学家应知必会的6种常见概率分布
https://blog.csdn.net/kicilove/article/details/78655856

通过阿里云API(php)搭建秒级DDNS动态域名解析
https://www.zimrilink.com/share/aliddns.html




代码的好味道和坏味道之22种坏味道
https://blog.csdn.net/zxh19800626/article/details/84781597

谈谈我们公司如何做Code Review
https://blog.csdn.net/zxh19800626/article/details/84995166#comments


office 文件预览
https://products.office.com/zh-cn/office-online/documents-spreadsheets-presentations-office-online?rtc=1
https://products.office.com/en-us/office-online/documents-spreadsheets-presentations-office-online?rtc=1
libreoffice


ssh2,node-rdp,xterm,monaco-editor,socket.io

请问感冒的分类有哪几种?
http://www.bu-shen.com/shen-usnyhbbn.htm

Web code editor with Monaco code editor, JQuery file tree browser , nodejs
https://github.com/moorthi07/marscode

Add close button to jquery ui tabs?
https://stackoverflow.com/questions/14357614/add-close-button-to-jquery-ui-tabs
http://jqueryui.com/tabs/#manipulation

How can I get file extensions with JavaScript?
https://stackoverflow.com/questions/190852/how-can-i-get-file-extensions-with-javascript

Linux中profile、bashrc、bash_profile之间的区别和联系
https://blog.csdn.net/m0_37739193/article/details/72638074

linux中CentOS、Ubuntu、Debian三个版本系统 差别
https://www.cnblogs.com/baichuanhuihai/p/8056976.html

代码规范 : 表驱动法(if switch 真讨厌)
https://blog.csdn.net/qq_22555107/article/details/78884261

umeditor储存型xss漏洞 
https://www.yuag.org/2017/09/19/ueditor%E5%82%A8%E5%AD%98%E5%9E%8Bxss%E6%BC%8F%E6%B4%9E/  

根据白名单过滤 HTML(防止 XSS 攻击)
https://github.com/leizongmin/js-xss/blob/master/README.zh.md


jQuery的.bind() .live() .delegate()和.on()之间的区别
https://blog.csdn.net/weixin_38840741/article/details/80272203

Connecting to remote SSH server (via Node.js/html5 console)
https://stackoverflow.com/questions/38689707/connecting-to-remote-ssh-server-via-node-js-html5-console

C 语言经典100例
http://www.runoob.com/cprogramming/c-100-examples.html
C 语言实例
http://www.runoob.com/cprogramming/c-examples.html

jqueryfiletree/jqueryfiletree
https://github.com/jqueryfiletree/jqueryfiletree

clean-code-javascript
https://github.com/ryanmcdermott/clean-code-javascript
clean-code-php
https://github.com/jupeter/clean-code-php


PHP 代码简洁之道 （ PHP Clean Code）
https://zhuanlan.zhihu.com/p/33451652

软件开发名言：kiss, dry, 单一指责，不要过早优化，破窗理论

[pyp-w3] Simple Database System
https://github.com/rmotr-group-projects/pyp-w3-gw-simple-database-system
3. Practical: A Simple Database
http://www.gigamonkeys.com/book/practical-a-simple-database.html
Practical Common Lisp
http://www.gigamonkeys.com/book/
Practical Common Lisp【个人翻译版】
https://www.douban.com/note/178148141/


需要将php.ini中的配置指令做如下修改： 

1. error_reporting = E_ALL ;将会向PHP报告发生的每个错误 

2. display_errors = Off ;不显示满足上条 指令所定义规则的所有错误报告 

3. log_errors = On ;决定日志语句记录的位置 

4. log_errors_max_len = 1024 ;设置每个日志项的最大长度 

5. error_log = /usr/local/error.log ;指定产生的 错误报告写入的日志文件位置 

在CentOS上安装Python3的三种方法
https://www.centos.bz/2018/01/%E5%9C%A8centos%E4%B8%8A%E5%AE%89%E8%A3%85python3%E7%9A%84%E4%B8%89%E7%A7%8D%E6%96%B9%E6%B3%95/
How do I check whether a file exists without exceptions?
https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.NOTSET, stream=sys.stdout)

Logging, StreamHandler and standard streams
https://stackoverflow.com/questions/1383254/logging-streamhandler-and-standard-streams

Insert an item into sorted list in Python
https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python

二分查找
Inserting and removing into/from sorted list in Python
https://stackoverflow.com/questions/17348332/inserting-and-removing-into-from-sorted-list-in-python

Python 顾问
https://stackoverflow.com/users/100297/martijn-pieters

跳跃表
EFFICIENT RUNNING MEDIAN USING AN INDEXABLE SKIPLIST (PYTHON RECIPE)
http://code.activestate.com/recipes/576930-efficient-running-median-using-an-indexable-skipli/

Python 树
http://interactivepython.org/runestone/static/pythonds/Trees/SearchTreeImplementation.html
https://pythonspot.com/python-tree/

Problem Solving with Algorithms and Data Structures using Python
http://interactivepython.org/runestone/static/pythonds/index.html
使用算法和数据结构解决问题
https://facert.gitbooks.io/python-data-structure-cn/content/
Learn to program, one byte at a time.
http://pythonschool.net/


在Ubuntu上使用golang
https://blog.csdn.net/miracle33/article/details/82875229

Replace a string in shell script using a variable
https://stackoverflow.com/questions/3306007/replace-a-string-in-shell-script-using-a-variable


500 Lines or Less
https://github.com/aosabook/500lines
http://aosabook.org/blog/
http://www.wangfz.com/archives/1254


WLP4 Programming Language Specification
https://www.student.cs.uwaterloo.ca/~cs241/wlp4/WLP4.html


Ubuntu防火墙 UFW 设置
http://www.cnblogs.com/sxwailyc/archive/2010/07/10/1774909.html

How can I open a range of ports in ubuntu using (g)ufw
https://askubuntu.com/questions/7099/how-can-i-open-a-range-of-ports-in-ubuntu-using-gufw

ufw allow 11200:11299/tcp
ufw allow 11200:11299/udp

ufw allow from AAA.BBB.CCC.DDD/EE to any port 11200:11299 proto tcp
ufw allow from AAA.BBB.CCC.DDD/EE to any port 11200:11299 proto udp

1700页数学笔记火了！全程敲代码，速度飞快易搜索，硬核小哥教你上手LaTeX+Vim
https://zhuanlan.zhihu.com/p/60049290
从基本原理到梯度下降，小白都能看懂的神经网络教程
https://zhuanlan.zhihu.com/p/59385110

主成分分析PCA
http://www.cnblogs.com/zhangchaoyang/articles/2222048.html
sklearn中PCA的使用方法
https://www.jianshu.com/p/8642d5ea5389
在线PS图片编辑器
http://ps.xunjiepdf.com/


当数据分布比较分散（即数据在平均数附近波动较大）时，各个数据与平均数的差的平方和较大，方差就较大；当数据分布比较集中时，各个数据与平均数的差的平方和较小。因此方差越大，数据的波动越大；方差越小，数据的波动就越小。


Ubuntu安装Java8和Java9
https://www.cnblogs.com/woshimrf/p/ubuntu-install-java.html

sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
sudo apt install oracle-java8-set-default

What is the Tomcat installation directory?
https://askubuntu.com/questions/135824/what-is-the-tomcat-installation-directory
Tomcat7 on Ubuntu14.04 html works but 404 error for servlets
https://stackoverflow.com/questions/32411364/tomcat7-on-ubuntu14-04-html-works-but-404-error-for-servlets

[Machine Learning & Algorithm] 随机森林（Random Forest）
http://www.cnblogs.com/maybe2030/p/4585705.html
[Machine Learning] 机器学习常见算法分类汇总
http://www.cnblogs.com/maybe2030/p/4665816.html#web
使用sklearn的cross_val_score进行交叉验证
https://blog.csdn.net/qq_36523839/article/details/80707678

直线回归方程：当两个变量x与y之间达到显著地线性相关关系时,应用最小二乘法原理确定一条最优直线的直线方程y=a+bx,这条回归直线与个相关点的距离比任何其他直线与相关点的距离都小,是最佳的理想直线.
回归截距a：表示直线在y轴上的截距,代表直线的起点.
回归系数b：表示直线的斜率,他的实际意义是说明x每变化一个单位时,影响y平均变动的数量.
即x每增加1单位,y变化b个单位.

https://blog.csdn.net/zxd1754771465/article/details/72896169
由方差、偏差、噪声、泛化误差的公式可以看出，偏差度量了模型预测的期望值与真实值之间的偏离程度，刻画了模型本身的拟合能力；方差度量了训练集的变动对预测结果的影响；噪声表达了能达到的期望误差的下界，刻画了学习问题本身的难度。

误差（包括训练误差，测试误差）=偏差+方差+噪声

经验误差与泛化误差、偏差与方差、欠拟合与过拟合、交叉验证
https://blog.csdn.net/zhihua_oba/article/details/78684257

偏差（Bias）与方差（Variance）
https://zhuanlan.zhihu.com/p/38853908

偏差，方差，训练误差，测试误差的区别
https://blog.csdn.net/xiaopihaierletian/article/details/68921609


## 机器学习

- 数据降维:
    - PCA
- 可视化：
    - 散点图：plt.scatter
- 描述性分析（describe）：mean，std，value_counts
- 数据标准化（normalize）：min-max，Z-score
- 划分训练集和测试集：train_test_split
- 调参：
    - 网格搜索：GridSearchCV
- 模型性能评估：
    - 指标：错误率，假阳性，假阴性，均方误差，残差图
    - 方法：
        - 交叉验证
- 算法：
    - K 近邻：KNN
    - 支持向量机：SVM
        - 线性支持向量机：LinearSVC
    - 随机森林：RF
    - 朴素贝叶斯（NB）
    - 线性回归：最小二乘，截距，回归系数
- 特征提取
    - 文本
        - 去除停止词
        - 词袋模型，CountVectorizer
        - 逆文档频率，TF-IDF，TfidfTransformer
- pipeline：vect->tfidf->svm

查看本机对外网网卡监听的所有端口
netstat -tnpl | awk '{print $4}' | grep -E '(0.0.0.0)|(:::)' |sed -r 's/(:::)|([0-9]+.[0-9]+.[0-9]+.[0-9]+:)//' | sort | uniq | sort -n



如果你是高端人才，你更需要的是“领悟力”，而不是“学习能力”
https://www.jianshu.com/p/fbfac3358f97

递归下降语法分析器实现过程
https://www.jianshu.com/p/988ce6fd0e67


How can jQuery deferred be used?
https://stackoverflow.com/questions/4869609/how-can-jquery-deferred-be-used

使用指定的列绘制散点图
Plot 2-dimensional NumPy array using specific columns
https://stackoverflow.com/questions/13634349/plot-2-dimensional-numpy-array-using-specific-columns

Matplotlib绘制高斯核概率分布(Gaussian kernel density estimator)等实战
https://blog.csdn.net/Wayne_Mai/article/details/80465478

python随机数组，高斯噪声，多项式函数
https://blog.csdn.net/ikerpeng/article/details/20703851

机器学习之朴素贝叶斯及高斯判别分析
https://www.cnblogs.com/zyber/p/6490663.html

常见的判别模型有线性回归、对数回归、线性判别分析、支持向量机、boosting、条件随机场、神经网络等。

常见的生产模型有隐马尔科夫模型、朴素贝叶斯模型、高斯混合模型、LDA、Restricted Boltzmann Machine等。


高斯分布可视化
Visualizing the bivariate Gaussian distribution
https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/


删除分支
git branch -d newmaster
git push --delete origin newmaster

打印日志 (log) 是比单步跟踪 (debugger) 更好的 Python 排错手段吗？
https://www.zhihu.com/question/20626825


Ubuntu 开机 PCIe Bus Error: severity=Corrected, type=Physical Layer, id=00eb(Receiver ID)
https://askubuntu.com/questions/863150/pcie-bus-error-severity-corrected-type-physical-layer-id-00e5receiver-id
sudo gedit /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pci=noaer"
sudo update-grub

在  /boot/grub/menu.lst  文件中增加  kopt=pcie_aspm=off
http://www.cnblogs.com/xianqingzh/archive/2013/11/29/3449075.html

“pci=nomsi,noaer” in “/etc/default/grub” : any downside?
https://askubuntu.com/questions/726100/pci-nomsi-noaer-in-etc-default-grub-any-downside


How to convert ISO8601 to Date format in php
https://stackoverflow.com/questions/22376415/how-to-convert-iso8601-to-date-format-in-php

Sortable, readable and standard time format for logs
https://stackoverflow.com/questions/17819834/sortable-readable-and-standard-time-format-for-logs

检测U盘拷贝速度

while true
do
        a=`du -s /mnt/labs|cut -f 1`
        sleep 1
        b=`du -s /mnt/labs|cut -f 1`
        echo $b-$a |bc
done


Katalon Studio：一款静候你使用的免费自动化测试工具
https://blog.csdn.net/huyanyan08/article/details/78753637
https://docs.katalon.com/katalon-studio/docs/spy-web-utility.html#capture-objects-using-web-object-spy

## 编译原理

https://github.com/kach/nearley
https://nearley.js.org/docs/getting-started
简单语言的 BNF
https://github.com/jxcjxcjzx/ezlang/blob/master/src/parser/grammar.ne
铁路图
http://dundalek.com/GrammKit/

小试牛刀-递归下降算法（3）
https://blog.csdn.net/lgh1700/article/details/78437563
https://blog.csdn.net/lgh1700/article/details/78450098
https://blog.csdn.net/lgh1700/article/details/78450116

重磅来袭 - Lua语法分析（1）
https://blog.csdn.net/lgh1700/article/details/78450159

C解释器
https://github.com/lotabout/write-a-C-interpreter
https://github.com/rswier/c4
https://github.com/rui314/8cc

https://www.jianshu.com/p/988ce6fd0e67

要实现一种语言的分析器，我们需要先写出这种语言的上下文无关文法(Context-free grammer)，我选择语言的文法如下：

    P -> S $

    S -> S ; S
    S -> id = EXP
    S -> print ( EXPS )

    EXPS -> EXP
    EXPS -> EXPS , EXP

    EXP -> int
    EXP -> id
    EXP -> EXP OP EXP
    EXP -> ( EXP )

    OP -> +
    OP -> -
    OP -> *
    OP -> /


递归下降分析器的构造方法非常简单，简单来说，就是为每一个非终结符写一个递归函数，函数中对该终结符可能转换成的所有情况的第一个token进行判断，做出对应的处理，例如，S的函数形式如下(要注意的是这个函数是错的，只是为了表现形式，下面会说明)：

void S(){
  switch(current_token){
    case S: 
      S(); eat(';'); S(); break;
    case id: 
      eat(id); eat('='); EXP(); break;
    case print:
      eat(print); eat('('); EXPS(); eat(')'); break;
    default:
      error_exit();
  }
}

消除左递归类似
    E -> E + T
    E -> T

的文法是左递归的，我们可以通过引入一个新的非终结符来消除它，结果如下：

    E -> T E'
    E' -> + T
    E' ->

上面展示的并不是一种模板，而是一种思想，我们的文法中，在非终结符EXP,S中也出现了左递归的情况，但是要复杂得多，你需要理解消除左递归的思想来解决它们。

最终的文法，在消除了EXP 和 S的左递归，提取了EXPS的左因子之后，文法如下：
    P -> S $

    S -> id = EXP S'
    S -> print ( EXP S ) S'
    S' -> ; S
    S' -> 

    EXPS -> EXP EXPS'
    EXPS' -> , EXPS
    EXPS' ->

    EXP -> ( EXP )
    EXP -> int EXP' 
    EXP -> id EXP'
    EXP' -> OP EXP
    EXP' ->

    OP -> +
    OP -> -
    OP -> *
    OP -> /


## 函数式编程



极客学院的产品是对标美国的Lynda。Lynda是美国一家垂直在线学习服务提供商，创办了17年，至今只有近100万用户，A轮融资中却募集资金1.03亿美元。

5天时间，亲测整理144个学习网站，多是免费资源！
https://baijiahao.baidu.com/s?id=1608132235565689872&wfr=spider&for=pc


对于教育产品，有几个数据是关键的：

付费用户数

续费用户数

完课率

Q11：您觉得，一家做内容的公司，比如极客邦科技，从小做到大，促成成功的因素中，哪些是必不可少的？


Kevin：还是上面所说的，除此之外再补充的话，可能一是对于目标是否强烈，是否有统一的愿景和使命。二是坚持，跟 InfoQ 一起发展的社区，有很多中途就放弃了，很可惜。


其实这个领域的人群并不算少。参考 GitHub 官方数据及年度报告、IDC 报告和中国统计年鉴，预测目前国内 IT 技术从业者应该在 700 万到 1000 万左右。假设年增长率为 30%，同时考虑编程教育进一步普及，未来可是人人都需要懂一点编程的时代，我们预计，在 2022 年这一人群将达到 3000 万人，甚至更多。

在这个过程里，产品的迭代速度和更新频率会起到决定性的作用。比如，如何更好的呈现音频和图文的展示，如何有别于传统的视频课，如何建立作者和读者的连接，如何推荐内容，如何实现高效的订阅流程，如何完成用户的学习闭环，如何承载更多的内容形式。除此之外，还要考虑产品的自传播特性，如何让用户更好的展示自我，如何让用户把产品和内容推荐给朋友等等。

产品发布仿佛就在昨天，一年已经过去了，未来我们会做什么呢？

1、继续打造和完善 IT 技能知识图谱，构建用户学习路径
2、增加用户自己的学习模块，引入已购和学习轨迹
3、增加社区属性，为用户建立更广泛的连接
4、建立用户技术信用等级
5、连接技术人和企业，构建职业通道
6、形成平台类产品……

Docker: 限制容器可用的 CPU
https://www.cnblogs.com/sparkdev/p/8052522.html
apt-get install stress && stress -c 4

cpu_limit (int) – CPU limit in units of 10^9 CPU shares.
cpu_reservation (int) – CPU reservation in units of 10^9 CPU shares.

每个 10^9 表示一个核，比如 cpu_limit = 2 * 10 ^ 9 就是每个容器最大 CPU 使用率限制在 2 个核，
但如果你宿主机有 10 个核的话，开 5 个容器照样把宿主机跑死，这时候可以设置 cpu_reservation，
比如设置 cpu_reservation  = 2 * 10 ^ 9，则在开第 5 个容器时，会提示 no suitable node 错误，因为
没有满足条件的节点可以运行此容器了。

docker service create 只支持上面两个参数，不支持 --cpu-shares 和 --cpuset-cpus 参数，所以限制了 
cpu_limit 后，即使宿主机 CPU 很空闲，容器也只能使用有限的 CPU，而不能实现在 CPU 资源紧张时指定
一组容器共享一组 CPU 的效果。

还是达不到想要的效果，现在每个容器限制最大 CPU 核数比较少的时候，即使宿主机 CPU 比较空闲，打开 eclipse 也非常慢，不能有效利用宿主机空闲 CPU，但如果限制每个容器可以用很多核 CPU 的话，一个宿主机可启动的容器数就会限制，最后如果对 CPU 不做任何限制的话，有可能一个容器就可以把整个宿主机跑死，甚至让宿主机 SSH 都连不上。


宿主机 4 核

container1(占坑用)， cpu_limit=2*1000000000，此时 CPU 消耗 0
container2，设置 cpu_reservation=2 * 1000 * 1000 * 1000，执行 stress -c 4，此时宿主机 CPU 消耗 2 个核
container2，设置 cpu_reservation=2 * 1000 * 1000 * 1000，执行 stress -c 4，此时宿主机已跑满

所以 container1 根本没起到占坑的作用，感觉设置cpu_reserver 其实就是设置 cpu_limit，值相当于 CPU总核数减去 cpu_reserver,
 而且只针对本容器，多个容器都跑起来的话，压根没起到位系统预留CPU的效果

Can comments be used in JSON?
https://stackoverflow.com/questions/244777/can-comments-be-used-in-json

Tips for using Docker Swarm mode in production
https://rock-it.pl/tips-for-using-docker-swarm-mode-in-production/


## 国内镜像：

JDK 下载
https://repo.huaweicloud.com/java/jdk/

php

        composer config -g repo.packagist composer https://packagist.phpcomposer.com

docker

python

apt-get

nodejs

maven

    sudo vi /etc/maven/settings.xml

    <mirror>
        <id>nexus-aliyun</id>
        <name>Nexus aliyun</name>
        <url>http://maven.aliyun.com/nexus/content/groups/public/</url>
        <mirrorOf>central</mirrorOf>
    </mirror>
    
    <localRepository>D:/.m2/repository</localRepository>



preview doc/ppt/xls online via unoconv.
https://github.com/nladuo/online_office_viewer

基于unoconv的在线office预览
https://segmentfault.com/a/1190000008995902

PDF、PPT、Excel、Word、视频等格式文件在线预览
https://segmentfault.com/a/1190000012164793

将Office Web App整合到自己的项目中（WOPI Host 开发）
https://blog.csdn.net/steveyg/article/details/52814803

office online server2016详细安装步骤及问题总结
https://blog.csdn.net/q386815991/article/details/81705128

PHP-Resque使用
http://www.pangxieke.com/php/php-resque.html

pear 是PHP的官司方开源类库,功能强大方便,好多人不知从何入手使用PEAR,网络搜索,多数为go-pear.php安装之类,
pear的使用

1.cmd模式进入php目录

2. 执行  pear ,检查是否能用该命令

3.pear list 显示已经安装的包

4. pear install **   安装新包  **

5.举例:   pear install html_quickform(安装快速表单),安装完成后,即可在pear目录下发现html/quickform

6. 在源码中,引用对应目录下的class文件,进行使用.


关于php的pear 出现Cannot use result of built-in function in write context in /usr/share/pear/Archive/Tar.php on line 639的问题
进入文件位置，找到639行

$v_att_list = &  func_get_args();

改变为

$v_att_list = func_get_args();

Failed opening 'PHP/CodeSniffer/autoload.php' for inclusion (include_path='.:') 
https://github.com/squizlabs/PHP_CodeSniffer/issues/1441


HTML 无障碍规范
http://squizlabs.github.io/HTML_CodeSniffer/Standards/Section508/#

git replacing LF with CRLF
https://stackoverflow.com/questions/1967370/git-replacing-lf-with-crlf

markdown 转脑图, freemind
https://github.com/Creased/markdown-to-mindmap

tornado vs wsgi(with gunicorn)
https://stackoverflow.com/questions/22062307/tornado-vs-wsgiwith-gunicorn


How do I iterate over a range of numbers defined by variables in Bash?
https://stackoverflow.com/questions/169511/how-do-i-iterate-over-a-range-of-numbers-defined-by-variables-in-bash

for i in {1..5}; do echo $i; done
for i in $(seq 1 $END); do echo $i; done

Full Example Configuration
https://www.nginx.com/resources/wiki/start/topics/examples/full/

ngx_http_proxy_module 模块
http://tengine.taobao.org/nginx_docs/cn/docs/http/ngx_http_proxy_module.html#proxy_pass_header
How to Configure Nginx as Reverse Proxy for WebSocket
https://www.tutorialspoint.com/articles/how-to-configure-nginx-as-reverse-proxy-for-websocket

How to Set Up a Redis Server as a Session Handler for PHP on Ubuntu 14.04
https://www.digitalocean.com/community/tutorials/how-to-set-up-a-redis-server-as-a-session-handler-for-php-on-ubuntu-14-04

Nginx常用变量大全
http://timd.cn/nginx/varindex/

车牌识别LPR（二）-- 车牌特征及难点
https://www.cnblogs.com/silence-hust/p/4191700.html

如何提高和优化Lucene搜索速度
https://blog.csdn.net/uniorg/article/details/6127365

Lucene底层原理和优化经验分享：Lucene简介和索引原理
https://blog.csdn.net/mine_song/article/details/56476247
Lucene底层原理和优化经验分享(2)-Lucene优化经验总结
https://blog.csdn.net/njpjsoftdev/article/details/54133548

Lucene7.2.1系列（一）快速入门
https://blog.csdn.net/qq_34337272/article/details/79764305

Lucene创建和查询索引库的HelloWorld(含详细注释)
https://blog.csdn.net/xzm_rainbow/article/details/18948597

Docker db container running. Another process with pid <id> is using unix socket file
https://stackoverflow.com/questions/36103721/docker-db-container-running-another-process-with-pid-id-is-using-unix-socket

容器里的 mysql 重启后启动不起来，可能是如下原因
rm -f /var/run/mysqld/*
/etc/init.d/mysql start &


Docker 容器初始化数据库

How can I initialize a MySQL database with schema in a Docker container?
https://stackoverflow.com/questions/29145370/how-can-i-initialize-a-mysql-database-with-schema-in-a-docker-container/29150538#29150538

RUN mysqld_safe & until mysqladmin ping; do sleep 1; done && \
    mysql -uroot -e "CREATE DATABASE somedb;" && \
    mysql -uroot -e "CREATE USER 'someuser'@'localhost' IDENTIFIED BY 'somepassword';" && \
    mysql -uroot -e "GRANT ALL PRIVILEGES ON somedb.* TO 'someuser'@'localhost';"
    
[ "$(ls -A /var/lib/mysql)" ] && echo "Running with existing database in /var/lib/mysql" || ( echo 'Populate initial db'; tar xpzvf default_mysql.tar.gz )    


Git 远程仓库搭建
https://www.cnblogs.com/whthomas/p/3643974.html

如何排查can not find symbol的编译错误
https://blog.csdn.net/dory2100/article/details/54890476

如何设置maven的local repository目录
https://www.cnblogs.com/jtlgb/p/5951666.html

## java 环境

- [classpath、path、JAVA_HOME的作用及JAVA环境变量配置](https://www.cnblogs.com/xwdreamer/archive/2010/09/08/2297098.html)
- [Google Guava官方教程（中文版）](http://ifeve.com/google-guava/)
- [Windows7下Maven环境搭建及其使用](http://blog.csdn.net/xuexiaoxu1990/article/details/52882664)

安装 JDK，设置环境变量

    CLASSPATH= .;%JAVA_HOME%/lib/dt.jar;%JAVA_HOME%/lib/tools.jar
    JAVA_HOME = C:/Program Files/Java/jdk1.5.0
    PATH  = %JAVA_HOME%/bin;%JAVA_HOME%/jre/bin

安装 maven 并配置

    # 增加镜像，设置本地缓存路径
    vi /etc/maven/settings.xml

    <mirror>
        <id>nexus-aliyun</id>
        <name>Nexus aliyun</name>
        <url>http://maven.aliyun.com/nexus/content/groups/public/</url>
        <mirrorOf>central</mirrorOf>
    </mirror>
    
    <localRepository>D:/.m2/repository</localRepository>

    # 搜索包
    https://maven.aliyun.com/mvn/search

    # 常用命令
    mvn clean compile
    mvn clean package
    
    mvn clean generate-sources package -DskipTests
    mvn install # 刚少这一步， readme 里没写
    cd jplag
    mvn clean generate-sources assembly:assembly
    java -jar target/jplag-2.12.1-SNAPSHOT-jar-with-dependencies.jar

编译 java 文件，在 windows git bash

    javac -encoding UTF-8 \
        -cp /d/.m2/repository/org/apache/lucene/lucene-core/7.1.0/lucene-core-7.1.0.jar \
        -d ./ \
        /d/lucene/demo/helloworld/HelloWorldLucene.java
  
执行 class 文件

    java -cp './;D:\.m2\repository\org\apache\lucene\lucene-core\7.1.0\lucene-core-7.1.0.jar' \
        lucene.demo.helloworld.HelloWorldLucene
    
性能调试

    # 每隔 2000 ms 输出 gc 情况，一共输出 20 次
    jstat -gc 10344 2000 20
    # 抓内存 dump，可以用 MAT(Memory Anlysis Tool) 打开
    jmap -dump:live,format=b,file=dump.hprof 10344
    # 查看 Finalizer 队列
    jmap -finalizerinfo 10344
    # 查看堆分布信息
    jmap -heap 10344
    # 查看对象分布
    jmap -histo:live 10344 | more
    # 图形化 dump 文件，打开 localhost:7000，重点关注 Heap Histogram
    jhat -J-Xmx512m dump.hprof
    # 查看调用栈
    jstack -l 10344 | more
    # 查看 java 参数
    jinfo -flags 10344    


关于怎么解决java.lang.NoClassDefFoundError错误
https://www.cnblogs.com/xyhz0310/p/6803950.html

Java中设置classpath、path、JAVA_HOME的作用
https://www.cnblogs.com/echomyecho/p/3334617.html


CLASSPATH环境变量。作用是指定类搜索路径，要使用已经编写好的类，前提当然是能够找到它们了，JVM就是通过CLASSPATH来寻找类的.class文件。我们需要把jdk安装目录下的lib子目录中的dt.jar和tools.jar设置到CLASSPATH中，当然，当前目录“.”也必须加入到该变量中。

 javac -c  路径 （可以指定class文件存放目录）

 java -cp  路径  （可以指定要执行的class目录）
 
 run main class of Maven project [duplicate]
 https://stackoverflow.com/questions/9846046/run-main-class-of-maven-project
 How to execute a java .class from the command line
 https://stackoverflow.com/questions/1279542/how-to-execute-a-java-class-from-the-command-line
 
 jstat命令使用
 https://www.cnblogs.com/lizhonghua34/p/7307139.html
 
 jvm系列(四):jvm调优-命令大全（jps jstat jmap jhat jstack jinfo）
 https://www.cnblogs.com/ityouknow/p/5714703.html
 
 JAVAC 命令使用方法
 https://www.cnblogs.com/mfryf/p/3159084.html
 
 Shallow heap & Retained heap
 https://bjyzxxds.iteye.com/blog/1532937
 
 
 ## git
 
删除远程分支

git push origin --delete dev
 
git config --global core.editor vim
git config --global core.autocrlf true
git config --global core.safecrlf true

git 不显示中文
git config --global core.quotepath false

    git pull tencent master --allow-unrelated-oo
    git clone --depth 1 -b v3/master --single-branch https://github.com/SpiderLabs/ModSecurity
    
Git diff 结果解读
https://www.cnblogs.com/alfayed/p/4682780.html

How to undo 'git reset'?
https://stackoverflow.com/questions/2510276/how-to-undo-git-reset

Git撤销&回滚操作(git reset 和 get revert)
https://blog.csdn.net/asoar/article/details/84111841

如何在 Git 里撤销(几乎)任何操作
https://www.cnblogs.com/allencelee/p/5603914.html

Change old commit message on Git
https://stackoverflow.com/questions/1884474/change-old-commit-message-on-git

git merge --no-ff是什么意思
https://segmentfault.com/q/1010000002477106

Git查看各个branch之间的关系图
https://blog.csdn.net/qq_32106647/article/details/79893475

关于git rebase 后 push rejected
https://blog.csdn.net/bigmarco/article/details/7842642

团队开发里频繁使用 git rebase 来保持树的整洁好吗?
https://segmentfault.com/q/1010000000430041

Git push rejected after feature branch rebase
https://stackoverflow.com/questions/8939977/git-push-rejected-after-feature-branch-rebase


代码查重，JPLAG
https://github.com/jplag/jplag

上海交大程序代码抄袭检测系统(重要参考)
https://wenku.baidu.com/view/5f1f5d07eff9aef8941e062c.html


代码查重，相似性，抄袭检测
https://stackoverflow.com/questions/10912349/similar-code-detector
https://github.com/xsgordon/duplo-fork
https://github.com/jplag/jplag
https://pmd.sourceforge.io/pmd-5.0.0/
http://www.drdobbs.com/architecture-and-design/detecting-source-code-plagiarism/184405734

    

java -jar jplag.jar -l java17 -r /var/www/html/result -s exercise1/





JPlag 重复代码段颜色不一致问题的解决方法。
https://blog.csdn.net/zhangchao19890805/article/details/79758617

JPlag: finding plagiarisms among a set of programs
https://www.researchgate.net/publication/36451198_JPlag_finding_plagiarisms_among_a_set_of_programs


maven 打包时，jar should not point at files within the project directory 问题解决
https://blog.csdn.net/ke7in1314/article/details/79152461

run main class of Maven project [duplicate]
https://stackoverflow.com/questions/9846046/run-main-class-of-maven-project

Type hinting for properties in PHP 7?
https://stackoverflow.com/questions/37254695/type-hinting-for-properties-in-php-7

PHPDoc type hinting for array of objects?
https://stackoverflow.com/questions/778564/phpdoc-type-hinting-for-array-of-objects

PHP 7 类型提示
https://zhuanlan.zhihu.com/p/27694633

## 数据库做队列

Best way to implement concurrent table based queue
https://dba.stackexchange.com/questions/98311/best-way-to-implement-concurrent-table-based-queue
http://johnbokma.com/mexit/2012/03/15/job-queue-in-mysql.html
https://www.engineyard.com/blog/5-subtle-ways-youre-using-mysql-as-a-queue-and-why-itll-bite-you
https://stackoverflow.com/questions/423111/whats-the-best-way-of-implementing-a-messaging-queue-table-in-mysql
https://stackoverflow.com/questions/2177880/using-a-database-table-as-a-queue

-- 建表
CREATE TABLE IF NOT EXISTS `queue`(
`id` INT(11) NOT NULL AUTO_INCREMENT,
`data` VARCHAR(256) NOT NULL,
`status` VARCHAR(16) NOT NULL DEFAULT 'todo',
PRIMARY KEY (`id`)
) ENGINE=INNODB ;
    
-- 入队    
INSERT INTO queue(DATA) VALUES(MD5(NOW()));

-- 出队
START TRANSACTION;
SELECT @id:=id AS id, `data`, `status` FROM queue WHERE STATUS = 'todo' ORDER BY id ASC LIMIT 1 FOR UPDATE;
UPDATE queue SET STATUS = 'done' WHERE id = @id AND STATUS = 'todo'; 
COMMIT;

以太坊DAPP开发（一）——环境搭建
https://www.jianshu.com/p/f72765389073
https://yq.aliyun.com/articles/554837
https://blog.csdn.net/huangshulang1234/article/details/79374085
https://solidity-cn.readthedocs.io/zh/develop/
https://solidity-cn.readthedocs.io/zh/develop/introduction-to-smart-contracts.html#simple-smart-contract
https://www.jianshu.com/p/eb60cb20db2f?from=timeline
https://github.com/ethereum/meteor-dapp-wallet/blob/develop/Wallet.sol
https://vitalik.ca/general/2017/06/09/sales.html
https://www.ethereum.org/crowdsale
https://github.com/ethereum/dapp-bin
https://github.com/ethereum/wiki/wiki/Dapp-using-Meteor
https://github.com/ethereum/mist

# 区块链

- 概念
    - 智能合约
    - 超级湛贲
- 技术    
    - solidity：http://www.tryblockchain.org/
    - Truffle
- 书
    - 区块链开发实战：以太坊关键技术与案例分析 https://item.jd.com/12380446.html
    
mysql主从配置（清晰的思路）
https://www.cnblogs.com/sustudy/p/4174189.html    

Google Fuchsia 对中国操作系统的启示 | 畅言
https://blog.csdn.net/csdnnews/article/details/80486873

测试金字塔
https://martinfowler.com/bliki/TestPyramid.html

配置mysql5.5主从服务器(转)
https://www.cnblogs.com/cchun/p/3712637.html

MariaDB 多主一从 搭建测试
https://www.cnblogs.com/zhoujinyi/p/4102984.html

传统复制模式 -> 多源复制的命令变化

    reset slave -> reset slave 'conn_erp',多个连接源名字
    start slave -> start slave 'connection_name' 或者start all slaves
    show slave status -> show slave 'conn_mall' status,或者 show all slaves status查看所有的slave连接状态
    sql_slave_skip_couter -> stop slave 'connection_name'，先指定连接源名称 set @@default_master_connection='connection_name';然后再set global sql_slave_skip_counter=1;最后start slave 'connection_name'
    多源复制环境下的replicate-... variables 变量问题
    在my.cnf replicate_ignore_db 前添加conn连接串前缀，比如 r1.replicate_ignore_db=ignore_database若不加前缀，就是忽略所有同名的数据库，其他变量类推。       

MySQL（MariaDB）的 SSL 加密复制
https://www.cnblogs.com/zhoujinyi/p/4191096.html
https://serverfault.com/questions/783861/enabling-ssl-in-mysql-when-using-windows-as-a-server-and-client

为MySQL设置安全的远程连接
https://blog.csdn.net/kai404/article/details/70249704

PHP - SSL certificate error: unable to get local issuer certificate
https://stackoverflow.com/questions/28858351/php-ssl-certificate-error-unable-to-get-local-issuer-certificate


1：在主服务器上创建CA证书：
openssl genrsa 2048 > ca-key.pem
openssl req -new -x509 -nodes -days 1000 -key ca-key.pem > ca-cert.pem

2：在主服务器上创建服务端的证书：
openssl req -newkey rsa:2048 -days 1000 -nodes -keyout server-key.pem > server-req.pem
openssl x509 -req -in server-req.pem -days 1000 -CA ca-cert.pem -CAkey ca-key.pem -set_serial 01 > server-cert.pem

3：在主服务器上创建客户端的证书：

openssl req -newkey rsa:2048 -days 1000 -nodes -keyout client-key.pem > client-req.pem
openssl x509 -req -in client-req.pem -days 1000 -CA ca-cert.pem -CAkey ca-key.pem -set_serial 01 > client-cert.pem

4：验证证书的有效性。     
openssl verify -CAfile ca-cert.pem server-cert.pem client-cert.pem

CA 证书，server 证书，client 证书的 Common name 不能相同，否则 verify 时会有报错
 
I think you missed this part of the instructions:

Whatever method you use to generate the certificate and key files, the Common Name value used for the server and client certificates/keys must each differ from the Common Name value used for the CA certificate. Otherwise, the certificate and key files will not work for servers compiled using OpenSSL.

When OpenSSL prompts you for the Common Name for each certificate, use different names. 

# mysql

查看索引： show index from table1
查看列：show columns from table1;   
# Mysql 主从同步

主库
    # 配置
    vi D:\xampp7\mysql\bin\mysql.ini
        server-id=1
        log-bin=mysql-bin
        binlog-do-db=test
        binlog-ignore-db=mysql
        
    # 启动
    D:\xampp7\mysql_start.bat
    
    # 配置主库
    D:\xampp7\mysql\bin\mysql.exe -h127.0.0.1 -P3306 -uroot
            
        GRANT REPLICATION SLAVE,FILE ON *.* TO 'mstest'@'localhost' IDENTIFIED BY '123456';
        show master status; 

从库
    # 复制文件，修改 my.ini 里的port 为 3307，以及路径 D:\xampp7\mysql2
    cp -r D:\xampp7\mysql  D:\xampp7\mysql2    
    
    # 配置文件
    vi D:\xampp7\mysql2\bin\my.ini
        server-id=2
        log-bin=mysql-bin
        replicate-do-db=test
        replicate-ignore-db=mysql
    
    # 启动
    D:\xampp7\mysql2\mysql_start.bat
    
    # 配置从库
    D:\xampp7\mysql\bin\mysql.exe -h127.0.0.1 -P3307 -uroot
    
        change master 'r1' to master_host='localhost', master_user='mstest', master_password='123456
        # 确保 Slave_IO_Running， Slave_SQL_Running 为 yes，Exec_Master_Log_Pos 值为最新
        show slave 'r1' status\G  

测试

    # 主，插入一条数据
    insert into test.queue(data) values(md5(now()));
    # 从，查看是否同步成功
    select count(*) from test.queue;
    
    
OpenSSL - error 18 at 0 depth lookup:self signed certificate
https://stackoverflow.com/questions/19726138/openssl-error-18-at-0-depth-lookupself-signed-certificate    


测试代码

    # 纯数字
    1
    100
    -1
    -100
    
    # 基本加减乘除
    1 + 1
    2 - 1
    2 * 2
    4 / 2
    3 % 2
    
    # 加减混合，乘除混合
    1 + 2 - 3
    2 * 3 / 2
    
    # 四则混合
    1 + 2 * 3 + 4
    
    # 带括号
    (1 + 2) * (3 + 4)
    
    # 变量及赋值
    var x = 2;
    x + 1;
    x + 1 * 2;
    (x + 1) * 2;
    
    
    # 布尔值
    true
    false
    
    # 比较表达式
    1 < 2
    1 > 2
    1 == 2
    1 != 2
    1 >= 2
    1 <= 2
    
    # 分支语句
    var x = 3;
    if (x < 4) {
        x = x + 1;
    } else {
        x = x - 1;
    }
    x;

    # 循环语句
    var x = 0;
    while (x < 10) {
        x = x + 1;
    }
    x;
    
    # 综合
    var x = 1;
    var y = 10;
    var z = 0;
    while (x < y) {
        if (x % 2 == 0) {
            z = z + x;
        }
        x = x + 1;
    }
    z;
    
k8s入门系列之介绍篇
http://www.cnblogs.com/xkops/p/6165565.html    

Is it possible to format ps RSS (memory) output to be more human friendly?
https://superuser.com/questions/665030/is-it-possible-to-format-ps-rss-memory-output-to-be-more-human-friendly
ps -eo pmem,comm,pid,maj_flt,min_flt,rss,vsz --sort -rss | numfmt --header --to=iec --field 4-5 | numfmt --header --from-unit=1024 --to=iec --field 6-7 | column -t | head

linux ps
10 basic examples of Linux ps command
https://www.binarytides.com/linux-ps-command/


只查看该进程：ps -ef | grep 11345
查看该进程打开的文件：lsof -p 11345
查看内存分配：lcat /proc/11345/maps
查看堆栈：pstack 11345
查看发出的系统调用:strace -p 11345
查看调用库函数:ltrace -p 11345

# 沟通 注意事项

- 倾听：尊重对方，不着急给建议
- 引导：多提问，让对方想答案
- 共情：换位思考，体会对方情绪，理解对方立场

选择 java 版本
sudo update-alternatives --config java

普通云盘，<1000 IOPS
高效云盘，1800 IOPS
SSD 云盘，7800 IOPS
ESSD 云盘，11800 IOPS
本地 SSD 1 ，17160 IOPS
本地 SSD 2，147510 IOPS
本地 SSD 3，240000 IOPS

云主机，4 核 16 G，100G 本地 SSD * 2，17160 IOPS，5912.40 元一年，使用高效云盘价格接近，但只有 1800 IOPS
云数据库，4 核 16 G，500 G 本地 SSD，4500 IOPS，一主一备高可用，20145 元一年
云 Redis，1 核 16 G（Redis 单线程），双击热备，硬盘持久化，保障高性能和可靠性，13260 元一年
云 Elasticsearch，4 核 16 G，3 节点集群，SSD 云硬盘，单节点 100G，28458 元一年

MySQL 最合适，架构复杂度低，调优和故障排查成熟，性能和容量的可扩展性都比较强，
不过数据量增大后，MySQL方案的性价比下降的厉害，8 核 32G 的云数据库一年 4.5w，8 核 32G 的云主机一年 1w


块存储性能：IOPS，吞吐量，访问时延
https://help.aliyun.com/document_detail/25382.html?spm=5176.ecsbuyv3.storage.2.1dbf3675BX54O8
https://help.aliyun.com/document_detail/63138.html?spm=a2c4g.11186623.2.17.51ca59a4c9yro6#concept-g3w-qzv-tdb

用Elasticsearch构建电商搜索平台，一个极有代表性的基础技术架构和算法实践案例
https://blog.csdn.net/jek123456/article/details/54562158
https://blog.csdn.net/qq_23864697/article/details/79950102
https://www.jianshu.com/p/3f4cf830accb

OKR与KPI管理的区别与联系
http://www.zhenrencai.cn/essay15


CentOS 7安装MariaDB 10详解以及相关配置
https://www.linuxidc.com/Linux/2018-03/151403.htm

利用requirejs实现vue的模块化开发
http://www.cnblogs.com/anrainie/p/8521085.html
https://segmentfault.com/a/1190000015669251

入门 Webpack，看这篇就够了
https://segmentfault.com/a/1190000006178770

PHP 内存泄漏分析定位
https://www.iteye.com/news/32726

$ php -i |grep memory  
memory_limit => 1024M => 1024M //php脚本执行最大可使用内存  
$php -i |grep max  
max_execution_time => 0 => 0 //最大执行时间，脚本默认为0不限制，web请求默认30s  
max_file_uploads => 20 => 20 //一个表单里最大上传文件数量  
max_input_nesting_level => 64 => 64 //一个表单里数据最大数组深度层数  
max_input_time => -1 => -1 //php从接收请求开始处理数据后的超时时间  
max_input_vars => 1000 => 1000 //一个表单（包括get、post、cookie的所有数据）最多提交1000个字段  
post_max_size => 8M => 8M //一次post请求最多提交8M数据  
upload_max_filesize => 2M => 2M //一个可上传的文件最大不超过2M  

Php-fpm.conf代码
pm = dynamic //仅dynamic模式下以下参数生效  
pm.max_children = 10 //最大子进程数  
pm.start_servers = 3 //启动时启动子进程数  
pm.min_spare_servers = 2 //最小空闲进程数，不够了启动更多进程  
pm.max_spare_servers = 5 //最大空闲进程数，超过了结束一些进程  
pm.max_requests = 500 //最大请求数，注意这个参数是一个php-fpm如果处理了500个请求后会自己重启一下，可以避免一些三方扩展的内存泄露问题  

一个 php-fpm 进程按 30MB 内存算，50 个 php-fpm 进程就需要 1500MB 内存，这里需要简单估算一下在负载最重的情况下所有 php-fpm 进程都启动后是否会把系统内存耗尽。 

遇到了内存泄露时先观察是程序本身内存不足还是外部资源导致，然后搞清楚程序运行中用到了哪些资源：写入磁盘日志、连接数据库 SQL 查询、发送 Curl 请求、 Socket 通信等， I/O 操作必然会用到内存，如果这些地方都没有发生明显的内存泄露，检查哪里处理大量数据没有及时释放资源，如果是 php 5.3 以下版本还需考虑循环引用的问题。多了解一些 Linux 下的分析辅助工具，解决问题时可以事半功倍。 

PHP 透明链路跟踪 Molten
https://github.com/chuan-yun/Molten/blob/master/README_ZH.md

[PHP] - 性能加速 - 开启Opcache
https://blog.csdn.net/abcdocker/article/details/55505063

[opcache]
opcache.enable=1
opcache.memory_consumption=128
opcache.validate_timestamps=On
opcache.revalidate_freq=2

zend_extension="opcache.so"


apache虚拟主机跟虚拟目录的配置与区别
https://blog.csdn.net/qq_36431166/article/details/81199068

<IfModule dir_module>
    DirectoryIndex index.html index.htm index.php
    Alias /code "D:/helloworld/github/codesnip"
    <Directory D:/helloworld/github/codesnip>
        Options Indexes FollowSymLinks Includes ExecCGI  
        AllowOverride All  
        Require all granted  
         RewriteEngine on  
        # 如果请求的是真实存在的文件或目录，直接访问  
        RewriteCond %{REQUEST_FILENAME} !-f  
        RewriteCond %{REQUEST_FILENAME} !-d  
        # 如果请求的不是真实文件或目录，分发请求至 index.php  
        RewriteRule . index.php  
    </Directory>
</IfModule>


https://www.cnblogs.com/chenhuichao/p/8308993.html

How to make a jQuery plugin loadable with requirejs
https://stackoverflow.com/questions/10918063/how-to-make-a-jquery-plugin-loadable-with-requirejs

You only need to do EITHER

define(["jquery"], // Require jquery
       function($){
// Put here the plugin code. 
// No need to return anything as we are augmenting the jQuery object
});
at the end of jquery-cookie.js, OR

requirejs.config( {
    "shim": {
        "jquery-cookie"  : ["jquery"]
    }
} );

Vue中的路由：分页获取数据以及路由更新
https://www.jianshu.com/p/9fc772eceff4

nginx日志打印响应时间request_time和upstream_response_time
https://happyqing.iteye.com/blog/2384266
ngxtop控制台终端Nginx日志实时监控工具
http://baijiahao.baidu.com/s?id=1604148523720174524&wfr=spider&for=pc
采集并分析Nginx访问日志
https://www.alibabacloud.com/help/zh/doc-detail/56728.htm?spm=a2c63.p38356.b99.31.1fda3751nTFqrb
为 PHP 应用安装探针
https://help.aliyun.com/document_detail/102783.html?spm=a2c4g.11186623.2.23.674c787ciXMzjM#concept-102783-zh
常用的NGINX配置
https://github.com/lebinh/nginx-conf
WDCP手工安全检测和提高安全性设置
http://www.wdcping.cn/news/html/?384.html

JMeter的基本介绍和入门（1）
http://www.51testing.com/html/54/n-854654-2.html
Jmeter查看QPS和响应时间随着时间的变化曲线
https://www.cnblogs.com/SH-xuliang/p/9242603.html

nginx 502
Nginx With PHP FPM - Resource Temporarily Unavailable - 502 Error
https://serverfault.com/questions/884468/nginx-with-php-fpm-resource-temporarily-unavailable-502-error/884477

正确使用PHP的Opcache加速模块
https://www.jianshu.com/p/f089b6d19382

See CPU use history as a graph in a Linux shell
https://softwarerecs.stackexchange.com/questions/24903/see-cpu-use-history-as-a-graph-in-a-linux-shell

tload,s-tui

s-tui：在 Linux 中监控 CPU 温度、频率、功率和使用率的终端工具
https://zhuanlan.zhihu.com/p/55528584

# 压力测试 CheckList

- 服务器和jmeter不要在同一机器，最好是同一局域网
- 服务器收集 vmstat, top, ngxtop, mytop, php-fpm status, nginx status，tload ，s-tui，mysql 连接数等信息
    - mytop -uroot -ppassword
    - mysql -uroot -ppassword -e 'show status;' | grep -E 'Threads_'
- nginx worker 数等于CPU个数
- php-fpm 最大 workers 数要足够，
    - 假如 100 qps，latency 1s，php 一个 worker 同时只能处理一个请求，大概需要 100 个 worker
    - worker 数大约需要 qps * latency 个
    - 如果阻塞请求多再适当加大，尽量不要出现 502 应答，错误率高压测无意义
- 开启 opcache    
    - opcache.enable=1
    - opcache.memory_consumption=128
    - opcache.validate_timestamps=On
    - opcache.revalidate_freq=2
    - zend_extension="opcache.so"
- 虚拟用户数要缓慢上升，不要瞬间跑满
    - 如 200 个虚拟用户 20 秒跑满，设置 JMeter 线程组的 Ramp-Up 参数
- 关闭 Nginx Access 日志
    - access_log Off;
- 关闭 CI 日志
    - $config['log_threshold'] = 0;
- PHP 错误报告级别调为最高
    - ini_set('display_errors', 0); error_reporting(E_ERROR);
- 遇到 CPU 跑满，可单独对某次请求添加 ?XDEBUG_PROFILE=1 参数以记录 Profiler 日志，然后用 WinCacheGrind 分析
    - xdebug.profiler_enable=0
    - xdebug.profiler_output_dir="/data/logs/xdebug"
    - xdebug.profiler_enable_trigger=1

服务器监控可以使用 tmux 开启多个子窗口

- htop
- watch -d -n 1 curl  -s http://localhost/nginx_status
- watch -d -n 1 curl  -s http://localhost/php_fpm_status
- watch -d -n 1 "mysql -uroot -p -e 'show status;' | grep  -E 'Threads_'"
- mytop -uroot -p
- s-tui
- iostat -d -x 1 | grep -E 'Device|sda'


简单结论：
- php7.2 开启 opcache ，性能提升 5 倍
    - 开启前：106.65 qps，937.663 latency
    - 开启后：520.26 qps， 192.211 latency
- nginx 开启 https，性能降低 50%
    - 开启前：203.95 qps，490.306 lantency
    - 开启后： 101.69，983.417  lantency
- think php 开启调试，性能降低 30%
    - 开启前：507.84 qps，196.912 lantency
    - 开启后：333.81 qps，299.567 lantency

## 压力测试基准

- 环境：
    -  4核 8G 机器
    - ab -k -c100 -n20000
- 结果：
    - Nginx 空跑： 34718.85 qps，2.880 lantency
    - PHP 空跑，只有一个 die 语句：5002.84 qps，19.989 lantency
    - 无框架 PDO 从 DB 查询一条数据：2519.65 qps，39.688 lantency
    - TP3.2 空跑 Controller 里直接 die：1208.36 qps，82.757 lantency
    - TP3.2 从 DB 查询一条数据：703.18 qps，142.212 lantency

Ubuntu php安装xdebug
https://www.cnblogs.com/jingjingdidunhe/p/6955619.html

sudo apt-get install php-xdebug
find /usr -name xdebug.so

mkdir -p /data/logs/xdebug/
chmod o+w  /data/logs/xdebug/

vi /etc/php/7.2/fpm/php.ini


[xdebug]
zend_extension="/usr/lib/php/20180731/xdebug.so"
xdebug.remote_enable=1
xdebug.remote_handler=dbgp 
xdebug.remote_mode=req
xdebug.remote_host=127.0.0.1 
xdebug.remote_port=9000
xdebug.profiler_output_dir="/data/logs/xdebug"
;profiler功能的开关，默认值0，如果设为1，则每次请求都会生成一个性能报告文件。
xdebug.profiler_enable=0
;默认值也是0，如果设为1 则当我们的请求中包含XDEBUG_PROFILE参数时才会生成性能报告
文件。
;例如http://localhost/index.php?XDEBUG_PROFILE=1(当然我们必须关闭xdebug.profiler_enable)。
;使用该功能就捕获不到页面发送的ajax请求，如果需要捕获的话我们就可以使用xdebug.profiler_enable功能。
xdebug.profiler_enable_trigger=1
;生成的文件的名字，默认 cachegrind.out.%t.%p
xdebug.profiler_output_name='cachegrind.out.%t.%p'



WinCacheGrind是windows下的profile查看程序，使用起来感觉还不错，profile文件太大的话偶尔会崩溃。

KCachegrind是linux下的一个图形化profile查看工具，功能很强劲。

How to trigger XDebug profiler for a command line PHP script?
https://stackoverflow.com/questions/2288612/how-to-trigger-xdebug-profiler-for-a-command-line-php-script

php -d xdebug.profiler_enable=On script.php

PHP执行耗时优化
https://blog.csdn.net/u013632190/article/details/80884117

接下来分析 SQL 性能： 
1) 查看 MySQL 版本： show variables like "%version%"; 
低于 5.0.37 的没有该功能。

2) 查看 性能分析 开关状态： show variables like "%pro%"; 
profiling 变量默认为 OFF，开启后为 ON 
（不必担心该操作会影响性能，每次退出数据库后会自动关闭）

3) 开启/关闭 性能分析 开关： set profiling = 1;、set profiling = 0;

4) 尝试执行某个 SQL

5) 然后查看该 SQL 耗时：

show profiles; --总耗时
show profile for query 1; --查询指定编号SQL耗时
show profile all for query 1; --查询指定编号SQL性能相关的全部信息
1
2
3
利用本方法，再结合 EXPLAIN 进行针对性 SQL 优化。


ab 压力测试

cat ab.data
ame=chang&password=11111ok
ulimit -SHn 65536
ab -n 1 -c 1 -v 4   -p ab.data -T application/x-www-form-urlencoded  "http://xxx.xxx.cn:8002/account/login"

Apache Bench （ab）测试出现 Failed requests
出现 Failed requests 的原因主要有三种：

Connect：连接失败、连接中断
Length：响应的内容长度不一致
Exception：未知错误
出现 Length 错误原因：

每一次 HTTP request 接收的 HTML 长度不同，这在进行「动态网页」压力测试时是合理的

性能测试工具 wrk,ab,locust,Jmeter 压测结果比较
https://testerhome.com/topics/17068

disown 示例2（如果提交命令时未使用“&”将命令放入后台运行，可使用 CTRL-z 和“bg”将其放入后台，再使用“disown”）

让进程 后台运行
我们已经知道，如果事先在命令前加上 nohup 或者 setsid 就可以避免 HUP 信号的影响。但是如果我们未加任何处理就已经提交了命令，该如何补救才能让它避免 HUP 信号的影响呢？
[root@pvcent107 build]# cp -r testLargeFile largeFile2
 
[1]+  Stopped                 cp -i -r testLargeFile largeFile2
[root@pvcent107 build]# bg %1
[1]+ cp -i -r testLargeFile largeFile2 &
[root@pvcent107 build]# jobs
[1]+  Running                 cp -i -r testLargeFile largeFile2 &
[root@pvcent107 build]# disown -h %1
[root@pvcent107 build]# ps -ef |grep largeFile2
root      5790  5577  1 10:04 pts/3    00:00:00 cp -i -r testLargeFile largeFile2
root      5824  5577  0 10:05 pts/3    00:00:00 grep largeFile2
[root@pvcent107 build]#
 
 
 How to set JAVA_HOME in Linux for all users
 https://stackoverflow.com/questions/24641536/how-to-set-java-home-in-linux-for-all-users
 
 find /usr/lib/jvm/java-1.x.x-openjdk
vim /etc/profile

Prepend sudo if logged in as not-privileged user, ie. sudo vim

Press 'i' to get in insert mode
add:

export JAVA_HOME="path that you found"

export PATH=$JAVA_HOME/bin:$PATH
logout and login again, reboot, or use source /etc/profile to apply changes immediately in your current shell

mariadb 修改密码
2.1 更新 mysql 库中 user 表的字段：
MariaDB [(none)]> use mysql;  
MariaDB [mysql]> UPDATE user SET password=password('newpassword') WHERE user='root';  
MariaDB [mysql]> flush privileges;  
MariaDB [mysql]> exit;

2.2 或者，使用 set 指令设置root密码：
MariaDB [(none)]> SET password for 'root'@'localhost'=password('newpassword');  
MariaDB [(none)]> exit; 


nginx完美支持thinkphp3.2.2(需配置URL_MODEL=>1pathinfo模式）
http://www.thinkphp.cn/topic/26657.html

nginx+php-fpm性能参数优化原则
http://blog.sina.com.cn/s/blog_5459f60d01012sjf.html

connect() to unix:/tmp/php-fpm.socket failed (11: Resource temporarily unavailable) while connecting to upstream错误的返回会减少
4.调整nginx、php-fpm和内核的backlog（积压）
nginx：
配置文件的server块
listen 80 default backlog=1024;

php-fpm:
配置文件的
listen.backlog = 2048

kernel参数：
/etc/sysctl.conf，不能低于上面的配置
net.ipv4.tcp_max_syn_backlog = 4096
net.core.netdev_max_backlog = 4096

sysctl –p 生效

5.增加单台服务器上的php-fpm的master实例，会增加fpm的处理能力，也能减少报错返回的几率
多实例启动方法，使用多个配置文件：
/usr/local/php/sbin/php-fpm -y /usr/local/php/etc/php-fpm.conf &
/usr/local/php/sbin/php-fpm -y /usr/local/php/etc/php-fpm1.conf &

nginx的fastcgi配置
    upstream phpbackend {
      server   unix:/var/www/php-fpm.sock weight=100 max_fails=10 fail_timeout=30;
      server   unix:/var/www/php-fpm1.sock weight=100 max_fails=10 fail_timeout=30;
      server   unix:/var/www/php-fpm2.sock weight=100 max_fails=10 fail_timeout=30;
      server   unix:/var/www/php-fpm3.sock weight=100 max_fails=10 fail_timeout=30;
    }

        location ~ \.php* {
            fastcgi_pass   phpbackend;
#           fastcgi_pass   unix:/var/www/php-fpm.sock;
            fastcgi_index index.php;
       ..........
       }


Linux如何在系统运行时修改内核参数(/proc/sys 与 /etc/sysctl.conf)  
https://blog.csdn.net/launch_225/article/details/9211731

方法一：修改/proc下内核参数文件内容

　　直接修改内核参数ip_forward对应在/proc下的文件/proc/sys/net/ipv4/ip_forward。用下面命令查看ip_forward文件内容：
　　# cat /proc/sys/net/ipv4/ip_forward
　　该文件默认值0是禁止ip转发，修改为1即开启ip转发功能。修改命令如下：
　　# echo 1 >/proc/sys/net/ipv4/ip_forward
　　修改过后就马上生效，即内核已经打开ip转发功能。但如果系统重启后则又恢复为默认值0，如果想永久打开需要通过修改/etc/sysctl.conf文件的内容来实现。

　　方法二．修改/etc/sysctl.conf文件
　　默认sysctl.conf文件中有一个变量是
　　net.ipv4.ip_forward = 0
　　将后面值改为1，然后保存文件。因为每次系统启动时初始化脚本/etc/rc.d/rc.sysinit会读取/etc/sysctl.conf文件的内容，所以修改后每次系统启动时都会开启ip转发功能。但只是修改sysctl文件不会马上生效，如果想使修改马上生效可以执行下面的命令：
　　# sysctl –p

　　在修改其他内核参数时可以向/etc/sysctl.conf文件中添加相应变量即可，下面介绍/proc/sys下内核文件与配置文件 sysctl.conf中变量的对应关系，由于可以修改的内核参数都在/proc/sys目录下，所以sysctl.conf的变量名省略了目录的前面部分（/proc/sys）。

　　将/proc/sys中的文件转换成sysctl中的变量依据下面两个简单的规则：

　　1．去掉前面部分/proc/sys

　　2．将文件名中的斜杠变为点

　　这两条规则可以将/proc/sys中的任一文件名转换成sysctl中的变量名。

　　例如：

　　/proc/sys/net/ipv4/ip_forward ＝》 net.ipv4.ip_forward

　　/proc/sys/kernel/hostname ＝》 kernel.hostname

　　可以使用下面命令查询所有可修改的变量名

　　# sysctl –a
　　# sysctl –a

　　下面例举几个简单的内核参数：

　　1．/proc/sys/kernel/shmmax
　　该文件指定内核所允许的最大共享内存段的大小。

　　2．/proc/sys/kernel/threads-max
　　该文件指定内核所能使用的线程的最大数目。

　　3．/proc/sys/kernel/hostname
　　该文件允许您配置网络主机名。

　　4．/proc/sys/kernel/domainname
　　该文件允许您配置网络域名 





1、net.ipv4.tcp_max_syn_backlog = 65536

记录的那些尚未收到客户端确认信息的连接请求的最大值。对于超过128M内存的系统而言，缺省值是1024，低于128M小内存的系统则是128。

SYN Flood攻击利用TCP协议散布握手的缺陷，伪造虚假源IP地址发送大量TCP-SYN半打开连接到目标系统，最终导致目标系统Socket队列资源耗尽而无法接受新的连接。为了应付这种攻击，现代Unix系统中普遍采用多连接队列处理的方式来缓冲(而不是解决)这种攻击，是用一个基本队列处理正常的完全连接应用(Connect()和Accept() )，是用另一个队列单独存放半打开连接。

这种双队列处理方式和其他一些系统内核措施(例如Syn-Cookies/Caches)联合应用时，能够比较有效的缓解小规模的SYN Flood攻击(事实证明<1000p/s)加大SYN队列长度可以容纳更多等待连接的网络连接数，一般遭受SYN Flood攻击的网站，都存在大量SYN_RECV状态，所以调大tcp_max_syn_backlog值能增加抵抗syn攻击的能力。

2、net.core.netdev_max_backlog =  32768

每个网络接口接收数据包的速率比内核处理这些包的速率快时，允许送到队列的数据包的最大数目。

3、net.core.somaxconn = 32768

调整系统同时发起并发TCP连接数，可能需要提高连接储备值，以应对大量突发入局连接请求的情况。如果同时接收到大量连接请求，使用较大的值会提高受支持的暂挂连接的数量，从而可减少连接失败的数量。大的侦听队列对防止DDoS攻击也会有所帮助。挂起请求的最大数量默认是128。

4、net.core.wmem_default = 8388608

该参数指定了发送套接字缓冲区大小的缺省值(以字节为单位)

5、net.core.rmem_default = 8388608

该参数指定了接收套接字缓冲区大小的缺省值(以字节为单位)

6、net.core.rmem_max = 16777216

该参数指定了接收套接字缓冲区大小的最大值(以字节为单位)

7、net.core.wmem_max = 16777216

该参数指定了发送套接字缓冲区大小的最大值(以字节为单位)

8、net.ipv4.tcp_timestamps = 0

Timestamps可以防范那些伪造的sequence号码。一条1G的宽带线路或许会重遇到带out-of-line数值的旧sequence号码(假如它是由于上次产生的)。时间戳能够让内核接受这种“异常”的数据包。这里需要将其关掉,以提高性能。

9、net.ipv4.tcp_synack_retries = 2

对于远端的连接请求SYN，内核会发送SYN＋ACK数据报，以确认收到上一个SYN连接请求包。这是所谓的三次握手(threeway handshake)机制的第二个步骤。这里决定内核在放弃连接之前所送出的SYN+ACK数目。不应该大于255，默认值是5，对应于180秒左右时间。(可以根据tcp_syn_retries来决定这个值)

10、net.ipv4.tcp_syn_retries = 2

对于一个新建连接，内核要发送多少个SYN连接请求才决定放弃。不应该大于255，默认值是5，对应于180秒左右时间。(对于大负载而物理通信良好的网络而言,这个值偏高,可修改为2.这个值仅仅是针对对外的连接,对进来的连接,是由tcp_retries1 决定的)

11、net.ipv4.tcp_tw_recycle = 1

表示开启TCP连接中TIME-WAIT Sockets的快速回收，默认为0，表示关闭。

#net.ipv4.tcp_tw_len = 1

12、net.ipv4.tcp_tw_reuse = 1

表示开启重用，允许将TIME-WAIT Sockets重新用于新的TCP连接，默认为0，表示关闭。这个对快速重启动某些服务,而启动后提示端口已经被使用的情形非常有帮助。

13、net.ipv4.tcp_mem = 94500000 915000000 927000000

tcp_mem有3个INTEGER变量：low, pressure, high

low：当TCP使用了低于该值的内存页面数时，TCP没有内存压力，TCP不会考虑释放内存。(理想情况下，这个值应与指定给tcp_wmem的第2个值相匹配。这第2个值表明，最大页面大小乘以最大并发请求数除以页大小 (131072*300/4096)

pressure：当TCP使用了超过该值的内存页面数量时，TCP试图稳定其内存使用，进入pressure模式，当内存消耗低于low值时则退出pressure状态。(理想情况下这个值应该是TCP可以使用的总缓冲区大小的最大值(204800*300/4096)

high：允许所有TCP Sockets用于排队缓冲数据报的页面量。如果超过这个值，TCP连接将被拒绝，这就是为什么不要令其过于保守(512000*300/4096)的原因了。在这种情况下，提供的价值很大，它能处理很多连接，是所预期的2.5倍；或者使现有连接能够传输2.5倍的数据。

一般情况下这些值是在系统启动时根据系统内存数量计算得到的。

14、net.ipv4.tcp_max_orphans = 3276800

系统所能处理不属于任何进程的TCP sockets最大数量。假如超过这个数量﹐那么不属于任何进程的连接会被立即reset，并同时显示警告信息。之所以要设定这个限制﹐纯粹为了抵御那些简单的DoS攻击﹐千万不要依赖这个或是人为的降低这个限制

#net.ipv4.tcp_fin_timeout = 30

#net.ipv4.tcp_keepalive_time = 120

15、net.ipv4.ip_local_port_range = 1024  65535

将系统对本地端口范围限制设置为1024~65000之间

16、net.ipv4.ip_conntrack_max = 10000

设置系统对最大跟踪的TCP连接数的限制(CentOS 5.6无此参数)

解决thinkphp关闭调试模式404报错问题
https://blog.csdn.net/dengjiexian123/article/details/53121552

chmod -R 777 Runtime

查看每个表的行数
SELECT TABLE_NAME,table_rows FROM information_schema.TABLES where TABLE_SCHEMA='xxxdb' order by table_rows;

## PHP 和 nginx 状态页

nginx 

    location /nginx_status {
          stub_status on;
          access_log off;
          allow 127.0.0.1;
          ##allow 192.168.249.0/24;
          deny all;
    }

    location ~ /php_fpm-status$ {
            allow 127.0.0.1;
            #deny all;
            fastcgi_param SCRIPT_FILENAME $fastcgi_script_name;
            include fastcgi_params;
            fastcgi_pass unix:/var/run/php7.0.9-fpm.sock;
    }
    
vi /etc/php/7.2/fpm/pool.d/www.conf

    pm.status_path = /php_fpm-status

PHP 显示错误
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

# PDO 测试

ini_set('display_errors', 0);
ini_set('display_startup_errors', 0);
error_reporting(E_ERROR);

try {
    $dbh = new PDO("mysql:host=localhost;dbname=mysql", 'root', 'password', array(PDO::ATTR_PERSISTENT => true));
    echo "Connection OK <br>";
    $stmt = $dbh->prepare("SELECT * from user limit 1");
    $stmt->execute();
    $row = $stmt->fetch();
    echo $row["User"];
    $dbh = null;
} catch (PDOException $e) {
    echo 'Connection failed: ' . $e->getMessage() . '<br>';
}

# ThinkPHP 

class IndexController extends Controller {
    public function index(){
        $data = M("user_list")->find();
        echo $data['nick_name'];
    }
｝

windows 查看文件 MD5 
certutil.exe -hashfile temp.zip MD5




比特率，帧率，分辨率对视频画质的影响
https://blog.csdn.net/matrix_laboratory/article/details/56291742

$pdo = new PDO("mysql:dbname=database;host=127.0.0.1", "user", "password");
$statement = $pdo->prepare("SELECT * FROM table");
$statement->execute();
$results = $statement->fetchAll(PDO::FETCH_ASSOC);
$json = json_encode($results);

How to apply bindValue method in LIMIT clause?
https://stackoverflow.com/questions/2269840/how-to-apply-bindvalue-method-in-limit-clause

$PDO->setAttribute( PDO::ATTR_EMULATE_PREPARES, false );

sphinx-for-chinese是一款专注于中文搜索的全文检索软件
http://sphinxsearchcn.github.io/
https://github.com/eric1688/sphinx

git clone https://github.com/eric1688/sphinx
cd sphinx
apt-get install libmysqlclient-dev libpq-dev unixodbc-dev
./configure --prefix=/usr/local/sphinx --with-mysql

https://blog.csdn.net/pwjiferox/article/details/79792469

mysql -h127.0.0.1 -P9306 -e "SELECT * FROM test WHERE MATCH('中国');"

坑爹的Sphinx3，各种报错各种错误
https://blog.csdn.net/default7/article/details/87932919


sphinx有两个主要进程indexer和searchd。indexer，正如其名，她的任务是从数据库（或者其他的数据源）收集原始的数据，然后建立相应的索引。searchd则是通过读取indexer建立的索引来响应客户端的请求。如果用图书馆来打比方的话，一个是按照索书号整理数目的工作人员，一个是帮你从书架上取书的人。
要让sphinx为你工作的话，需要做几件事(在配置文件中完成)：
1) 告诉sphinx，数据源在哪儿(配置source，对应MySQL的话，就是填写其主机名、登录用户、密码等)
2) 建立“索引任务”。告诉indexer针对数据源的哪一部分建立索引，建立索引时数据源的编码、最小索引长度等细节信息
执行indexer，完成前面配置文件中的“索引任务”；启动searchd服务。
这样你的应用就可以通过sphinx提供的API来搜索数据库中的数据了，sphinx会为你返回对应数据的主键。

Sphinx-for-chinese (中文全文搜索)安装步骤
https://blog.csdn.net/samxx8/article/details/6635001
sphinx中文索引实现中文检索
https://segmentfault.com/a/1190000008485870?utm_source=tag-newest

windows 下的php Sphinx 3.0入门
https://www.jianshu.com/p/6b77df752b8f

# sphinx 

最快使用步骤

    # 安装依赖
    yum install mysql-devel
    # 下载解压
    wget http://sphinxsearch.com/files/sphinx-3.1.1-612d99f-linux-amd64-glibc2.12.tar.gz
    tar xf sphinx*.gz
    cd sphinx-3.1.1/bin/
    # 创建配置文件：设置索引源，增量源，分区索引，主索引，增量索引，开启 CJK 支持
    vi sphinx.conf
    # 启动搜索服务
    ./searchd -c ./sphinx.conf
    # 查看搜索服务状态
    ./searchd -c ./sphinx.conf --status
    # 创建全量索引，crontab 设置每天凌晨 1 点执行
    ./indexer testindex
    # 创建增量索引，crontab 设置每 10 分钟执行
    ./indexer testindex_delta --rotate
    # 合并增量索引，crontab 设置每 15 分钟执行
    ./indexer --merge testindex testindex_delta --rotate
    # 测试搜索
    mysql -h127.0.0.1 -P9306 -e "SELECT * FROM testindex WHERE MATCH('中国');"
    # 要实现实时索引，可使用 sphinx 的 rt 类型索引，增删改都直接实时操作索引，而不是定时索引

sphinx的配置详情讲解和配置性能优化管理中文解释
http://blog.chinaunix.net/uid-26602509-id-4128576.html

一、sphinx的配置
1. sphinx配置文件结构介绍
Sphinx的配置文件结构如下：
Source 源名称1{ 
#添加数据源，这里会设置一些连接数据库的参数比如数据库的IP、用户名、密码等
#设置sql_query、设置sql_query_pre、设置sql_query_range等后面会结合例子做详细介绍
 ……
}
Index 索引名称1{
     Source=源名称1
#设置全文索引
     ……
}
Indexer{
#设置Indexer程序配置选项，如内存限制等
……
}
Searchd{ 
#设置Searchd守护进程本身的一些参数
……
}
Source和Index都可以配置多个。
 
2. spinx配置案例详细解释
接下来就来针对一个配置案例来做详细的配置介绍：
#定义一个数据源
source search_main
{
           #定义数据库类型
    type = mysql
           #定义数据库的IP或者计算机名
    sql_host = localhost
           #定义连接数据库的帐号
    sql_user = root
           #定义链接数据库的密码
    sql_pass = test123
           #定义数据库名称
    sql_db = test
           #定义连接数据库后取数据之前执行的SQL语句
    sql_query_pre = SET NAMES utf8
    sql_query_pre = SET SESSION query_cache_type=OFF
           #创建一个sph_counter用于增量索引
    sql_query_pre = CREATE TABLE IF NOT EXISTS sph_counter \
                                      ( counter_id INTEGER PRIMARY KEY NOT NULL,max_doc_id INTEGER NOT NULL)
           #取数据之前将表的最大id记录到sph_counter表中
    sql_query_pre = REPLACE INTO sph_counter SELECT 1, MAX(searchid) FROM v9_search
           #定义取数据的SQL，第一列ID列必须为唯一的正整数值
    sql_query = SELECT searchid,typeid,id,adddate,data FROM v9_search where \
                                      searchid<( SELECT max_doc_id FROM sph_counter WHERE counter_id=1 ) \
                                        and searchid>=$start AND searchid<=$end
           # sql_attr_uint和sql_attr_timestamp用于定义用于api过滤或者排序，写多行制定多列
    sql_attr_uint = typeid
    sql_attr_uint = id
    sql_attr_timestamp = adddate
           #分区查询设置
    sql_query_range = SELECT MIN(searchid),MAX(searchid) FROM v9_search
           #分区查询的步长
    sql_range_step = 1000
           #设置分区查询的时间间隔
    sql_ranged_throttle = 0
           #用于CLI的调试
    sql_query_info = SELECT * FROM v9_search WHERE searchid=$id
}
#定义一个增量的源
source search_main_delta : search_main
{
    sql_query_pre = set names utf8
           #增量源只查询上次主索引生成后新增加的数据
#如果新增加的searchid比主索引建立时的searchid还小那么会漏掉
    sql_query = SELECT searchid,typeid,id,adddate,data FROM v9_search where \
                                  searchid>( SELECT max_doc_id FROM sph_counter WHERE counter_id=1 ) \
                                   and searchid>=$start AND searchid<=$end
    sql_query_range = SELECT MIN(searchid),MAX(searchid) FROM v9_search where \
                                       searchid>( SELECT max_doc_id FROM sph_counter WHERE counter_id=1 )
}
 
#定义一个index_search_main索引
index index_search_main
{
           #设置索引的源
    source = search_main
           #设置生成的索引存放路径
    path = /usr/local/coreseek/var/data/index_search_main
           #定义文档信息的存储模式，extern表示文档信息和文档id分开存储
    docinfo = extern
           #设置已缓存数据的内存锁定，为0表示不锁定
    mlock = 0
           #设置词形处理器列表，设置为none表示不使用任何词形处理器
    morphology = none
           #定义最小索引词的长度
    min_word_len = 1
           #设置字符集编码类型，我这里采用的utf8编码和数据库的一致
    charset_type = zh_cn.utf-8
           #指定分词读取词典文件的位置
    charset_dictpath = /usr/local/mmseg3/etc
           #不被搜索的词文件里表。
    stopwords = /usr/local/coreseek/var/data/stopwords.txt
           #定义是否从输入全文数据中取出HTML标记
    html_strip = 0
    # sphinx 3 直接支持中文
    ngram_len     = 1
    ngram_chars = U+4E00..U+9FBB, U+3400..U+4DB5, U+20000..U+2A6D6, U+FA0E, U+FA0F, U+FA11, U+FA13, U+FA14, U+FA1F, U+FA21, U+FA23, U+FA24, U+FA27, U+FA28, U+FA29, U+3105..U+312C, U+31A0..U+31B7, U+3041, U+3043, U+3045, U+3047, U+3049, U+304B, U+304D, U+304F, U+3051, U+3053, U+3055, U+3057, U+3059, U+305B, U+305D, U+305F, U+3061, U+3063, U+3066, U+3068, U+306A..U+306F, U+3072, U+3075, U+3078, U+307B, U+307E..U+3083, U+3085, U+3087, U+3089..U+308E, U+3090..U+3093, U+30A1, U+30A3, U+30A5, U+30A7, U+30A9, U+30AD, U+30AF, U+30B3, U+30B5, U+30BB, U+30BD, U+30BF, U+30C1, U+30C3, U+30C4, U+30C6, U+30CA, U+30CB, U+30CD, U+30CE, U+30DE, U+30DF, U+30E1, U+30E2, U+30E3, U+30E5, U+30E7, U+30EE, U+30F0..U+30F3, U+30F5, U+30F6, U+31F0, U+31F1, U+31F2, U+31F3, U+31F4, U+31F5, U+31F6, U+31F7, U+31F8, U+31F9, U+31FA, U+31FB, U+31FC, U+31FD, U+31FE, U+31FF, U+AC00..U+D7A3, U+1100..U+1159, U+1161..U+11A2, U+11A8..U+11F9, U+A000..U+A48C, U+A492..U+A4C6    
}
#定义增量索引
index index_search_main_delta : index_search_main
{
    source = search_main_delta
    path = /usr/local/coreseek/var/data/index_search_main_delta
}
 
#定义indexer配置选项
indexer
{
           #定义生成索引过程使用索引的限制
    mem_limit = 512M
}
 
#定义searchd守护进程的相关选项
searchd
{
           #定义监听的IP和端口
    #listen = 127.0.0.1
    #listen = 172.16.88.100:3312
    listen = 3312
    listen = /var/run/searchd.sock
           #定义log的位置
    log = /usr/local/coreseek/var/log/searchd.log
           #定义查询log的位置
    query_log = /usr/local/coreseek/var/log/query.log
           #定义网络客户端请求的读超时时间
    read_timeout = 5
           #定义子进程的最大数量
    max_children = 300
           #设置searchd进程pid文件名
    pid_file = /usr/local/coreseek/var/log/searchd.pid
           #定义守护进程在内存中为每个索引所保持并返回给客户端的匹配数目的最大值
    max_matches = 100000
           #启用无缝seamless轮转，防止searchd轮转在需要预取大量数据的索引时停止响应
    #也就是说在任何时刻查询都可用，或者使用旧索引，或者使用新索引
    seamless_rotate = 1
           #配置在启动时强制重新打开所有索引文件
    preopen_indexes = 1
           #设置索引轮转成功以后删除以.old为扩展名的索引拷贝
    unlink_old = 1
           # MVA更新池大小，这个参数不太明白
    mva_updates_pool = 1M
           #最大允许的包大小
    max_packet_size = 32M
           #最大允许的过滤器数
    max_filters = 256
           #每个过滤器最大允许的值的个数
    max_filter_values = 4096
}
 
二、sphinx的管理
1. 生成Sphinx中文分词词库(新版本的中文分词库已经生成在了/usr/local/mmseg3/etc目录下)
cd /usr/local/mmseg3/etc
/usr/local/mmseg3/bin/mmseg -u thesaurus.txt
mv thesaurus.txt.uni uni.lib
2. 生成Sphinx中文同义词库
#同义词库是说比如你搜索深圳的时候，含有深圳湾等字的也会被搜索出来
/data/software/sphinx/coreseek-3.2.14/mmseg-3.2.14/script/build_thesaurus.py unigram.txt > thesaurus.txt
/usr/local/mmseg3/bin/mmseg -t thesaurus.txt
将thesaurus.lib放到uni.lib同一目录
3. 生成全部索引
/usr/local/coreseek/bin/indexer --config /usr/local/coreseek/etc/sphinx.conf –all
若此时searchd守护进程已经启动，那么需要加上—rotate参数：
/usr/local/coreseek/bin/indexer --config /usr/local/coreseek/etc/sphinx.conf --all --rotate
4. 启动searchd守护进程
/usr/local/coreseek/bin/searchd --config /usr/local/coreseek/etc/sphinx.conf
5. 生成主索引
写成shell脚本，添加到crontab任务，设置成每天凌晨1点的时候重建主索引
/usr/local/coreseek/bin/indexer --config /usr/local/coreseek/etc/sphinx.conf --rotate index_search_main
6. 生成增量索引
写成shell脚本，添加到crontab任务，设置成每10分钟运行一次
/usr/local/coreseek/bin/indexer --config /usr/local/coreseek/etc/sphinx.conf --rotate index_search_main_delta
7. 增量索引和主索引的合并
写成shell脚本，添加到计划任务，每15分钟跑一次
/usr/local/coreseek/bin/indexer --config /usr/local/coreseek/etc/sphinx.conf --merge index_search_main index_search_main_delta --rotate
8. 使用search命令在命令行对索引进行检索
/usr/local/coreseek/bin/search --config /usr/local/coreseek/etc/sphinx.conf 游戏


修改 cmd 代码页

rem utf-8
chcp 65001  


细线表格
table {border-collapse:collapse;  border: 1px solid #000000; }
td {  border-collapse:collapse;  border: 1px solid #000000;



sphinx检索语法与匹配模式备忘
https://blog.csdn.net/luochuan/article/details/7313052
CentOS6.9编译安装Sphinx并使用php7的sphinx扩展实现全文搜索
https://www.jmsite.cn/blog-655.html

centos 6.9
http://vault.centos.org/6.9/isos/x86_64/

VMware安装CentOS后网络设置
https://www.cnblogs.com/owaowa/p/6123902.html
修改CentOS默认yum源为国内yum镜像源
https://blog.csdn.net/inslow/article/details/54177191

centos 默认不能使用网络，ifconfig 不能看到 eth0，需要 ifup eth0

# centos
yum makecache
yum install wget

mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.backup
cd /etc/yum.repos.d/
wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-6.repo
yum makecache
yum -y update

记一次 linux 被入侵全过程
https://mp.weixin.qq.com/s?__biz=MzA4Nzg5Nzc5OA==&mid=2651677446&idx=1&sn=37ec373c2d063b8669453572be37dd78
0x04 安全建议
一、服务器

1. 禁用ROOT
2. 用户名和密码尽量复杂
3. 修改ssh的默认22端口
4. 安装DenyHosts防暴力破解软件
5. 禁用密码登录，使用RSA公钥登录
二、redis

1. 禁用公网IP监听，包括0.0.0.0
2. 使用密码限制访问redis
3. 使用较低权限帐号运行redis


# shell

Using grep in a if-else statement [closed]
https://unix.stackexchange.com/questions/156835/using-grep-in-a-if-else-statement

while :
 do
     echo "Please enter a string"
     read input_string
     echo "Please enter the file name too see if that string is present in it - (Enter .abw after)"
     read input_string1
     grep -q "${input_string}" "${input_string1}"                                                                 
     if [ $? -eq 0 ] ; then
         echo  "Your string has been found"
     else 
         echo "Your string has not been found"
     fi
 done
 
 tar xf -C dstdir xxx.tar.gz
 unzip -d dstdir xxx.zip
 
 
 微信小程序精选Demo合集【持续更新中...】
 https://www.jianshu.com/p/0ecf5aba79e1
 
 - 仿QQ应用程序（#社交 #聊天 #SNS）：https://github.com/xiehui999/SmallAppForQQ
 - 仿知乎微信小程序（#SNS #社区 #论坛）：https://github.com/RebeccaHanjw/weapp-wechat-zhihu
 - 微信小程序实现移动端商城（#电商 #商城）：https://github.com/liuxuanqiang/wechat-weapp-mall
 - 知乎日报（#新闻 #资讯）：https://github.com/LiuCaiHe/wechat-app-sample
 - V2EX-微信小程序版（#社区 #社交）：https://github.com/jectychen/wechat-v2ex
 - 音乐播放器（#音乐 #多媒体播放）: https://github.com/eyasliu/wechat-app-music
 - Apple Music（#音乐）:https://github.com/Sioxas/AppleMusic
 - 示例教程（豆瓣电影）: https://github.com/zce/weapp-demo
 - Artand Demo（#艺术 #展示 #媒体）:https://github.com/SuperKieran/weapp-artand
 - cnode社区版（#社区 #论坛）: https://github.com/vincentSea/wechat-cnode
 - 仿芒果TV（#媒体 #视频）: https://github.com/web-Marker/wechat-Development
 - 电影推荐（#媒体 #电影）: https://github.com/yesifeng/wechat-weapp-movie
 - 「ONE · 一个」 的微信小程序（#资讯 #媒体）: https://github.com/ahonn/weapp-one
 - 「分答」 （#问答）: https://github.com/davedavehong/fenda-mock
  
  
mysql 中文全文索引
https://dev.mysql.com/doc/refman/5.7/en/fulltext-search-ngram.html  

# 小程序

小程序组件库
https://weui.io/

6个最优秀的微信小程序UI组件库
https://www.jianshu.com/p/4182f4a18cb6

https://github.com/Tencent/weui-wxss
https://github.com/TalkingData/iview-weapp
https://github.com/youzan/vant-weapp
https://github.com/meili/minui
https://github.com/wux-weapp/wux-weapp

微信小程序案例TODO备忘录
https://www.cnblogs.com/flyingjun/p/9739394.html

50张完美的配色卡！
https://www.jianshu.com/p/15728f68892d

现代 JavaScript 教程
https://zh.javascript.info/

纯CSS实现垂直居中的几种方法
https://www.cnblogs.com/hutuzhu/p/4450850.html




Blockly培训案例-puzzle游戏的制作（一）
https://www.jianshu.com/p/ed74f318ffc1
https://itbilu.com/other/relate/4JL8NjUP7.html
https://itbilu.com/other/relate/Ek5ePdjdX.html
https://itbilu.com/other/relate/r1IhFZV-X.html


上面的示例的Blockly.inject行中，第二个参数是一个键/值对字典。其用于配置Blockly，可用的配置项有：

collapse - boolean。允许“块”折叠或展开。如果工具箱有类别，默认为true；其它情况为false
comments - boolean。允许“块”有注释。如果工具箱有类别，默认为true；其它情况为false
css - boolean。如果设置false，则不注入 CSS；默认为true
disable - boolean。使“块”不可用。如果工具箱有类别，默认为true；其它情况为false
grid - object。配置一个网格，使块可以捕获到。见Grid
horizontalLayout - boolean。设置true则工具箱使用水平布局；false则使用垂直布局。默认为false
maxBlocks - number。最大可创建的“块”数量。默认为Infinity
media - string。Blockly 媒体文件目录路径。默认为"https://blockly-demo.appspot.com/static/media/"
oneBasedIndex - boolean。设置为true则字符串操作索引会从1开始；false则从0开始。默认为true
readOnly - boolean。设置为true，则禁止用户编辑。影响“工具箱”和“垃圾桶”。默认为false
rtl - boolean。设置为true，则镜像化编辑器。默认为false。见RTL Demo
scrollbars - boolean。设置工作区是否可滚动。如果工具箱有类别，默认为true；其它情况为false
sounds - boolean。设置为false，则点击或删除时不会播放声音。默认为true
toolbox - XML节点或string。用户可用“分类”和“块”的结构树。
toolboxPosition - string。设置为start，则工具箱在上部（水平显示时）或左则（垂直显示时）或右则（垂直LTR显示时）。设置为end，则在相对僧。默认为start
trashcan - boolean。显示或隐藏“垃圾桶”。如果工具箱有类别，默认为true；其它情况为false
zoom - object。工作区缩放配置。见Zoom


识别图片中的文字 - Tesseract 和 百度云OCR的对比
https://segmentfault.com/a/1190000012861561?utm_source=tag-newest


sed insert line with spaces to a specific line
https://stackoverflow.com/questions/18439528/sed-insert-line-with-spaces-to-a-specific-line

No display in scala console (REPL) on Ubuntu 18.04
https://github.com/sbt/sbt/issues/4173


IDEA配置Spark找不到spark-assembly-*.jar
https://blog.csdn.net/qq_41378429/article/details/84177785

问题是我在spark文件夹中根本没有看到spark-assembly-*.jar？？？？？

后来才发现因为spark升级到spark2以后，原有lib目录下的大JAR包被分散成多个小JAR包，原来的spark-assembly-*.jar已经不存在。原来的jar包分解为：jars/*.jar


spark build path is cross-compiled with an incompatible version of Scala (2.10.0)
https://stackoverflow.com/questions/34507966/spark-build-path-is-cross-compiled-with-an-incompatible-version-of-scala-2-10-0

Locality
https://www.cnblogs.com/JeffreyZhao/archive/2009/01/22/system-architecture-and-program-performance.html

Spark on Yarn提交任务时报ClosedChannelException解决方案
https://www.linuxidc.com/Linux/2017-01/140068.htm?hmsr=toutiao.io

Code page 850 (Latin-1 - Western European languages)
https://www.ascii-codes.com/cp850.html

Git知识总览(五) Git中的merge、rebase、cherry-pick以及交互式rebase
https://www.cnblogs.com/ludashi/p/8213550.html
Git-用 cherry-pick 挑好看的小樱桃
https://blog.csdn.net/qq_32452623/article/details/79449534

命令行的艺术
https://github.com/jlevy/the-art-of-command-line/blob/master/README-zh.md

Spark on yarn Container beyond virtual memory limits
https://blog.csdn.net/dai451954706/article/details/48828751

# small basic
download：https://social.technet.microsoft.com/wiki/contents/articles/30513.download-microsoft-small-basic.aspx
blog：https://blogs.msdn.microsoft.com/smallbasic/
wiki：https://social.technet.microsoft.com/wiki/contents/articles/17553.wiki-small-basic-portal.aspx
课程：https://social.technet.microsoft.com/wiki/contents/articles/16301.international-small-basic-curriculum.aspx
入门：https://social.technet.microsoft.com/wiki/contents/articles/16298.international-small-basic-getting-started-guide.aspx
扩展：https://gallery.technet.microsoft.com/Small-Basic-LitDev-75827dc5

https://social.technet.microsoft.com/wiki/contents/articles/26950.small-basic-international-resources.aspx

把里面的.dll和.xml共两个文件  复制到Small Basic安装目录的lib目录下 
http://www.smallbasic.cn/thread-1203-1-1.html
https://www.douban.com/group/topic/43015146/

Small Basic常见问题解答
http://www.smallbasic.cn/article-9-1.html

Small Basic让代码语言变得更容易学习 http://www.smallbasic.cn/thread-1201-1-1.html
第一门编程语言选谁？ http://www.smallbasic.cn/thread-1199-1-1.html

Small Basic是一个非常好的针对“零基础”人的入门编程语言，特别适合于年纪较小的学习者（比如初高中学生），也可供非计算机专业（比如文科专业）的大学生编程快速入门。
Small Basic适合那些想尝试自己编程的初学者，Small Basic已经在10到16岁的青少年中成功做过测试，但Small Basic并不仅局限于青少年使用，他同样也适用于那些试图快速上手编程开发和希望寻找编程乐趣的成人。

就我个人观点，计算机专业的学生应该在大一，最晚推迟到大二，就掌握一门主流的通用型编程语言和开发工具（Java和C#是我当前推荐的两种编程语言），并且在今后的专业学习中，使用它们把在后继计算机专业课中学到的理论知识应用于实践。

https://blogs.msdn.microsoft.com/smallbasic/2016/10/30/road-to-small-basic-1-3-blog-2-new-mascots/

JavaScript For Cats
http://jsforcats.com/
http://javascript.info/

https://gitconnected.com/


可能是历史上最全的CC0版权可以免费商用的图片网站
https://segmentfault.com/a/1190000013952992

编程教育的思考
https://blogs.msdn.microsoft.com/user_ed/2016/01/23/cs-spotlight-girls-in-computer-programming-why-it-matters/

每月挑战
Small Basic – About Monthly Challenges
https://blogs.msdn.microsoft.com/smallbasic/2014/06/14/small-basic-about-monthly-challenges/

jison VS PEG.js
https://xwjgo.github.io/2018/03/09/jison%20VS%20PEG.js/

一个parser通常由两部分组成：lexer和parser。

其中lexer又称为scanner或者tokenizer，主要工作是词法分析，即将input转化为一个个小的token。而parser的主要工作是组合lexer产生的tokens来计算结果。

通常不同的语言类型，对应不同的grammar来描述：

如果一门语言可以用一系列的正则表达式来描述，且不包含递归，那么称为regular language，其对应Regular grammar。
如果grammar中包含递归，比如HTML语言，那么可以称为context free language，其对应Context Free grammar(CFG)。
后来又出现了Parsing Expression grammar(PEG)，它和CFG都很强大，但是它能够更加自然地描述变成语言。
CFG和PEG都可以描述我们常见的编程语言，但是PEG不需要进行词法分析，PEG.js就采用了这种grammar，而jison则是采用CFG。

通过 PEG.js 实现一个 JSON Parser
https://www.codercto.com/a/45502.html


http://tedfelix.com/qbasic/

VB 的语法
https://rosettacode.org/wiki/BNF_Grammar

各种编程语言完成同一个任务
https://rosettacode.org/wiki/Category:Programming_Tasks

# 产品课笔记

## 用户

### 定义用户
目标用户是谁？
用户的喜好是什么？
用户在什么场景下用你的产品？

### 接近用户
10-100-1000法则：
- 每个月电话或面对面接触访谈10个用户
- 每个月回复论坛或微博上的100个用户
- 阅读1000个用户在各个渠道的反馈

方式
- 问卷调查
- 回复发帖
- 观察行为
- 走进场景
- 分析数据
- 用户访谈
- 阅读反馈

### 了解用户

变换不同身份，角度。环境，场景，理解各种用户群的使用需求。

## 定位

一句话说明产品定位

- 产品服务于谁？
- 解决用户什么场景下的问题？
- 边界在哪里？

要点
- 产品初期，定位要清晰且聚焦，围绕定位持续打磨
- 产品定位随产品成长，边界也在扩大，定位也会转型
- 时刻谨记核心定位，基础能力值得持续打磨

## 商业化

产品商业化的要点
- 产品能为用户提供价值
- 基于场景挖掘商业模式
- 商业化不能伤害用户体验

互联网常见商业模式
- 直接售卖：内容付费，电商
- 增值服务：会员，特权，道具
- 流量变现：广告，流量分发



JSBasic - A BASIC to JavaScript Compiler
https://www.codeproject.com/articles/25069/jsbasic-a-basic-to-javascript-compiler

深入理解JavaScript系列（4）：立即调用的函数表达式
https://www.cnblogs.com/TomXu/archive/2011/12/31/2289423.html

PARAMETER和ARGUMENT的区别
http://smilejay.com/2011/11/parameter_argument/

根据网上一些资料，对parameter和argument的区别，做如下的简单说明。

1. parameter是指函数定义中参数，而argument指的是函数调用时的实际参数。
2. 简略描述为：parameter=形参(formal parameter)， argument=实参(actual parameter)。
3. 在不很严格的情况下，现在二者可以混用，一般用argument，而parameter则比较少用。


vbs print
https://stackoverflow.com/questions/4388879/vbscript-output-to-console

How do I include a common file in VBScript (similar to C #include)?
https://stackoverflow.com/questions/316166/how-do-i-include-a-common-file-in-vbscript-similar-to-c-include

asp
https://ns7.webmasters.com/caspdoc/html/what_s_in_this_documentation.htm

各种it管理脚本，vbs, lotus
http://scripts.dragon-it.co.uk/links/vbscript-writing-to-stdout-stderr?OpenDocument

ReasonML 先睹为快
https://www.jianshu.com/p/b53f97b0e0a6

现代编程语言Swift、Kotlin等十大有趣功能
http://baijiahao.baidu.com/s?id=1584661819597246308&wfr=spider&for=pc


32 位的 docker ubuntu
https://blog.csdn.net/qq_33530388/article/details/72802817
https://gitlab.com/docker-32bit/ubuntu


NppSnippets, Snippets plug-in for Notepad++
https://www.fesevur.com/nppsnippets/


如何查看linux是32位还是64位
　　使用命令“getconf LONG_BIT”

git clone https://gitlab.com/docker-32bit/ubuntu.git
bash build-image.sh
docker run  32bit/ubuntu:bionic getconf LONG_BIT


debootstrap是debian/ubuntu下的一个工具，用来构建一套基本的系统(根文件系统)。生成的目录符合Linux文件系统标准(FHS)，即包含了/boot、/etc、/bin、/usr等等目录，但它比发行版本的Linux体积小很多，当然功能也没那么强大，因此，只能说是“基本的系统”。

https://www.cnblogs.com/NKCZG/p/4192557.html


极简Ubuntu发布--Minimal Ubuntu：专为服务器、容器和云定制的 
https://www.sohu.com/a/240568576_395306

Run Linux from Web Browser with these Six Websites
https://geekflare.com/run-linux-from-a-web-browser/

Learn And Practice Linux Commands Online For FREE!
https://www.ostechnix.com/learn-and-practice-linux-commands-online-for-free/
https://linuxinabrowser.com/
https://fra.me/


/dev/fb0入门练习(linux FrameBuffer)
https://blog.csdn.net/zgrjkflmkyc/article/details/9402541

rm /  后还剩什么
http://lambdaops.com/rm-rf-remains/

== jdk 11
sudo add-apt-repository ppa:linuxuprising/java
sudo apt-get update
apt-cache search oracle-java11-installer
sudo apt-get install oracle-java11-installer
sudo apt-get install oracle-java11-set-default


传感器
http://www.steves-internet-guide.com/simple-controllable-mqtt-sensor/
https://github.com/vr000m/SensorTrafficGenerator
https://github.com/edfungus/Crouton

https://www.cnblogs.com/asukafighting/p/10922305.html

https://detail.tmall.com/item.htm?spm=a230r.1.14.13.1d0e2e90ty9Tn4&id=41248630584&cm_id=140105335569ed55e27b&abbucket=16

https://www.cnblogs.com/chenfulin5/p/8882055.html


npm install -g cnpm --registry=https://registry.npm.taobao.org

https://iot.stackexchange.com/questions/1205/how-to-enable-websockets-on-mosquitto-running-on-windows


机器人
ros机器人开发概述
https://www.cnblogs.com/yoyo-sincerely/p/5740608.html
Ubuntu系统下的ROS开发 gazebo和rviz 有具体的区别吗？哪个更好用？
https://www.zhihu.com/question/268658280/answer/340190866

## 机器人实验

http://mozdevs.github.io/gamedev-js-3d-aabb/raw_box.html
https://qiao.github.io/PathFinding.js/visual/
https://www.zhihu.com/question/268658280/answer/340190866
https://blog.csdn.net/weixin_41045354/article/details/84198041#comments
gazebo入门教程（一）
https://blog.csdn.net/weixin_41045354/article/details/84881498#comments
https://www.cnblogs.com/zxouxuewei/p/5249935.html
https://www.cnblogs.com/zxouxuewei/tag/%E6%9C%BA%E5%99%A8%E4%BA%BA%E7%B3%BB%E7%BB%9FROS%E7%9A%84%E5%AD%A6%E4%B9%A0%E4%B9%8B%E8%B7%AF/
在Ubuntu 18.04 LTS安装ROS Melodic版机器人操作系统（2019年3月更新MoveIt! 1.0）
https://blog.csdn.net/zhangrelay/article/details/80241758

$ apt-key adv --recv-keys --keyserver keyserver.ubuntu.com F42ED6FBAB17C654
https://blog.csdn.net/ldinvicible/article/details/91427185

机器人仿真软件Gazebo介绍
https://blog.csdn.net/kevin_chan04/article/details/78467218

https://ignitionrobotics.org/tutorials/fuel_tools/1.0/md__data_ignition_ign-fuel-tools_tutorials_installation.html

https://ignitionrobotics.org/api/fuel_tools/3.1/index.html

# gazebo
gazebo: symbol lookup error: /usr/lib/x86_64-linux-gnu/libgazebo_common.so.9: undefined symbol: _ZN8ignition10fuel_tools12ClientConfig12SetUserAgentERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE

# c++filt _ZN8ignition10fuel_tools12ClientConfig12SetUserAgentERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
ignition::fuel_tools::ClientConfig::SetUserAgent(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)

# ldd /usr/lib/x86_64-linux-gnu/libgazebo_common.so.9 | grep ignition
	libignition-math4.so.4 => /usr/lib/x86_64-linux-gnu/libignition-math4.so.4 (0x00007f91e8ba5000)
	libignition-fuel_tools1.so.1 => /usr/lib/x86_64-linux-gnu/libignition-fuel_tools1.so.1 (0x00007f91e895f000)
	libignition-common1.so.1 => /usr/lib/x86_64-linux-gnu/libignition-common1.so.1 (0x00007f91d8c94000)

# nm -D /usr/lib/x86_64-linux-gnu/libignition-fuel_tools1.so.1 | awk '{print $3}' | c++filt | grep -v "^$" | grep ClientConfig
ignition::fuel_tools::FuelClient::FuelClient(ignition::fuel_tools::ClientConfig const&, ignition::fuel_tools::REST const&, ignition::fuel_tools::LocalCache*)
ignition::fuel_tools::FuelClient::FuelClient(ignition::fuel_tools::ClientConfig const&, ignition::fuel_tools::REST const&, ignition::fuel_tools::LocalCache*)
ignition::fuel_tools::LocalCache::LocalCache(ignition::fuel_tools::ClientConfig const*)
ignition::fuel_tools::LocalCache::LocalCache(ignition::fuel_tools::ClientConfig const*)
ignition::fuel_tools::ClientConfig::LoadConfig()
ignition::fuel_tools::ClientConfig::CacheLocation(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)
ignition::fuel_tools::ClientConfig::SetConfigPath(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)
ignition::fuel_tools::ClientConfig::AddServer(ignition::fuel_tools::ServerConfig const&)
ignition::fuel_tools::ClientConfig::operator=(ignition::fuel_tools::ClientConfig const&)
ignition::fuel_tools::ClientConfig::ClientConfig(ignition::fuel_tools::ClientConfig const&)
ignition::fuel_tools::ClientConfig::ClientConfig()
ignition::fuel_tools::ClientConfig::ClientConfig(ignition::fuel_tools::ClientConfig const&)
ignition::fuel_tools::ClientConfig::ClientConfig()
ignition::fuel_tools::ClientConfig::~ClientConfig()
ignition::fuel_tools::ClientConfig::~ClientConfig()
ignition::fuel_tools::ClientConfig::CacheLocation[abi:cxx11]() const
ignition::fuel_tools::ClientConfig::Servers() const

https://ignitionrobotics.org/tutorials/fuel_tools/1.0/md__data_ignition_ign-fuel-tools_tutorials_installation.html

sudo apt-get remove libignition-fuel-tools-dev
sudo apt-get install mercurial cmake pkg-config python ruby-ronn libignition-cmake-dev libignition-common-dev libzip-dev libjsoncpp-dev libcurl4-openssl-dev libyaml-dev
apt-get install libignition-cmake2-dev
apt-get install libignition-common3-dev
hg clone https://bitbucket.org/ignitionrobotics/ign-fuel-tools /tmp/ign-fuel-tools
cd /tmp/ign-fuel-tools
mkdir build
cd build
cmake ../
make -j8
make install

apt-get install libignition-fuel-tools1-1


庄
https://baijiahao.baidu.com/s?id=1632228588716085927&wfr=spider&for=pc

产品彩页
https://www.template.net/business/brochure/product-brochure-template/

# vim

3，屏幕滚动：

按键 操作描述
^F 屏幕向下滚动一屏；
^B 屏幕向上滚动一屏；
^E 屏幕向下滚动一行；
^Y 屏幕向上滚动一行；
^D 屏幕向下滚动半屏；
^U 屏幕向上滚动半屏；
z + Enter 滚动屏幕，使当前光标所在行处于屏幕第一行；
z + . 滚动屏幕，使当前光标所在行处于屏幕中间行；
z + - 滚动屏幕，使当前光标所在行处于屏幕最后一行；

保存无权限文件

1. 输入命令:%! sudo tee % > /dev/null
2. 按提示输入sudo权限密码
3. 输入“L"（Load File）
4. 输入:q命令退出

shift + k 查看函数的 man

ctags -R *
cd /usr/include && ctags -R *

vim ~/.vimrc 
    set tags+=./tags
    set tags+=/usr/include/tags

tag命令用法：
Ctrl＋］  跳到当前光标下单词的标签
Ctrl＋O  返回上一个标签
Ctrl＋T  返回上一个标签

vimgrep /匹配模式/[g][j] 要搜索的文件/范围 
g：表示是否把每一行的多个匹配结果都加入
j：表示是否搜索完后定位到第一个匹配位置

vimgrep /pattern/ %           在当前打开文件中查找
vimgrep /pattern/ *             在当前目录下查找所有
vimgrep /pattern/ **            在当前目录及子目录下查找所有
vimgrep /pattern/ *.c          查找当前目录下所有.c文件
vimgrep /pattern/ **/*         只查找子目录

cn                                          查找下一个
cp                                          查找上一个
copen                                    打开quickfix
cw                                          打开quickfix
cclose                                   关闭qucikfix
help vimgrep                       查看vimgrep帮助


# 行业数据下载

- 国家数据 http://data.stats.gov.cn/index.htm

Linux 基础教程
https://github.com/Web-Dev-Tutor/GNU-Linux/blob/master/8.%20Shell%20Scripts%20.md

笨办法学 Linux 中文版
https://github.com/wizardforcel/llthw-zh

笨办法学 C
https://github.com/wizardforcel/lcthw-zh

草根学 Python
https://github.com/TwoWater/Python

## C

➜  workspace git:(master) ✗ cat hello.h 
char* hello();                                                                                                                                                                                                                                                          
➜  workspace git:(master) ✗ cat hello.c 
#include <stdio.h>

char* hello() {
  return "hello world\n";
}                                                                                                                                                                                                                                                                       
➜  workspace git:(master) ✗ cat main.c 
#include <stdio.h>
#include "hello.h"

int main() {  
  printf("%s", hello());
  return 0;
}
➜  workspace git:(master) ✗ cat Makefile 
proc: main.c hello.h hello.o
        gcc -o main.o main.c hello.o 
hello.o: hello.c
        gcc -c hello.c
.PHONY : clean run
clean:
        rm *.o 
run:
        ./main.o
        

# 旋转的正方形
speed 1000
for i = 1 to 300
    fd i
    rt 91
end

# 五角星
def star(n)
  for i = 1 to n
    fd 100
    rt 2 * 360 / n
  end
end

jumpto 0-50, 50
pen "turquoise", 4
star 5

jumpto 0-120, 0-80
pen "red", 4
star 7

jumpto 100, 0-80
pen "blue", 4
star 3

def circle(l)
  rt 360, l 
end 

speed 10
pen "red"

jump 0 - 100, 0
circle 100

jump 50, 20
circle 20

jump 80, 0
circle 20

jump 0, 70
rt 40
fd 40
rt 140
fd 40

jump 100, 0
rt 40 + 180
fd 40
rt 140
fd 40


python 笔记之“海龟”画图 演示画小猪佩奇，机器猫
https://blog.csdn.net/qq_42179526/article/details/83031653

代码小时：铅笔代码编程
http://event.pencilcode.net/home/hoc2014/

移动 H5 首屏秒开优化方案探讨
http://blog.cnbang.net/tech/3477/

谈谈异步加载JavaScript
https://www.cnblogs.com/jinguangguo/p/4187641.html


gzip on;
gzip_min_length 1k;
gzip_buffers 4 16k;
#gzip_http_version 1.0;
gzip_comp_level 2;
gzip_types text/plain application/x-javascript text/css application/xml text/javascript application/x-httpd-php image/jpeg image/gif image/png;
gzip_vary off;
gzip_disable "MSIE [1-6]\.";

3、解释一下

第1行：开启Gzip
第2行：不压缩临界值，大于1K的才压缩，一般不用改
第3行：buffer，就是，嗯，算了不解释了，不用改
第4行：用了反向代理的话，末端通信是HTTP/1.0，有需求的应该也不用看我这科普文了；有这句的话注释了就行了，默认是HTTP/1.1
第5行：压缩级别，1-10，数字越大压缩的越好，时间也越长，看心情随便改吧
第6行：进行压缩的文件类型，缺啥补啥就行了，JavaScript有两种写法，最好都写上吧，总有人抱怨js文件没有压缩，其实多写一种格式就行了
第7行：跟Squid等缓存服务有关，on的话会在Header里增加"Vary: Accept-Encoding"，我不需要这玩意，自己对照情况看着办吧
第8行：IE6对Gzip不怎么友好，不给它Gzip了



【一步步学习编写Makefile】Makefile介绍
https://www.jianshu.com/p/3c91ec0b8f05


TAB替换为空格：
:set ts=4
:set expandtab
:%retab!

空格替换为TAB：
:set ts=4
:set noexpandtab
:%retab!

Spark Shell简单使用
https://www.cnblogs.com/csguo/p/7753565.html


搜狗语料库
http://download.labs.sogou.com/resource/q.php

iconv -c -f gbk -t utf-8 SogouQ.reduced > SogouQ.reduced.utf8
bash sogou-log-extend.sh SogouQ.reduced.utf8 SogouQ.reduced.utf8.ext
bash sogou-log-filter.sh SogouQ.reduced.utf8.ext SogouQ.reduced.utf8.flt
hadoop fs -mkdir -p /sogou
hadoop fs -put SogouQ.reduced.utf8.flt /sogou

mysql -uroot -p

    create database hive;  
    grant all on hive.* to 'hive'@'localhost' identified by 'hive';  

cd $HIVE_HOME/lib
wget $FILE_SERVER/mysql-connector-java-5.1.26-bin.jar


cd $HIVE_HOME/conf
vi hive-site.xml

    <?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
    <configuration>
    <property>
    <name>javax.jdo.option.ConnectionURL</name>
    <value>jdbc:mysql://localhost:3306/hive?createDatabaseIfNotExist=true</value>
    </property>
     
    <property>
    <name>javax.jdo.option.ConnectionDriverName</name>
    <value>com.mysql.jdbc.Driver</value>
    </property>
     
    <property>
    <name>javax.jdo.option.ConnectionUserName</name>
    <value>hive</value>
    </property>
     
    <property>
    <name>javax.jdo.option.ConnectionPassword</name>
    <value>hive</value>
    </property>
    </configuration>

cd /usr/local/hive/bin
schematool -dbType mysql -initSchema


hive

    #  Hive的基本操作：
    show databases;
    create database sogou;
    use sogou;
    show tables;

    create external table sogou.sogou_ext(ts string,uid string,keyword string,rank int,sorder int,url string,hour int)row format delimited fields terminated by '\t' stored as textfile location 'hdfs://localhost:9000/sogou/SogouQ.reduced.utf8.flt';

    CREATE EXTERNAL TABLE sogou.sogou_partition(ts STRING,uid STRING,keyword STRING,rank INT, sorder INT,url STRING)
        COMMENT 'This is the sogou search data by partition' 
        partitioned by (hour INT)
        ROW FORMAT DELIMITED
        FIELDS TERMINATED BY '\t'
        STORED AS TEXTFILE;

    set hive.exec.dynamic.partition.mode=nonstrict;
    insert overwrite table sogou.sogou_partition partition(hour) select * from sogou.sogou_ext;


    show create table sogou.sogou_partition;
    describe sogou.sogou_partition;

    # 实现数据分析需求一：条数统计：
    select * from sogou_ext limit 10;
    select url from sogou_ext limit 10;
    select count(*) from sogou.sogou_ext;
    select count(*) from sogou.sogou_ext where keyword is not null and keyword!='';
    select count(*) from (select ts, uid, keyword,url from sogou.sogou_ext group by ts,uid,keyword,url having count(*)=1) a;
    select count(distinct(uid)) from sogou.sogou_ext;
    
    #实现数据分析需求二：关键词分析：
    select avg(a.cnt) from (select length(keyword) as cnt from sogou.sogou_ext) a;
    select keyword,count(*) as cnt from sogou.sogou_ext group by keyword order by cnt desc limit 50;
    
    # 实现数据分析需求三：UID 分析
    select SUM(IF(uids.cnt=1,1,0)),SUM(IF(uids.cnt=2,1,0)),SUM(IF(uids.cnt=3,1,0)),SUM(IF(uids.cnt>3,1,0)) from (select uid,count(*) as cnt from sogou.sogou_ext group by uid) uids;
    select sum(a.cnt)/count(a.uid) from (select uid,count(*) as cnt from sogou.sogou_ext group by uid) a;
    select count(a.uid) from ( select uid,count(*) as cnt from sogou.sogou_ext group by uid having cnt > 2) a;
    select count(distinct (uid)) from sogou.sogou_ext;
    select count(a.uid) from ( select uid,count(*) as cnt from sogou.sogou_ext group by uid having cnt > 2) a;
    select b.* from (select uid,count(*) as cnt from sogou.sogou_ext group by uid having cnt > 2) a join sogou.sogou_ext b on a.uid=b.uid limit 50;
    
    # 实现数据分析需求四：用户行为分析：



微服务架构体系的深度治理
https://www.infoq.cn/article/q65dDiRTdSbF*E6Ki2P4
人人都是 API 设计师：我对 RESTful API、GraphQL、RPC API 的思考
https://www.infoq.cn/article/ZgAAVBZZaoo4I0-pkgV8
聊聊服务稳定性保障这些事
https://www.infoq.cn/article/69TYjy_v9u4FxXNUk2gK
SSH 登陆问题及排查思路
https://www.infoq.cn/article/pqU7iMf8cHpz-RNLOslJ

首个在线教育规范即将落地，或影响未来十年行业格局
https://36kr.com/p/5218918
教培行业硝烟再起：从C端到B端，互联网巨头重划势力版图
https://36kr.com/p/5217902
typedef的用法，C语言typedef详解
http://c.biancheng.net/view/298.html

这个时代最容易的创造就是写程序
https://mp.weixin.qq.com/s?__biz=MzIxMzkwNDEyNQ==&mid=2247495471&idx=1&sn=fe9637e684da0531c1b186ddccbff3eb
如何快速入门数据分析？
https://mp.weixin.qq.com/s?__biz=MzIxMzkwNDEyNQ==&mid=2247495451&idx=1&sn=7d02c3e8cca738e13ab59d5b5d3955f5
李笑来：为什么把编程当作自学的入口？
https://mp.weixin.qq.com/s?__biz=MzIxMzkwNDEyNQ==&mid=2247495350&idx=1&sn=ee92d6e6f4c45ea28e6a0ade1773856f

所有这些人，他们的共同身份是创造者（creator），创造不分大小都很幸福。这个时代最容易的创造就是写程序。

编程这件事，本质上就是像教小孩一样教计算机做事，计算机学会之后反过来帮助我们突破自身极限，完成新的进化。未来“可编程”的东西会越来越多，所以我希望能帮助更多人掌握编程这个工具，做真正的“计算机普及教育”。

首先，编程这个东西反正要自学—— 不信你问问计算机专业的人，他们会如实告诉你的，学校里确实也教，但说实话都教得不太好……

其次，编程这个东西最适合 “仅靠阅读自学”—— 这个领域发展很快，到最后，新东西出来的时候，没有老师存在，任由你是谁，都只能去阅读 “官方文档”，只此一条路。

然后，也是最重要的一条，别管是不是很多人觉得编程是很难的东西，事实上它就是每个人都应该具备的技能。

自学训练营 - Python进阶
https://detail.youzan.com/show/goods?alias=1y91dy04salys&banner_id=g.474318213~recommend_fixed~2~ZQlrdXUs&alg=common_by_item.view_fpgrowth.0%3A20190628%23474318213%23browse%2Csimple_rank&reft=1561788613248&spm=g.474318213&sf=wx_sm&is_share=1&from_uuid=5f05bbd6-7e2a-7db6-dc9c-d9d1d51104fa

vim查看函数原型
https://blog.csdn.net/kulala082/article/details/60578869
偶用shift+K
光标处Ctrl-]键：跳到光标所在单词的tag。Ctrl-T键：跳回原来的位置。g]键（先按g再按]）：如果有同名的多个tag，可以用这两个键进行跳转，会提示选择序号。

安装中文包
sudo yum install man-pages-zh-CN

查找man手册安装位置
sudo find / -name man
其中有/usr/share/man/将其复制
添加别名 cman
在/etc/.bashrc中添加别名cman
文件添加字符为：
alias cman='man -M /usr/share/man/zh_CN/'

编译.bashrc文件，使别名 cman 生效
source ./barchrc


命令源码
rpm -qf /bin/echo
coreutils-8.4-46.el6.x86_64

教育科技和职业教育是36氪长期关注的赛道之一，欢迎关注这两个赛道的朋友交流、约稿（静婷微信 15500016421，请备注姓名、公司、职位和来意哦~）



# rm -rf /usr/lib/python2.7/dist-packages/OpenSSL
# rm -rf /usr/lib/python2.7/dist-packages/pyOpenSSL-0.15.1.egg-info
# sudo pip install pyopenssl


set DEBUG=express:* & node index.js
DEBUG=express:* node index.js

splint的安装与使用
https://blog.csdn.net/pngynghay/article/details/18356475
代码静态分析工具——splint的学习与使用
https://www.cnblogs.com/bangerlee/archive/2011/09/07/2166593.html

splint test.c +posixlib 支持POSIX库
splint test.c +unixlib 支持Unix库



机器学习之分类器性能指标之ROC曲线、AUC值
https://www.cnblogs.com/dlml/p/4403482.html

roc曲线：接收者操作特征(receiveroperating characteristic),roc曲线上每个点反映着对同一信号刺激的感受性。

横轴：负正类率(false postive rate FPR)特异度，划分实例中所有负例占所有负例的比例；(1-Specificity)

纵轴：真正类率(true postive rate TPR)灵敏度，Sensitivity(正类覆盖率)

横轴FPR:1-TNR,1-Specificity，FPR越大，预测正类中实际负类越多。

纵轴TPR：Sensitivity(正类覆盖率),TPR越大，预测正类中实际正类越多。


为什么学习编程的孩子更聪明？怎么连高考都考编程题了？
https://mp.weixin.qq.com/s?__biz=Mzg2MzE3MjExNw==&mid=2247483696&idx=1&sn=bfe7a331ad8ae28e9a842d0c065e0e8a
为什么要学习编程？这么有趣的技能你应该Get到
https://mp.weixin.qq.com/s?__biz=MzIxMjI1MDcyMQ==&mid=2247506132&idx=2&sn=6e244f395e61e2a833ac6c786746d27c

snprintf vs. strcpy (etc.) in C
https://stackoverflow.com/questions/2606539/snprintf-vs-strcpy-etc-in-c

理解pid_max，ulimit -u和thread_max之间的区别
http://www.kbase101.com/question/31438.html

《编写高质量代码改善C++程序的150个建议》摘录
https://blog.csdn.net/fengbingchun/article/details/9193577



计算机组装与维护技能考试方案(
https://wenku.baidu.com/view/0977040abf23482fb4daa58da0116c175e0e1e51.html


ganglia配置文件详解
https://blog.csdn.net/CPP_MAYIBO/article/details/76147514


强制重新安装
pip install --upgrade --force-reinstall <package>
When upgrading, reinstall all packages even if they are already up-to-date.

pip install -I <package>
pip install --ignore-installed <package>
Ignore the installed packages (reinstalling instead).

How to get file creation & modification date/times in Python?
https://stackoverflow.com/questions/237079/how-to-get-file-creation-modification-date-times-in-python

势高，则围广：TiDB 的架构演进哲学
https://www.infoq.cn/article/Qw_8ubZFgtQlcZmZHBlA

Openresty最佳实践 1.0
https://openresty.net.cn/get_req_body.html#%E6%9C%80%E7%AE%80%E5%8D%95%E7%9A%84-hello-
OpenResty 最佳实践
https://moonbingbing.gitbooks.io/openresty-best-practices/content/


查看Linux的CPU信息，核数等
# 总核数 = 物理CPU个数 X 每颗物理CPU的核数 
# 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数

# 查看物理CPU个数
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l

# 查看每个物理CPU中core的个数(即核数)
cat /proc/cpuinfo| grep "cpu cores"| uniq

# 查看逻辑CPU的个数
cat /proc/cpuinfo| grep "processor"| wc -l

查看CPU信息（型号）
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c

 查看内 存信息
# cat /proc/meminfo

统计容器数
docker service ls | awk '{if (NR!=1) sum+= substr($4,1,1)} END{print sum}'

Go语言高级编程
https://chai2010.gitbooks.io/advanced-go-programming-book/content/ch1-basic/ch1-01-genesis.html
Go 语言圣经 中文版
https://wizardforcel.gitbooks.io/gopl-zh/content/

Install and Configure TigerVNC server on Ubuntu 18.04
https://www.cyberciti.biz/faq/install-and-configure-tigervnc-server-on-ubuntu-18-04/

sudo apt install tigervnc-standalone-server tigervnc-xorg-extension tigervnc-viewer

放弃 Tightvnc, 选择 Tigervnc
https://www.cnblogs.com/johnsonshu/p/8432307.html

你必须懂的前端性能优化
https://mp.weixin.qq.com/s?__biz=MjM5NTEwMTAwNg==&mid=2650216664&idx=2&sn=81f0586aa25b0b1b27c4d00db6ba7ee7

$ docker ps --format '{{.ID}}' | xargs docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
10.0.0.178
10.0.0.174
10.0.0.172
10.0.0.69
10.0.0.250
192.168.192.2
10.255.0.6


sed -ri.bak "s/(DAEMON_ARGS=\")/\1-R /" /etc/init.d/php7.2-fpm
sed -ri.bak 's/www-data/root/' /etc/php/7.2/fpm/pool.d/www.conf

# 题库
100个Java经典例子（1-10）初学者的利器高手的宝典JavaSE
https://blog.csdn.net/fljxzxb/article/details/6889084
5个JAVA入门必看的经典实例
https://www.jb51.net/article/126116.htm
Java 测验
https://www.runoob.com/quiz/java-quiz.html
一些java小案例
https://blog.csdn.net/TOKISAKIYUU/article/details/78076768



linux下zip解压乱码问题的解决
https://blog.csdn.net/liyuerge/article/details/78998131
unzip -O CP936 xxx.zip


图片分类 Image Classification using Python and Scikit-learn
https://gogul09.github.io/software/image-classification-python



为人父母者，也许并不在乎孩子将来能否成为精英，但是一定都希望自己的孩子成为一个人格健全、智力完善、快乐自信的人。千万不要让强行的知识灌输
把人们与生俱来、主动探索知识的热情完全扼杀掉！阿儿法营致力于中国少儿电脑启蒙教育，我们所做的一切只为孩子从快乐中获取知识、获得成长，从孩子的兴趣和学习效果出发。

阿儿法营的一个教育核心，和两个教育基本点。

快乐教育：教育原本不应该是呆板的，应试的，纯粹的自由与快乐是快速掌握和灵活运用知识的最佳催化剂。由兴趣驱动的学习，能激发好奇心、扩展想象力、培植求知欲，帮助学生形成创造型人格。
信息科学和技术：以趣味程序设计,引导孩子了解信息科学和技术。
科学探索的精神：以计算机为工具,对各领域问题进行探索研究。

我们的理念：越科技越奇幻，越探索越快乐
我们的特色：小班授课，学费一贯制
我们的愿景：做最好的少儿编程启蒙教育
我们的使命：引领孩子走上科学创造、求索真理的道路
我们的目标：让孩子快乐自由地成长
我们的价值观：教育要面向世界、面向未来！
我们的人才观：爱  激情 使命

MTS教学模式是阿儿法营的教学理念，让学生把快乐和思考结合起来，帮助养成批判性思维和动手实践能力。

M：Meaningful 富有意义的电脑编程创作，创作内容和孩子的生活兴趣紧密相关；
T：Thoughtful 以自我探索的方式深入游戏的因果关系、逻辑顺序、模拟现实；
S：Social 编程伙伴间开展协作、评论和相互激励。
卡丁车教育法则是阿儿法营独创的“创意编程”教育方法，也是经由多年的教学实践检验过的。

挑战性，无需了解卡丁车内部构造，学生在简单了解操控方法后，就可以直接上路；
兴奋度，让学生在实践中发现问题，再解决问题，这个学习 - 修正 - 再学习的过程，帮助孩子建立一个正向反馈的学习循环；
操作性，驾驶过程由学生独立操作，除非遇到障碍无法前进时，才需要老师指导，培养学生的自我学习技能。

Linux下C语言的socket网络编程
https://www.cnblogs.com/uestc-mm/p/7630145.html

Linux 中如何快速查看 C 库函数的头文件、库文件
https://blog.csdn.net/byxdaz/article/details/80381584
查找运行程序依赖那些动态库
在linux中查询程序依赖那些动态库， ldd是list, dynamic, dependencies的缩写， 意思是， 列出动态库依赖关系。是查看运行程序的执行库文件。

 find /usr/include -name '*.h' | xargs grep sockaddr_in
 
 Vim自动补全神器YouCompleteMe的配置
 https://www.cnblogs.com/alinh/p/6699789.html
 
 How do I run a C program from VIM?
 https://stackoverflow.com/questions/2627886/how-do-i-run-a-c-program-from-vim
 
 关于【微服务】，你必须了解这些
 https://zhuanlan.zhihu.com/p/40852402
 微服务下使用GraphQL构建BFF
 https://zhuanlan.zhihu.com/p/35108457
 
 gdb strace ltrace
 
 linux 性能分析工具——perf
 https://blog.csdn.net/u014608280/article/details/80265718
 
 Reading input from stdin
 https://codereview.stackexchange.com/questions/116342/reading-input-from-stdin
 https://stackoverflow.com/questions/13993742/is-there-any-way-to-peek-at-the-stdin-buffer
 https://stackoverflow.com/questions/3919009/how-to-read-from-stdin-with-fgets
 https://stackoverflow.com/questions/4023895/how-do-i-read-a-string-entered-by-the-user-in-c
 https://stackoverflow.com/questions/15883568/reading-from-stdin
 
 https://stackoverflow.com/questions/3767284/using-printf-with-a-non-null-terminated-string
 printf("%.*s", stringLength, pointerToString);
 fwrite(your_string, sizeof(char), number_of_chars, stdout);
 
char buffer[10];
read(STDIN_FILENO, buffer, 10);

 sudo grep 'STDIN_FILENO' /usr/include/* -R | grep 'define'

printf("%s", "Hello world\n");              // "Hello world" on stdout (using printf)
fprintf(stdout, "%s", "Hello world\n");     // "Hello world" on stdout (using fprintf)
fprintf(stderr, "%s", "Stack overflow!\n"); // Error message on stderr (using fprintf)

sprintf has an extremely useful return value that allows for efficient appending.

Here's the idiom:

char buffer[HUGE] = {0}; 
char *end_of_string = &buffer[0];
end_of_string += sprintf( /* whatever */ );
end_of_string += sprintf( /* whatever */ );
end_of_string += sprintf( /* whatever */ );

char sBuffer[iBufferSize];
char* pCursor = sBuffer;

pCursor += snprintf(pCursor, sizeof(sBuffer) - (pCursor - sBuffer),  "some stuff\n");

for(int i = 0; i < 10; i++)
{
   pCursor += snprintf(pCursor, sizeof(sBuffer) - (pCursor - sBuffer),  " iter %d\n", i);
}

pCursor += snprintf(pCursor, sizeof(sBuffer) - (pCursor - sBuffer),  "into a string\n");

 dhclient命令使用动态主机配置协议动态的配置网络接口的网络参数。

语法
dhclient(选项)(参数)
选项
0：指定dhcp客户端监听的端口号；
-d：总是以前台方式运行程序；
-q：安静模式，不打印任何错误的提示信息；
-r：释放ip地址。
参数
网络接口：操作的网络接口。

实例
dhclient -r     #释放IP
dhclient        #获取IP

etcd，zookeeper, lstio, consoul


为配合国家大数据战略，加快产业人才培养，教育部增设了“数据科学与大数据技术”本科专业和“大数据技术与应用”高职专业，并得到各高校的积极响应。为协助高校推动大数据相关学科的深化发展，助力产业创新人才培养，教育部高等学校计算机类专业教学指导委员会、中国工程院下属中国工程科技知识中心、联合国教科文组织国际工程科技知识中心作为联合主办单位，复旦大学、同济大学、北京师范大学、西安电子科技大学、大连理工大学、郑州大学和华南师范大学作为承办单位，浪潮集团、百度、联创教育和清华大学出版社等单位或企业作为协办单位，联合发起“全国高校大数据应用创新大赛”（以下简称“竞赛”）。

全国高校大数据应用创新大赛是全国性的大数据学科竞赛，旨在为相关专业的学生提供一个应用创新设计竞技平台，促进学生技术技能、创新思维、实践能力和协作能力的培养，并围绕竞赛建立起专业研讨、师资研修和产学研融合创新体系，逐步推动大数据相关专业的建设，为产业发展提供人才支撑。


Python3.6 Dtrace实战
https://zhuanlan.zhihu.com/p/24743984
Instrumenting CPython with DTrace and SystemTap
https://docs.python.org/3/howto/instrumentation.html
使用 Linux tracepoints, perf以及eBPF跟踪网络数据包的流程
https://www.4hou.com/technology/6994.html
eBPF跟踪技术初学教程
https://baijiahao.baidu.com/s?id=1621543051336579437&wfr=spider&for=pc

# apt-get update
# apt-get install bpfcc-tools
# apt-get install systemtap-sdt-dev

# wget https://npm.taobao.org/mirrors/python/3.7.4/Python-3.7.4.tgz
# tar xf Python-3.7.4.tgz
# cd Python-3.7.4/
# ./configure --with-dtrace
# make -j8
# ./python --version
# readelf -S ./python | grep .note.stapsdt


root@ubuntu3:/home/helloworld/tmp# export PATH=/home/helloworld/download/Python-3.7.4:$PATH
root@ubuntu3:/home/helloworld/tmp# python dtracetest.py &
[1] 19393
root@ubuntu3:/home/helloworld/tmp# echo $!
19393
root@ubuntu3:/home/helloworld/tmp# /usr/sbin/lib/uflow -l python $!
Tracing method calls in python process 19393... Ctrl-C to quit.
CPU PID    TID    TIME(us) METHOD
1   19393  19393  0.999    -> dtracetest.py.mytest
1   19393  19393  0.999      -> /home/helloworld/tmp/mylib.py.md5
1   19393  19393  0.999        -> /home/helloworld/tmp/mylib.py._md5
1   19393  19393  1.000          -> /home/helloworld/tmp/mylib.py.__md5
1   19393  19393  1.000          <- /home/helloworld/tmp/mylib.py.__md5
1   19393  19393  1.000        <- /home/helloworld/tmp/mylib.py._md5
1   19393  19393  1.000      <- /home/helloworld/tmp/mylib.py.md5
1   19393  19393  1.000      -> /home/helloworld/tmp/mylib.py.md5
1   19393  19393  1.000        -> /home/helloworld/tmp/mylib.py._md5
1   19393  19393  1.000          -> /home/helloworld/tmp/mylib.py.__md5
1   19393  19393  1.000          <- /home/helloworld/tmp/mylib.py.__md5
1   19393  19393  1.000        <- /home/helloworld/tmp/mylib.py._md5
1   19393  19393  1.000      <- /home/helloworld/tmp/mylib.py.md5
1   19393  19393  1.000    <- dtracetest.py.mytest


XPath轴(XPath Axes)可定义某个相对于当前节点的节点集：

     1、child  选取当前节点的所有子元素

     2、parent  选取当前节点的父节点

     3、descendant 选取当前节点的所有后代元素（子、孙等）

     4、ancestor  选取当前节点的所有先辈（父、祖父等）

     5、descendant-or-self 选取当前节点的所有后代元素（子、孙等）以及当前节点本身

     6、ancestor-or-self  选取当前节点的所有先辈（父、祖父等）以及当前节点本身

     7、preceding-sibling 选取当前节点之前的所有同级节点

     8、following-sibling 选取当前节点之后的所有同级节点

     9、preceding   选取文档中当前节点的开始标签之前的所有节点

     10、following   选取文档中当前节点的结束标签之后的所有节点

     11、self  选取当前节点

     12、attribute  选取当前节点的所有属性

     13、namespace 选取当前节点的所有命名空间节点

## 生产环境问题排查

- 总体
    - apt-get install sysstat ， yum install sysstat
    - 查看内存，CPU，网卡，ip 信息
    - uptime
    - vmstat
    - top
    - 进程数
    - 磁盘空间，找出最大的目录，日志滚动，cat /dev/null >xxx.log
    - 备份
    - sar
    - /proc 文件
    - 32, 64
    - htop
    - lscpu
    - lsmem
    - lspci
    - lsb_release -a
    -  cat /etc/issue
    - !$， !!， history，, ctrl +r， ctrl +p, ctrl +n， ctrl +l，
    - ctrl + u, ctrl + k, ctrl +h, alt + 左右方向，ctrl + f, ctrl +b, alt + f, alt + b, , esc + f, esc + b, ctrl +e, ctrl + a, ctrl+w, shift +insert, ctrl +xu，Ctrl+v+Tab
    - vmstat
    - ps -ef | grep xxx
    - atop
    - htop
    - dmesg
- CPU 问题
    - profile
    - 火焰图
    - ps top cpu
    - perf
    - mpstat -A，mpstat -P ALL 5 2
    - dstat -tcdrlmn --top-cpu --top-mem  #查看时间、CPU、磁盘读写、IO、负载、内存、网络、最高的CPU占用和最高的内存占用
    - dstat -cl -C 0,1,2,3,4,5,6,7 --top-cpu #查看8颗核心，每颗核心的使用情况和CPU使用情况    
- 内存问题
    - free
    - ps top mem
    - 内存走势
    - 交换分区
    - ps -eo 'pid,rss' --sort=-rss    
    - cat /proc/meminfo |grep -i slab
    -  LANG=en free -m
    - top, shift +m 
- IO 问题
    - iotop
    - iostat
    - lsof
    - 查看文件被谁读写
- 网络问题
    - dstat  --tcp
    - nload
    - socat
    - watch netstat -i
    - nicstat
    - ping
    - curl -v
    - netstat, ss    
    - tcpdump -A -s0 -i eth0
    - curl 测速
    - dns 问题，dig + trace
    - 实时抓包
    - wireshark 使用，https 分析
    - fiddle
    - 持续ping
- 应用问题
    - pstree -apnh -p 17963
    - pgrep -a python, pidof nginx
    - 日志
    - dtrace
    - ltrace
    - slow log
    - strace, php 单进程
    - strace -f -p 17963
    - 权限问题, ls 文件，目录，可执行，mysql 权限问题，php写日志权限问题
    - 链接问题: ldd, nm, c++filt
    - 环境变量
    - find
    - local
    - where
    - which
    - diff
    - sed -i.bak
    - grep
    - lsattr, chattr
    - lslocks 查看文件锁
    - iconv unix2dos
    - pidstat -d -t -p 17963， pidstat -r -t -p 17963, pidstat -u -t -p 17963, pidstat -w -t -p 17963
    - strace -fp <pid> -e trace=stat
    - sudo strace -fp <pid> -e trace=open,stat,close,unlink
    - gdb 段错误， core dump    
- db
    - 查看配置参数 show var
    - show process
    - explan
    - slow log
    
超全整理！Linux性能分析工具汇总合集
http://rdc.hundsun.com/portal/article/731.html?ref=myread    
        
Can I monitor a local unix domain socket like tcpdump?
https://superuser.com/questions/484671/can-i-monitor-a-local-unix-domain-socket-like-tcpdump
sudo mv /path/to/sock /path/to/sock.original
sudo socat -t100 -x -v UNIX-LISTEN:/path/to/sock,mode=777,reuseaddr,fork UNIX-CONNECT:/path/to/sock.original        

Linux的进程间通信-文件和文件锁
https://www.cnblogs.com/wanghuaijun/p/7738788.html

Shell快捷键
https://www.cnblogs.com/xhcdream/p/6973461.html

Ctrl+d 删除光标所在处字符
Ctrl+h 删除光标所在处前一个字符
Ctrl+(x u) 按住Ctrl的同时再先后按x和u，撤销刚才的操作

mpstat 命令详解
https://blog.csdn.net/evils798/article/details/7524474

让我们看一看这些列值的含义：
 
 
%user     表示处理用户进程所使用 CPU 的百分比。用户进程是用于应用程序（如 Oracle 数据库）的非内核进程。
          在本示例输出中，用户 CPU 百分比非常低。
 
%nice     表示使用 nice 命令对进程进行降级时 CPU 的百分比。在之前的部分中已经对 nice 命令进行了介绍。简单来说，nice 命令更改进程的优先级。
 
%system   表示内核进程使用的 CPU 百分比
 
%iowait   表示等待进行 I/O 所使用的 CPU 时间百分比
 
%irq      表示用于处理系统中断的 CPU 百分比
 
%soft     表示用于软件中断的 CPU 百分比
 
%idle     显示 CPU 的空闲时间
 
%intr/s   显示每秒 CPU 接收的中断总数

当您拥有前面所述的 vmstat 时，您可能想知道 mpstat 命令的作用。差别很大：mpstat 可以显示每个处理器的统计，
而 vmstat 显示所有处理器的统计。因此，编写糟糕的应用程序（不使用多线程体系结构）可能会运行在一个多处理器机器上，
而不使用所有处理器。从而导致一个 CPU 过载，而其他 CPU 却很空闲。通过 mpstat 可以轻松诊断这些类型的问题。


针对 Oracle 用户的用法
   与 vmstat 相似，mpstat 命令还产生与 CPU 有关的统计信息，因此所有与 CPU 问题有关的讨论也都适用于 mpstat。
当您看到较低的 %idle 数字时，您知道出现了 CPU 不足的问题。当您看到较高的 %iowait 数字时，
您知道在当前负载下 I/O 子系统出现了某些问题。该信息对于解决 Oracle 数据库性能问题非常方便。

dstat工具的安装和使用
https://www.jianshu.com/p/30d0cf662cd2

atop工具检测linux硬件异常
https://www.cnblogs.com/python-cat/p/7346301.html

当我们的服务器出现问题的时候，外在的表现是业务功能不能正常提供，内在的原因，从程序的角度看，可能是业务程序的问题(程序自身的bug)，也可能是服务器上人为的误操作(不当地执行脚本或命令)；从系统资源的角度看，可能是CPU抢占、内存泄漏、磁盘IO读写异常、网络异常等。出现问题后，面对各种各样可能的原因，我们应如何着手进行分析？我们有什么工具进行问题定位吗？

我们知道CPU可被用于执行进程、处理中断，也可处于空闲状态(空闲状态分两种，一种是活动进程等待磁盘IO导致CPU空闲，另一种是完全空闲)

SLAB是Linux操作系统的一种内存分配机制。其工作是针对一些经常分配并释放的对象，您可以看看哪些应用进程的slab占用的内存比较多，是否这些应用需要频繁的请求和释放内存，比如进行一些小文件的读写。如果都是应用的正常使用，可以考虑升级服务器内存，如果内存不足影响业务，需要临时释放一下slab占用的内存，

可以参考以下步骤： #echo 2 > /proc/sys/vm/drop_caches
等内存回收完毕后再 #echo 0 > /proc/sys/vm/drop_caches
其中drop_caches的4个值有如下含义：
0：不做任何处理，由系统自己管理 1：清空pagecache 2：清空dentries和inodes 3：清空pagecache、dentries和inodes

$sync 
$sudo sysctl -w vm.drop_caches=3 
$sudo sysctl -w vm.drop_caches=0 #recovery drop_caches 
操作后可以通过sudo sysctl -a | grep drop_caches查看是否生效。

通过观察/proc/meminfo发现，slab内存分为两部分：

SReclaimable // 可回收的slab
SUnreclaim // 不可回收的slab

3. 可以通过以下方法调大这个阈值：将vm.extra_free_kbytes设置为vm.min_free_kbytes和一样大，则/proc/zoneinfo中对应的low阈值就会增大一倍，同时high阈值也会随之增长，以此类推。

$ sudo sysctl -a | grep free_kbytes       
vm.min_free_kbytes = 39847
vm.extra_free_kbytes = 0
$ sudo sysctl -w vm.extra_free_kbytes=836787  ######1GB
####系统中 没有vm.extra_free_kbytes 这个参数,修改下面的参数
$ /sbin/sysctl -w vm.min_free_kbytes=836787

4. 举个例子，当low阈值被设置为1GB的时候，当系统free的内存小于1GB时，观察到kswapd进程开始工作（进程状态从Sleeping变为Running），同时dcache开始被系统回收，直到系统free的内存介于low阈值和high阈值之间，停止回收。

Linux服务器Cache占用过多内存导致系统内存不足问题的排查解决
https://www.cnblogs.com/panfeng412/p/drop-caches-under-linux-system.html
Linux服务器Cache占用过多内存导致系统内存不足问题的排查解决（续）
https://www.cnblogs.com/panfeng412/p/drop-caches-under-linux-system-2.html

 首先，弄清楚dentry_cache的概念及作用：目录项高速缓存，是Linux为了提高目录项对象的处理效率而设计的；它记录了目录项到inode的映射关系。因此，当应用程序发起stat系统调用时，就会创建对应的dentry_cache项（更进一步，如果每次stat的文件都是不存在的文件，那么总是会有大量新的dentry_cache项被创建）。
 

 
 LIBRARY_PATH和LD_LIBRARY_PATH环境变量的区别
 https://www.cnblogs.com/panfeng412/archive/2011/10/20/library_path-and-ld_library_path.html
 LIBRARY_PATH和LD_LIBRARY_PATH是Linux下的两个环境变量，二者的含义和作用分别如下：

LIBRARY_PATH环境变量用于在程序编译期间查找动态链接库时指定查找共享库的路径，例如，指定gcc编译需要用到的动态链接库的目录。设置方法如下（其中，LIBDIR1和LIBDIR2为两个库目录）：

export LIBRARY_PATH=LIBDIR1:LIBDIR2:$LIBRARY_PATH
LD_LIBRARY_PATH环境变量用于在程序加载运行期间查找动态链接库时指定除了系统默认路径之外的其他路径，注意，LD_LIBRARY_PATH中指定的路径会在系统默认路径之前进行查找。设置方法如下（其中，LIBDIR1和LIBDIR2为两个库目录）：

export LD_LIBRARY_PATH=LIBDIR1:LIBDIR2:$LD_LIBRARY_PATH
举个例子，我们开发一个程序，经常会需要使用某个或某些动态链接库，为了保证程序的可移植性，可以先将这些编译好的动态链接库放在自己指定的目录下，然后按照上述方式将这些目录加入到LD_LIBRARY_PATH环境变量中，这样自己的程序就可以动态链接后加载库文件运行了。
区别与使用： 

开发时，设置LIBRARY_PATH，以便gcc能够找到编译时需要的动态链接库。

发布时，设置LD_LIBRARY_PATH，以便程序加载运行时能够自动找到需要的动态链接库。

epoll基本原理及使用框架
https://www.cnblogs.com/panfeng412/articles/2229095.html

epoll是Linux下多路复用IO接口select/poll的增强版本，它能显著减少程序在大量并发连接中只有少量活跃的情况下的系统CPU利用率，因为它不会复用文件描述符集合来传递结果而迫使开发者每次等待事件之前都必须重新准备要被侦听的文件描述符集合，另一点原因就是获取事件的时候，它无须遍历整个被侦听的描述符集，只要遍历那些被内核IO事件异步唤醒而加入Ready队列的描述符集合就行了。epoll除了提供select/poll 那种IO事件的电平触发（Level Triggered）外，还提供了边沿触发（Edge Triggered），这就使得用户空间程序有可能缓存IO状态，减少epoll_wait/epoll_pwait的调用，提高应用程序效率。


 Linux环境下段错误的产生原因及调试方法小结
 https://www.cnblogs.com/panfeng412/archive/2011/11/06/segmentation-fault-in-linux.html
 
 gcc -g -o segfault3 segfault3.c
  nm segfault3
 使用nm命令列出二进制文件中的符号表，包括符号地址、符号类型、符号名等，这样可以帮助定位在哪里发生了段错误。以程序2.3为例：
 
 ldd ./segfault3
 使用ldd命令查看二进制程序的共享链接库依赖，包括库的名称、起始地址，这样可以确定段错误到底是发生在了自己的程序中还是依赖的共享库中。以程序2.3为例：
 
 简单来说，就是在程序的重要代码附近加上像printf这类输出信息，这样可以跟踪并打印出段错误在代码中可能出现的位置。

为了方便使用这种方法，可以使用条件编译指令#ifdef DEBUG和#endif把printf函数包起来。这样在程序编译时，如果加上-DDEBUG参数就能查看调试信息；否则不加该参数就不会显示调试信息。

ulimit -c
ulimit -c 1024
./segfault3
gdb ./segfault3 ./core 

ulimit -c unlimited
当时这个方法有个弊端是，在那个终端上设置的ulimit就在那个终端上生效，如果你在另一个终端上执行程序，你会发现，纵然有段错误，你也没生成core文件
第二个办法是修改 /root/.bash_profile 文件添加一行  ulimit -S -c  unlimited 然后保存关闭文件。
有很多 linux 系统 /root 目录下并没有 .bash_profile 文件，比如 SUSE，这没有关系你自己 vi  创建这个文件即可。
修改完这个文件之后，执行 source /root/.bash_profile，你就可以查看你的修改生效了没有。
查看方法是 ulimit -c.你会发现，终端打印出 unlimited，表示你的配置生效了，你可以新开终端 执行 ulimit -c，发现新开终端也是unlimited 。

如echo “/corefile/core-%p-%e-%t” > /proc/sys/kernel/core_pattern
这个语句的含义是将core文件生成在 /corefile/这个目录下
生成的文件名的格式是：“core”-“pid”-可执行程序名-段错误时间
%p ---------段错误进程的PID 
%e-----------发生段错误的可执行文件名
%t------- 发生段错误的时间

gdb
run
l
q



[3556306.638909] segfault3[17334]: segfault at 5607116686a4 ip 000056071166860d sp 00007ffdd874bd50 error 7 in segfault3[560711668000+1000]
objdump -d ./segfault3 > segfault3Dump
 grep -n -A 10 -B 10 "80484e0" ./segfault3Dump 
 

1、出现段错误时，首先应该想到段错误的定义，从它出发考虑引发错误的原因。
2、在使用指针时，定义了指针后记得初始化指针，在使用的时候记得判断是否为NULL。
3、在使用数组时，注意数组是否被初始化，数组下标是否越界，数组元素是否存在等。
4、在访问变量时，注意变量所占地址空间是否已经被程序释放掉。
5、在处理变量时，注意变量的格式控制是否合理等。


binutils工具集之---addr2line
https://www.cnblogs.com/yangguang-it/p/6435297.html

通过 sudo cat /var/log/messages |grep segfault 或者 sudo dmesg|grep segfault 获得
这种信息一般都是由内存访问越界造成的，不管是用户态程序还是内核态程序访问越界都会出core, 并在系统日志里面输出一条这样的信息。这条信息的前面分别是访问越界的程序名，进程ID号，访问越界的地址以及当时进程堆栈地址等信息，比较有用的信息是 最后的error number. 在上面的信息中，error number是4 ,下面详细介绍一下error number的信息： 
在上面的例子中，error number是6, 转成二进制就是110, 即bit2=1, bit1=1, bit0=0, 按照上面的解释，我们可以得出这条信息是由于用户态程序读操作访问越界造成的。 
error number是由三个字位组成的，从高到底分别为bit2 bit1和bit0,所以它的取值范围是0~7. 

bit2: 值为1表示是用户态程序内存访问越界，值为0表示是内核态程序内存访问越界 
bit1: 值为1表示是写操作导致内存访问越界，值为0表示是读操作导致内存访问越界 
bit0: 值为1表示没有足够的权限访问非法地址的内容，值为0表示访问的非法地址根本没有对应的页面，也就是无效地址 
如：
Nov  3 09:27:50 ip-172-31-18-210 kernel: [12702742.866113] nginx[53350]: segfault at 1b852e2 ip 00007f9085b3b616 sp 00007ffdf15f1368 error 4 in libc-2.17.so[7f90859ec000+1b7000]
nginx[31752]: segfault at 0 ip 000000000047c0d5 sp 00007fff688cab40 error 4 in nginx[400000+845000]

一次segfault错误的排查过程
https://blog.csdn.net/zhaohaijie600/article/details/45246569

xxxxx.o[2374]: segfault at7f0ed0bfbf70 ip 00007f0edd646fe7 sp 00007f0ed3603978 error 4 inlibc-2.17.so[7f0edd514000+1b6000]
7f0ed0bfbf70，00007f0edd646fe7，00007f0ed3603978这三个值：第一个值为出错的地址，用处不大；第二个值为发生错误时指令的地址，这个值在有些错误时是错误的，下面会讲一下，第三个值为堆栈指针。

C++段错误就几类，读写错误，这个主要是参数没有控制好，这种错误比较常见，我们经常把NULL指针、未初始化或非法值的指针传递给函数，从而引出此错误；指令地址出错，这类错误主要是由虚函数，回调函数引起，最常出现的是虚函数，由于虚函数保存在类变量中，如果不小心用了非安全函数，就可能把虚数指针覆盖掉，从而影响出现错误。但指令地址出错的情况相对参数出错来讲还是要少很多的，因为用到此功能的还是需要一定的水平的，不容易犯一些低级错误。

catchsegv ./segfault3

Backtrace:
./segfault3(+0x60d)[0x56022e68960d]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xe7)[0x7f89506f7b97]
./segfault3(+0x51a)[0x56022e68951a]

Backtrace:
./segfault3(+0x60d)[0x5566ba43760d]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xe7)[0x7f087ecf1b97]
./segfault3(+0x51a)[0x5566ba43751a]

[3558016.762963] segfault3[25670]: segfault at 55d2830c86a4 ip 000055d2830c860d sp 00007ffefdbe48a0 error 7 in segfault3[55d2830c8000+1000]
[3558268.899781] segfault3[27978]: segfault at 560bf22f36a4 ip 0000560bf22f360d sp 00007ffc131130c0 error 7 in segfault3[560bf22f3000+1000]

00000000000005fa <main>:
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
int main()
{
 5fa:   55                      push   %rbp
 5fb:   48 89 e5                mov    %rsp,%rbp
    char *ptr = "test";
 5fe:   48 8d 05 9f 00 00 00    lea    0x9f(%rip),%rax        # 6a4 <_IO_stdin_used+0x4>
 605:   48 89 45 f8             mov    %rax,-0x8(%rbp)
    strcpy(ptr, "TEST");
 609:   48 8b 45 f8             mov    -0x8(%rbp),%rax
 60d:   c7 00 54 45 53 54       movl   $0x54534554,(%rax)
 613:   c6 40 04 00             movb   $0x0,0x4(%rax)
    return 0;
 617:   b8 00 00 00 00          mov    $0x0,%eax
}
 61c:   5d                      pop    %rbp
 61d:   c3                      retq
 61e:   66 90                   xchg   %ax,%ax



readelf与objdump
http://harmonyhu.com/2018/06/02/objdump/
Bash shell脚本
http://harmonyhu.com/2018/05/25/linux-shell/

python3 -c "print((0x0000560bf22f360d - 0x560bf22f3000).to_bytes(4, 'big').hex())"

查看头信息(-h) readelf -h mytest
查看段信息(-S) readelf -S mytest
查看符号(-s) readelf -s mytest
查看依赖库(-d) readelf -d mytest

objdump -s test.so 显示section的完整内容（所有section）
objdump --section=.text -s test.so 显示test.so中的.text段的内容
objdump -d test.so 反汇编出具有特定指令的section （精简）
objdump -D test.so 反汇编出所有的section （全集）
objdump -S test.so 反汇编（尽可能输出源代码，一般-g下回比较明显有用）


Linux下磁盘分析
http://harmonyhu.com/2017/09/14/IO-TEST/

df -h
fdisk -l
硬盘健康度：smartctl -A /dev/sda
监控io状态：iostat -d 2 10，iostat -d -x 1 | grep -E (Device|sda)
坏道检测：badblocks -v /dev/sda
读取速度验证：hdparm -tT /dev/sda
块数据拷贝：dd if=/dev/sda iflag=direct bs=4k of=/dev/zero
性能检测：fio --ioengine=libaio --randrepeat=0 --norandommap --thread --direct=1 --stonewall --group_reporting --name=mytest --ramp_time=60 -runtime=600 --numjobs=8 --iodepth=32 --filename=/dev/sda --rw=randread --bs=4k

磁盘性能指标
IOPS: (Input/Output Per Second)，每秒IO操作次数。处理器单位时间内能处理的I/O请求数量，尤其在随机读写应用中，该指标非常重要。
IO带宽: 单位时间内操作数据的数量，也可以说是单位时间磁盘I/O的流量。在顺序读写应用中，该指标非常重要。IO带宽=IOPS*单次IO请求量。

/dev/null ： 在类Unix系统中，/dev/null，或称空设备，是一个特殊的设备文件，它丢弃一切写入其中的数据（但报告写入操作成功），读取它则会立即得到一个EOF。 在程序员行话，尤其是Unix行话中，/dev/null 被称为位桶(bit bucket)或者黑洞(black hole)。空设备通常被用于丢弃不需要的输出流，或作为用于输入流的空文件。这些操作通常由重定向完成。

/dev/zero ： 在类UNIX 操作系统中, /dev/zero 是一个特殊的文件，当你读它的时候，它会提供无限的空字符(NULL, ASCII NUL, 0x00)。 其中的一个典型用法是用它提供的字符流来覆盖信息，另一个常见用法是产生一个特定大小的空白文件。BSD就是通过mmap把/dev/zero映射到虚地址空间实现共享内存的。可以使用mmap将/dev/zero映射到一个虚拟的内存空间，这个操作的效果等同于使用一段匿名的内存（没有和任何文件相关）。

Linux常用命令
http://harmonyhu.com/2018/07/08/linux-cmd/

init 0 或 poweroff: 关机
init 6 或 reboot: 重启
passwd username: 修改username密码，比如修改root密码，passwd root
find /home -name file: 在/home目录及子目录查找文件file，可以包含通配符*.
ln -sf abc.sh /usr/bin/abc: 建立软链接，-s表示软链接，-f表示force
diff file1 file2: 比较2个文件
diff folder1 folder2: 比较2个文件夹
cmp file1 file2: 比较二进制，-s安静模式
scp -r username@192.168.1.1:/home/test .: 将远程目录拷贝到本地，-r代表目录
script: 保存终端所有输入输出到当前目录的typescript，退出用exit
ps -aux: 查看所有进程
kill <pid>: 杀掉指定进程
minicom -D /dev/ttyUSB0 -b 115200: 开启usb串口
du -h -d 1 . : 查看当前目录的各个文件(夹)大小
chown -R user:user * : 将当前目录以及子目录的用户都改为user
readlink file : 查看软链接file的路径
last | grep 'shutdown\|reboot' : 查看上一次重启时间
tar -xzvf file.tar.gz : 解压tar.gz
tar –czf jpg.tar.gz *.jpg : 将所有的jpg图片压缩到文件jpg.tar.gz中

BTrace : Java 线上问题排查神器
https://cloud.tencent.com/developer/article/1015128

BTrace 是什么

BTrace 是检查和解决线上的问题的杀器，BTrace 可以通过编写脚本的方式，获取程序执行过程中的一切信息，并且，注意了，不用重启服务，是的，不用重启服务。写好脚本，直接用命令执行即可，不用动原程序的代码。

原理

总体来说，BTrace 是基于动态字节码修改技术(Hotswap)来实现运行时 java 程序的跟踪和替换。大体的原理可以用下面的公式描述：Client(Java compile api + attach api) + Agent（脚本解析引擎 + ASM + JDK6 Instumentation） + Socket其实 BTrace 就是使用了 java attach api 附加 agent.jar ，然后使用脚本解析引擎+asm来重写指定类的字节码，再使用 instrument 实现对原有类的替换。

vmstat 

▪ procs：r这一列显示了多少进程在等待cpu，b列显示多少进程正在不可中断的休眠（等待IO）。

▪ memory：swapd列显示了多少块被换出了磁盘（页面交换），剩下的列显示了多少块是空闲的（未被使用），多少块正在被用作缓冲区，以及多少正在被用作操作系统的缓存。

▪ swap：显示交换活动：每秒有多少块正在被换入（从磁盘）和换出（到磁盘）。

▪ io：显示了多少块从块设备读取（bi）和写出（bo）,通常反映了硬盘I/O。

▪ system：显示每秒中断(in)和上下文切换（cs）的数量。

▪ cpu：显示所有的cpu时间花费在各类操作的百分比，包括执行用户代码（非内核），执行系统代码（内核），空闲以及等待IO。

内存不足的表现：free  memory急剧减少，回收buffer和cacher也无济于事，大量使用交换分区（swpd）,页面交换（swap）频繁，读写磁盘数量（io）增多，缺页中断（in）增多，上下文切换（cs）次数增多，等待IO的进程数（b）增多，大量CPU时间用于等待IO（wa）

iostat

常见linux的磁盘IO指标的缩写习惯：rq是request,r是read,w是write,qu是queue，sz是size,a是verage,tm是time,svc是service。

▪rrqm/s和wrqm/s：每秒合并的读和写请求，“合并的”意味着操作系统从队列中拿出多个逻辑请求合并为一个请求到实际磁盘。

▪r/s和w/s：每秒发送到设备的读和写请求数。

▪rsec/s和wsec/s：每秒读和写的扇区数。

▪avgrq –sz：请求的扇区数。

▪avgqu –sz：在设备队列中等待的请求数。

▪await：每个IO请求花费的时间。

▪svctm：实际请求（服务）时间。

▪%util：至少有一个活跃请求所占时间的百分比。


▲lsof

lsof(list open files)是一个列出当前系统打开文件的工具。通过lsof工具能够查看这个列表对系统检测及排错，常见的用法：

查看文件系统阻塞  lsof /boot

查看端口号被哪个进程占用   lsof  -i : 3306

查看用户打开哪些文件   lsof –u username

查看进程打开哪些文件   lsof –p  4838

查看远程已打开的网络链接  lsof –i @192.168.34.128

从小白到大师，这里有一份Pandas入门指南
https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650768067&idx=4&sn=a7d8431040e066610c66aa58d5c86ea8

istio简介和基础组件原理（服务网格Service Mesh）
https://blog.csdn.net/luanpeng825485697/article/details/84560659

虽然微服务对开发进行了简化，通过将复杂系统切分为若干个微服务来分解和降低复杂度，使得这些微服务易于被小型的开发团队所理解和维护。但是，复杂度并非从此消失。微服务拆分之后，单个微服务的复杂度大幅降低，但是由于系统被从一个单体拆分为几十甚至更多的微服务， 就带来了另外一个复杂度：微服务的连接、管理和监控。试想， 对于一个大型系统， 需要对多达上百个甚至上千个微服务的管理、部署、版本控制、安全、故障转移、策略执行、遥测和监控等，谈何容易。更不要说更复杂的运维需求，例如A/B测试，金丝雀发布，限流，访问控制和端到端认证。开发人员和运维人员在单体应用程序向分布式微服务架构的转型中， 不得不面临上述挑战。

找个Spring Cloud或者Dubbo的成熟框架，直接搞定服务注册，服务发现，负载均衡，熔断等基础功能。然后自己开发服务路由等高级功能， 接入Zipkin等Apm做全链路监控，自己做加密、认证、授权。 想办法搞定灰度方案，用Redis等实现限速、配额。 诸如此类，一大堆的事情， 都需要自己做，无论是找开源项目还是自己操刀，最后整出一个带有一大堆功能的应用程序，上线部署。然后给个配置说明到运维，告诉他说如何需要灰度，要如何如何， 如果要限速，配置哪里哪里。这些工作，相信做微服务落地的公司，基本都跑不掉，需求是现实存在的，无非能否实现，以及实现多少的问题，但是毫无疑问的是，要做到这些，绝对不是一件容易的事情。问题是，即使费力做到这些事情到这里还没有完：运维跑来提了点要求，在他看来很合理的要求，比如说：简单点的加个黑名单， 复杂点的要做个特殊的灰度：将来自iPhone的用户流量导1%到Stagging环境的2.0新版本……

这里就有一个很严肃的问题， 给每个业务程序的开发人员: 你到底想往你的业务程序里面塞多少管理和运维的功能? 就算你hold的住技术和时间，你有能力一个一个的满足各种运维和管理的需求吗？ 当你发现你开始疲于响应各种非功能性的需求时，就该开始反省了: 我们开发的是业务程序，它的核心价值在业务逻辑的处理和实现，将如此之多的时间精力花费在这些非业务功能上， 这真的合理吗? 而且即使是在实现层面，微服务实施时，最重要的是如何划分微服务，如何制定接口协议，你该如何分配你有限的时间和资源？

控制平面的特点：

不直接解析数据包
与控制平面中的代理通信，下发策略和配置
负责网络行为的可视化
通常提供API或者命令行工具可用于配置版本化管理，便于持续集成和部署

数据平面的特点：

通常是按照无状态目标设计的，但实际上为了提高流量转发性能，需要缓存一些数据，因此无状态也是有争议的
直接处理入站和出站数据包，转发、路由、健康检查、负载均衡、认证、鉴权、产生监控数据等
对应用来说透明，即可以做到无感知部署

Service Mesh的出现，弥补了Kubernetes在微服务的连接、管理和监控方面的短板，为Kubernetes提供更好的应用和服务管理。

“K8S实际上只是负责基本的调度和编排，负责主机上容器的运行，Istio做的是互补，因为它主要做的是服务发现。当用容器来实现不同的服务时，需要发现服务、动态连接服务、进行服务升级，甚至还会产生一些安全性、可靠性、性能方面的功能需求，实际上这些功能都是K8S原生没有或者要用手动的方法才能实现的。Istio就是一个可以在K8S上，把这些应用、服务进行较好管理的框架。”

梁胜认为，K8S本身的调度编排功能不会被Istio取代，除非将来有更好的调度编排。而且Istio与K8S并不是紧耦合的，除了K8S，Istio实际上也可以与Mesos、Consol等其他的框架来整合

通过以上对Istio的实践大家不难看出Istio相比于Spring Cloud的几个优点。首先相比于Spring Cloud学习组件内容多，门槛高的痛点，Istio是非常容易上手的，只需在Kubernetes平台上跑一个yaml文件即可完成部署；再者相比于Spring Cloud需要把认证授权、分布式追踪、监控等这些高级功能加入到应用程序内部导致了应用本身复杂度的痛点，反观Istio是很轻量级的，它将以上那些高级功能作为组件内置在Istio中，从而达到了对应用程序的无侵入性，只需要配置相应的yaml文件并下发至Istio控制平面执行后续的操作即可，操作过程用户是无感知的；最后相比于Spring Cloud对Java环境的过度依赖以及跨语言痛点，Istio也完美的解决了，其支持多种语言，包括新兴编程语言Golang、Rust、Node.js、R语言等。

用Light-locker淘汰掉Xscreensaver并在suspend情况下自动锁屏
https://zhouxiaobo.wordpress.com/2017/07/06/%E7%94%A8light-locker%E6%B7%98%E6%B1%B0%E6%8E%89xscreensaver%E5%B9%B6%E5%9C%A8suspend%E6%83%85%E5%86%B5%E4%B8%8B%E8%87%AA%E5%8A%A8%E9%94%81%E5%B1%8F/


Disabling screen lock is not possible
https://bugs.launchpad.net/ubuntu/+source/light-locker/+bug/1287255

Convert ISO-8859-1 to UTF-8 [duplicate]

in your case you could simply use:
    utfstring = unescape(encodeURIComponent(ajaxreturn));
Edit: if this does not work either, try the other way round:
    fixedstring = decodeURIComponent(escape(ajaxreturn));
    
    
    
初级多在写代码，高级多在设计代码；
初级多在解决一个问题，高级多在解决一类问题；
初级多在考虑技术问题，高级还要参与业务上的需求；
初级工程师只管接需求，导致自己忙不过来，高级工程师会砍需求， 用自己得经验告诉产品这个需求不需要，告诉设计师这个交互没必要；
初级工程师可能做完一个项目就完了，高级工程师可能会封装几个组件，整理一个脚手架出来。    

深度解析GraphQL：澄清你对GraphQL的误解
https://mp.weixin.qq.com/s?__biz=MzUxMzcxMzE5Ng==&mid=2247492167&idx=1&sn=43004262fd2bd2610bd218fd7302555a

格林斯潘第十定律：

任何 C 或 Fortran 程序复杂到一定程度之后，都会包含一个临时开发的、不合规范的、充满程序错误的、运行速度很慢的、只有一半功能的 Common Lisp 实现。
任何接口设计复杂到一定程度后，都会包含一个临时开发的、不合规范的、只有一半功能的 GraphQL 实现。

从 SearchParams， FormData 到 JSON，再到 GraphQL 查询语句，我们看到不断有新的数据通讯方式出现，满足不同的场景和复杂度的要求。

站在这个层面上看，GraphQL 模式的出现，有一定的必然性。

我们可以做一个思想实验。
假设你是一名架构师，你接到一项任务，设计一门前端友好的查询语言。要求：
查询语法跟查询结果相近。
能精确查询想要的字段。
能合并多个请求到一个查询语句。
无接口版本管理问题。
代码即文档。

Electron 中文文档
https://www.w3cschool.cn/electronmanual/
用Electron写个带界面的nodejs爬虫
https://blog.csdn.net/u011510678/article/details/80007028

electron安装+运行+打包成桌面应用+打包成安装文件+开机自启动
https://www.cnblogs.com/kakayang/p/9559777.html

MOOC 网站：Coursera、Udacity、edX，哪个更适合中国人？你有何经验分享？
https://www.zhihu.com/question/21095181/answer/21891006


浅谈订单号生成设计方案
https://mp.weixin.qq.com/s?__biz=MzIwODA4NjMwNA==&mid=2652899180&idx=1&sn=1f8c05f3f3b369aed0839c15b24fc58f


# 编程入门教程

- [官方教程](https://docs.python.org/zh-cn/3/tutorial/index.html)
- [笨办法学 Python](https://www.kancloud.cn/kancloud/learn-python-hard-way/49863)
- [菜鸟教程：Python 基础教程](https://www.runoob.com/python/python-tutorial.html)
- [Python 101](http://www.davekuhlman.org/python_101.html）
- [廖雪峰 Python 教程](https://www.liaoxuefeng.com/wiki/1016959663602400)
- [慕课网](http://www.imooc.com/learn/177)
- http://c.biancheng.net/python/


如何零基础开始自学Python编程
https://www.cnblogs.com/shwee/p/9071829.html

为什么很多Python开发者写GUI不用Tkinter，而要选择PyQt和wxPython或其他？
https://www.zhihu.com/question/32703639

Python GUI之tkinter概述
https://blog.csdn.net/yingshukun/article/details/53985080
https://blog.csdn.net/yingshukun/column/info/37999

程序员的C
https://blog.csdn.net/yingshukun/article/category/8978571

The Tkinter Grid Geometry Manager
http://www.effbot.org/tkinterbook/grid.htm

Python GUI进阶(ttk)—让界面变得更美
https://blog.csdn.net/i_chaoren/article/details/56296713

Does tkinter have a table widget?
https://stackoverflow.com/questions/9348264/does-tkinter-have-a-table-widget

7月在线 Python
https://www.julyedu.com/course/getDetail/148



How to build a parse tree of a mathematical expression?
https://stackoverflow.com/questions/24520841/how-to-build-a-parse-tree-of-a-mathematical-expression

数学表达式转换成树形
https://github.com/carlos-chaguendo/arboles-binarios/tree/master

如何定义一个好的变量名
https://www.techug.com/post/how-to-give-var-a-good-name.html
https://www.cnblogs.com/lijianwu/p/5715224.html

https://dreampuf.github.io/GraphvizOnline/

互联网上的免费书籍
https://github.com/ruanyf/free-books#web-%E5%BC%80%E5%8F%91


数据架构选型必读：8 月数据库产品技术解析
https://www.infoq.cn/article/50xDgQVcjQdfFMwDGPM9


hdfs namenode  -initializeSharedEdits
 将所有journal node的元文件的VERSION文件的参数修改成与namenode的元数据相同


hdfs namenode -bootstrapStandby
  将active namenode的 {dfs.namenode.name.dir} 目录的内容复制到 standby namenode的{dfs.namenode.name.dir} 目录下
  
  
300来行代码带你实现一个能跑的最小Linux文件系统
https://mp.weixin.qq.com/s?__biz=MzI3NzA5MzUxNA==&mid=2664606817&idx=1&sn=94cd2ebe02cd0959382fd3396a8352f4

使用 Redis 储存社交关系
https://mp.weixin.qq.com/s?__biz=MzUzMTk4Mjk5Ng==&mid=2247483747&idx=1&sn=bd653e155ba84f014963de4bc8efe33b

学习 Python 的 12 个方式 | Linux 中国
https://mp.weixin.qq.com/s?__biz=MzI1NDQwNDYyMg==&mid=2247487634&idx=1&sn=0f4e7c154be3bad02da52e0533ec9db8

make -j$JOBS > build.log 2>&1 || (cat build.log && exit 1)  

purpose just in case


搭建基于 IDEA 的 Scala 开发环境
https://www.oschina.net/question/54100_12783

https://www.jetbrains.com/idea/download/index.html#section=windows
https://plugins.jetbrains.com/plugin/1347-scala

Best way to integrate Python and JavaScript?
https://stackoverflow.com/questions/683462/best-way-to-integrate-python-and-javascript

Python in the browser, precompiled for speed: http://www.transcrypt.org
https://github.com/qquick/Transcrypt

Intellij之Spark Scala开发环境搭建
https://www.jianshu.com/p/200473f264bc

tar xf /helloworld-data/download/ideaIC-2019.2.1.tar.gz -C /opt/
unzip  /helloworld-data/download/scala-intellij-bin-2019.2.20.zip -d /opt/idea-IC-192.6262.58/plugins/
tar xf /helloworld-data/scala-2.11.12.tgz -C /usr/local/
mv /usr/local/scala-2.11.12/ /usr/local/scala
echo 'export PATH=/usr/local/scala/bin:$PATH' >>/etc/profile




Intellij Idea搭建Spark开发环境
https://blog.csdn.net/u012877472/article/details/51000690



Spark runs on Java 8+, Python 2.7+/3.4+ and R 3.1+. For the Scala API, Spark 2.3.2 uses Scala 2.11. You will need to use a compatible Scala version (2.11.x).



Effective Java（第3版）各章节的中英文学习参考（已完成）
https://github.com/clxering/Effective-Java-3rd-edition-Chinese-English-bilingual

计算机开放电子书汇总
https://segmentfault.com/a/1190000004612033
飞龙的计算机公开课推荐
https://segmentfault.com/a/1190000004317463

数据科学/人工智能比赛解决方案汇总
https://github.com/apachecn/awesome-data-comp-solution/blob/master/README.md

用一段话简单介绍一下ApacheCN
https://www.ibooker.org.cn/organization/

8、 关于 ApacheCN，你们还有什么想说的？请畅所欲言：）
我觉得我们真的没什么特别的地方，
我们都是一群普普通通的一群爱吃辣条、爱分享的人，
我们没有靠什么背景，也没有什么资源，分享的东西也没有多么牛逼，
甚至我们分享的教程、比赛、也没有得过什么大奖（当然群友有一些牛逼的），
但我们是真心愿意高标准要求自己，低姿态接纳小白，
把学到的知识真心的去分享给他们、帮助他们、让他们少走弯路，并带他们一起牛逼。
虽然在GitHub上面被网友力推，得到全球组织排名119，我感觉的只是我们愿意奉献而已。
期待更多的人加入ApacheCN或者和ApacheCN一起推动国内的知识开源，让开源更健康成长。


SELECT (unix_timestamp('2013-01-01 10:10:10') - unix_timestamp('1970-01-01 00:00:00'))/60 ；
SELECT from_unixtime(unix_timestamp('2013-01-01 10:10:10') + 10 * 60) AS result ;


Python Cookbook 3rd Edition Documentation
https://python3-cookbook.readthedocs.io/zh_CN/stable/

使用ReadtheDocs托管文档
https://www.xncoding.com/2017/01/22/fullstack/readthedoc.html

Running Python in the Browser
https://pythontips.com/2019/05/22/running-python-in-the-browser/

如何计算三角形面积
https://zh.wikihow.com/%E8%AE%A1%E7%AE%97%E4%B8%89%E8%A7%92%E5%BD%A2%E9%9D%A2%E7%A7%AF

高校教师/自由翻译
https://about.ac/
https://about.ac/2015/02/translation-process.html

docker lib 迁移，默认在系统盘占太大空间

systemctl stop docker
mkdir -p /data/docker/lib
rsync -avz /var/lib/docker /data/docker/lib/

sudo mkdir -p /etc/systemd/system/docker.service.d/
sudo vi /etc/systemd/system/docker.service.d/devicemapper.conf

    [Service]
    ExecStart=
    ExecStart=/usr/bin/dockerd  --graph=/data/docker/lib/docker

systemctl daemon-reload 
systemctl restart docker 
systemctl enable docker
docker info

    Docker Root Dir: /data/docker/lib/docker
docker images    

rm -rf /var/lib/docker/


彻底解决Linux索引节点(inode)用满导致故障的方法
df -i
for i in /*; do echo $i; find $i |wc -l|sort -nr; done 



要停止Tomcat，运行
sudo /etc/init.d/tomcat6 stop
Tomcat 配置文件路径
Tomcat home directory : /usr/share/tomcat8
Tomcat base directory : /var/lib/tomcat6或/etc/tomcat8


Tomcat重新加载jar
1.在tomcat/common/lib下的jar文件，若更新或新增了，则只能重启服务器，才能重新加载jar包，使jar包生效。

2.如果application的WEB-INF/lib下的jar文件更新，则可以不重启tomcat便能使之生效，做法是修改application的Context，修改其reloadable属性为true，（如果没有该属性就添加），该属性默认是false。  

例如：

<Context path="/myweb" docBase="D:\workplace\myweb\WebRoot"
  debug="5" reloadable="true" crossContext="true"/>
  
http://shop.oreilly.com/product/9780596007973.do

How to set JAVA_HOME for multiple Tomcat instances?

place a setenv.sh in the the bin directory with

JAVA_HOME=/usr/java/jdk1.6.0_43/
JRE_HOME=/usr/java/jdk1.6.0_43/jre  


使用Spark读写CSV格式文件
https://www.iteblog.com/archives/1380.html

## spark

import org.apache.spark.{ SparkConf, SparkContext }
 
object WordCount{
   def main(args: Array[String]): Unit = {
       var conf = new SparkConf().setMaster("local").setAppName("wordcount")
       var sc = new SparkContext(conf)
       System.setProperty("hadoop.home.dir", "/usr/local/hadoop")
       var input = "/usr/local/spark/README.md"
       var count = sc.textFile(input).flatMap(x => x.split(" ")).map(x => (x, 1)).reduceByKey((x, y) => x + y)
       count.foreach(x => println(x._1 + "," + x._2))
    }
}



./bin/spark-submit --class 'cn.helloworld.Test' /root/IdeaProjects/wordcount/out/artifacts/wordcount_jar/wordcount.jar

import org.apache.spark.sql.SQLContext
 
val sqlContext = new SQLContext(sc)
val df = sqlContext.load("com.databricks.spark.csv", Map("path" -> "cars.csv", "header" -> "true"))
df.select("year", "model").save("newcars.csv", "com.databricks.spark.csv")


val data = sc.makeRDD(List(("aaa", "18"),("bbb", "19")))
val people = data.map(p=>Person(p._1,p._2.trim.toInt)).toDF
people.show()
people.repartition.write.format("csv").save("/mytest2")


// 读取 csv 并按 \t 分割
val csv = sc.textFile("/root/2.csv").map(_.split("\t"))

// 确认每行的列数是否正确
csv.filter(_.length==23).count
csv.filter(_.length<23).count


// 加载 csv
val df = spark.read.format("csv").option("delimiter", "\t").load("/root/2.csv")

// 去掉最后无效列
val df2 = df.drop("_c23")

// 重新映射列名
val colnames = "a b c d e".split(" ")
val df3 = df2.toDF(colnames: _*)
df3.printSchema

// 转换列的类型
import org.apache.spark.sql.types.IntegerType
val df4 = df3.withColumn("a",$"a".cast(IntegerType))
df4.printSchema

val sql = "SELECT hotel_name,(a-b)/b AS lv FROM `hotels` ORDER BY a DESC, lv DESC LIMIT 10"
val df5 = df4.sqlContext.sql(sql)
df5.show

// 导出到 csv
df5.repartition(1).write.format("csv").save("/root/df5")


Renaming column names of a DataFrame in Spark Scala
https://stackoverflow.com/questions/35592917/renaming-column-names-of-a-dataframe-in-spark-scala

val newNames = Seq("id", "x1", "x2", "x3")
val dfRenamed = df.toDF(newNames: _*)

How to change column types in Spark SQL's DataFrame?
https://stackoverflow.com/questions/29383107/how-to-change-column-types-in-spark-sqls-dataframe

import org.apache.spark.sql.types.IntegerType
val df2 = df.withColumn("yearTmp", df.year.cast(IntegerType))
    .drop("year")
    .withColumnRenamed("yearTmp", "year")
    
val df2 = df.selectExpr("cast(year as int) year", 
                        "make", 
                        "model", 
                        "comment", 
                        "blank")    
                        
                        
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.Duration
import org.apache.spark.streaming.Seconds

val ssc = new StreamingContext(sc, Seconds(1))
val lines = ssc.socketTextStream("localhost", 8888)
val words = lines.flatMap(_.split(" "))
val wordCounts = words.map(x=>(x,1)).reduceByKey(_+_)
wordCounts.print()
ssc.start()


nc -lk -p 8888

## irc
apt-get install ircd-hybrid
apt-get install irssi weechat

IRC 急速入门
http://unifreak.github.io/tutorial/IRC-quick-start


Scrapy 入门教程
https://www.runoob.com/w3cnote/scrapy-detail.html

pip install --upgrade pip
pip install Scrapy
scrapy startproject mySpider
cd mySpider
scrapy genspider jiudian "192.168.1.60"
scrapy.exe crawl jiudian

群晖rsync 备份网站
https://bugging.com.cn/2018/04/11/%E7%BE%A4%E6%99%96rsync-%E5%A4%87%E4%BB%BD%E7%BD%91%E7%AB%99/

rsync 到群晖，启用 rsync，开启帐号 rsync 权限
rsync -e "/usr/bin/ssh" -avzl vmstat.log helloworld@192.168.1.114::4T

如何科学的学习一门编程语言？
https://gitchat.blog.csdn.net/article/details/91965944


基于TLS1.3的微信安全通信协议mmtls介绍
https://mp.weixin.qq.com/s?__biz=MzAwNDY1ODY2OQ==&mid=2649286266&idx=1&sn=f5d049033e251cccc22e163532355ddf

移动端IM/推送系统的协议选型：UDP还是TCP？
http://www.52im.net/thread-33-1-1.html




2.2 介绍一下使用到的ssh参数：
反向代理
ssh -fCNR

正向代理
ssh -fCNL

-f 后台执行ssh指令
-C 允许压缩数据
-N 不执行远程指令
-R 将远程主机(服务器)的某个端口转发到本地端指定机器的指定端口
-L 将本地机(客户机)的某个端口转发到远端指定机器的指定端口
-p 指定远程主机的端口

autossh的参数与ssh的参数是一致的，但是不同的是，在隧道断开的时候，autossh会自动重新连接而ssh不会。另外不同的是我们需要指出的-M参数，这个参数指定一个端口，这个端口是外网的B机器用来接收内网A机器的信息，如果隧道不正常而返回给A机器让他实现重新连接。


nc -vzw 2 baidu.com 80


sudo usermod -aG docker $USER
sudo setfacl -m user:$USER:rw /var/run/docker.sock


如何在 Ubuntu 18.04 下正确配置网络
https://www.hi-linux.com/posts/49513.html

在 Ubuntu 18.04 中如果再通过原来的 ifupdown 工具包继续在 /etc/network/interfaces 文件里配置管理网络接口是无效的。

/etc/resolvconf/resolv.conf.d/base
/etc/systemd/resolved.conf
/etc/resolv.conf
/run/resolvconf/resolv.conf

Netplan
NetworkManager
Systemd-networkd
ifupdown
networkd


人工智能学院本硕博培养体系
https://item.jd.com/12693860.html
人工智能本科专业知识体系与课程设置
https://item.jd.com/12568637.html

一流“新工科”，需工程教育专业认证 “点石成金”
https://baijiahao.baidu.com/s?id=1613858094022498977&wfr=spider&for=pc

新工科，到底是什么？ 
http://www.sohu.com/a/304388578_120103333

应对人工智能挑战，人才培养有方案 
http://www.sohu.com/a/237755690_726454

实践能力调查
https://www.wjx.cn/jq/26770745.aspx


腾讯云在线教育：https://cloud.tencent.com/solution/education
华为云智慧教育：https://e.huawei.com/cn/solutions/industries/education
阿里云在线教育：https://www.aliyun.com/solution/education/eduonline

查找目录下重复的文件

find . -type f -exec md5sum {} \; | awk '{if(md5s[$1]){printf "%s <= %s\n",md5s[$1],$NF}else{md5s[$1]=$0}}'

Relational Data Tools: SQL, Awk, Pig
http://hyperpolyglot.org/data

http://hyperpolyglot.org/  各种语言语法对比



工程教育认证标准
http://www.ceeaa.org.cn/main!newsList4Top.w?menuID=01010702

5.1与本专业毕业要求相适应的数学与自然科学类课程（至少占总学分的15%）。
5.2符合本专业毕业要求的工程基础类课程、专业基础类课程与专业类课程（至少占总学分的30%）。工程基础类课程和专业基础类课程能体现数学和自然科学在本专业应用能力培养，专业类课程能体现系统设计和实现能力的培养。
5.3工程实践与毕业设计（论文）（至少占总学分的20%）。设置完善的实践教学体系，并与企业合作，开展实习、实训，培养学生的实践能力和创新能力。毕业设计（论文）选题要结合本专业的工程实际问题，培养学生的工程意识、协作精神以及综合应用所学知识解决实际问题的能力。对毕业设计（论文）的指导和考核有企业或行业专家参与。
5.4人文社会科学类通识教育课程（至少占总学分的15%），使学生在从事工程设计时能够考虑经济、环境、法律、伦理等各种制约因素。


准确率(Accuracy), 精确率(Precision), 召回率(Recall)和F1-Measure
https://www.cnblogs.com/sddai/p/5696870.html
自己理解 + 我老师的说法就是，准确率就是找得对，召回率就是找得全。


set([line.strip('12345667890?. \r\n') for line in open(r'D:\helloworld\me\全部学生名单.txt')]) - set([line.split(',')[5].strip('"') for line in open(r'D:\helloworld\me\参与问卷学生名单.csv', encoding='utf-8')][1:])


命令行 IRC 客户端 irssi 的基本操作
https://www.cnblogs.com/tsdxdx/p/7291877.html


$ irssi

登陆 IRC 服务器 /connect irc.freenode.net
修改昵称 /nick nickname
直接带昵称登录指定的服务器 irssi -c irc.freenode.net -p 6667 -n zhang_san
进入公开频道 /join #archlinux-cn
进入密码频道 /join #channel password
退出当前频道 /wc
退出指定的服务器 /disconnect irc.freenode.net
退出irssi /quit
窗口切换快捷键：ctrl+n,p 或 Alt+1~0, Alt+q~p
关闭当前窗口 /window close
查看窗口列表 /window list
给某人发送私信 /msg nickname
在频道给指定的人发送消息 /say nickname
屏蔽某人聊天内容 /ignore nickname 
查看频道列表 /list
列出当前服务器或指定聊天室下的所有人员名称 /names #channel
查看频道的所有人 /who
查看某人的基本资料 /whois nickname
查询指定别名是否在线 /ison 别名1 别名2 …
查询服务器信息 /info
查询当前服务器上的 Admin /admin
查询当前服务器上的统计信息 /lusers
查询当前服务器今日的统计信息 /motd
查询当前的服务器 /links
注册 freenode 帐号 /msg NickServ REGISTER [name] [passwd] [email]
申请频道
    - 先试着进入想要申请的频道看看是否已经有人注册
    - /msg ChanServ REGISTER [#channel] [passwd]
/set autolog on，自动保存聊天记录    

第一次使用可以用下面的命令配置irssi

/network add -autosendcmd “/^msg nickserv helloworld 123456;wait 2000″ FREENODE
/server add -auto -network FREENODE irc.freenode.net 6667
/channel add -auto #python FREENODE
这样就可以自动登录了，上面第一命令表明你已经向其服务器注册了帐号了！

注册
/nick nickname
/msg NickServ REGISTER 密码 电子邮件地址
/msg NickServ VERIFY REGISTER helloworld2019 xxxx

屏蔽进入离开等消息 /ignore -channels #channel * JOINS PARTS QUITS NICKS


#python #ubuntu-cn 和 #archlinux-cn @ freenode

自动保存日志 /SET autolog ON
/SET autolog_path ~/irclogs/%Y/$tag/$0.%m-%d.log

#wikipedia-zh

贴图：https://img.vim-cn.com/
贴代码：https://bpaste.net/+python

irc 搜索：https://irc2go.com/

保存记录：/LASTLOG -file ~/irc.log

因为情怀, 有些人依然在使用IRC
https://blog.csdn.net/dietime1943/article/details/79900120

PaaS益处
专有主机托管的扩展性
共享托管的简易性
更快地应用开发和部署
内置N层架构支持
后台是完全可管理的
比起自己拥有服务器来说，更少的前期花费

平台即服务 (PaaS) 是什么？
https://www.oracle.com/cn/cloud/platform/what-is-paas/

To new Irssi users (not to new IRC users ..)
https://irssi.org/documentation/startup/



随着科技的进步，尤其是计算机网络技术的发展，传统的实体考场、有纸化考试已经不适应现代考试的需要，取而代之的是网络化、无纸化、自动化在线的电子化考试。相对传统考试，在线电子化考试系统的优势不言而喻：无需考场，无需纸质试卷，无需繁琐的改卷披卷过程，从而节省了大量的人力物力。中开在线考试系统正是适应时代要求的产物，可以广泛应用于需要开展网络考试测评的学校、企业、政府及军队等单位，并充分考虑了中小企事业单位、大企业、电信企业、高校、中小学校、认证考试机构等不同实体在考试测评上的差异，全面满足其对考试测评的需要，并确实解决其在考试测评中的问题。 

主要功能包括：
题库管理：对考试题库的维护，包括知识点、试题类型、试题难度、试题题目、试题选项个数、试题所有的选项值、正确答案、评分要点等信息；
试卷管理：对考试试卷的维护，包括试卷的试卷分类、出题方式、试卷名称、试卷说明、启用时间、过期时间、对应大题、大题分值、对应小题、小题分值等信息；
考试管理：对具体的某场考试进行管理，包括考试基本信息维护、参考人员安排，考试安排等功能；
群组管理：对参加考试的可能人员进行群组分类，便于在考试安排时可按群组安排；
评卷管理：对考试试卷进行手工评卷，以获得科学、客观的考试结果，试卷支持纯自动评卷、纯手工评卷、自动和人工结合等评卷模式。
我的考试：考试人员登入系统参加考试以及进行考试成绩的查询等；
考试分析：考试管理人员对考试进行多角度、多层次的分析统计，供教学效果评判及调整、考试出题、人员素质分析参考。 

产品特点：
中开在线考试系统具有如下特点：
（1）技术先进：支持高并发，大数据量形式的网络考评；
（2）设计灵活：适应于多种考试题型，考试形式，评卷模式；
（3）操作简单：从试题设置、试卷设置、考试设置、手工评卷，到参加考试，处处体现操作上的人性化；
（4）功能全面：考虑了不同考试测评发起实体的不同需求；
（5）扩展性强：预留了各业务实体的自定义属性，设计了开放式业务接口。 

波若高校大数据实训教学平台流程：

交互式学习模式

提供体系完整、简单易用的在线教学课堂；以基础知识学习，在线视频教学、习题、线上测试、评估等为主线的方法，确保学生在短时间内掌握大数据课程知识及数据分析技能。

大数据实战及案例分析

提供实战案例数据，包括网站流量数据、汽车数据、房屋交易数据、电商商品数据、搜索引擎等多种业务数据，数据超过100T，按周期更新数据内容。

真机实验实训

实验训练体系设计成各模块相对独立的形式，各模块交互式实验任务、大数据实验机、实际项目上机操作，通过多方位的训练，最灵活的、渐进式的掌握大数据生态体系。

充分支撑科研工作

提供行业数据及案例解剖用于基础研究，提供数据分析方案及流程，提供数据更新接口，可以对行业数据进行挖掘分析，按需求生成数据报表，为科研工作提供数据支持。


实训营不止是培训，更要让你听懂，我们只说人话：
1、录播课与直播课结合，不受时间限制，又能得到每周服务
2、助教答疑、进度追踪、作业辅导，过了关才算学会
3、做的案例都是企业真实项目，与真实上班没区别～

相比于其他学习班，XX 学院有着自己的服务体系和课程质量管理，因为学习任何一种编程语言，都需要专业的导师、科学的课程、实战演练项目和学习辅导，这不仅能节省大量时间，还能保证不会半途而废：


tty、pty、pts等(小记)
https://blog.csdn.net/dbzhang800/article/details/6939742

tty teletypewriter (电传打字机) 终端设备(Terminal)
pty pseudo tty 伪终端
pts pseudo-terminal slave
ptmx pseudo-terminal master multiplexer

tty[1-6]就是你用ctr+alt+f[1-6]所看到的那个终端; 即虚拟控制台。其他的是外部终端和网络终端。
pts/*为伪(虚拟)终端, 其中pts/0,1,2在桌面Linux中是标准输入，标准输出，标准出错。


轻量级Python IDE : Thonny
https://www.jianshu.com/p/8d20137027ee
    
Pragmatic Unicode
https://nedbatchelder.com/text/unipain.html

I'll give those people the benefit of the doubt     
im not mocking, it's just cute :)
i initially assumed G-d was some emacs thing
nevermind
Can anyone point me in the right direction?
sounds ambitious.

ganglia 监控没有数据，报 Error creating multicast server mcast_join=239.2.11.71 错误，可以加如下命令后重启 ganglia-monitor 后解决

/sbin/route add -host 239.2.11.71 dev enp3s0

最后一个参数是网卡名称，需要根据实际情况改动


Building an IRC (ro)bot
https://pythonspot.com/building-an-irc-bot/


各位程序员大佬们，请让我听到你们的声音！不管你是前端开发、后端研发、测试开发、移动端开发、全栈工程师、运维工程师、网络工程师、架构师、研发经理还是其他职位，不管你在做Android开发 、iOS开发、U3D 、COCOS2DX、 自动化测试、功能测试、性能测试、白盒测试、灰盒测试、黑盒测试、ETL、数据仓库、数据开发、数据挖掘、数据分析、数据架构、算法研究、精准推荐、分布式、系统集成、地图引擎、人工智能、大数据、深度学习、机器学习、图像处理、图像识别、语音识别、语音学习、机器视觉、自然语言处理、视频开发、区块链还是其它技术支持。欢迎在评论中踊跃发表意见！

最后和程序员身边的朋友们嘱咐一句，就算程序员的工位再乱，也不要帮他整理。否则，当心会毁灭世界哦~

作者：狗哥，一个幽默的理工男青年，愿做你身边最有趣的朋友。公众号：有趣青年（ID：v_danshen）

日志采集系统flume和kafka有什么区别及联系，它们分别在什么时候使用，什么时候又可以结合？
https://www.zhihu.com/question/36688175

Flume 安装测试
https://www.jianshu.com/p/f0a08bd4f975

Kafka 安装测试
https://blog.csdn.net/justry_deng/article/details/88381595


对搭建日志系统而言，也没什么新花样，以下这些依样画葫芦就可以了：1. 日志采集（Logstash/Flume，SDK wrapper）2. 实时消费 (Storm, Flink, Spark Streaming)3. 离线存储 （HDFS or Object Storage or NoSQL) +  离线分析（Presto，Hive，Hadoop）4. 有一些场景可能需要搜索，可以加上ES或ELK



dockerfiles-centos-ssh
https://github.com/CentOS/CentOS-Dockerfiles/tree/master/ssh/centos7

docker systemctl replacement
https://github.com/gdraheim/docker-systemctl-replacement

Cannot access Centos sshd on Docker
https://stackoverflow.com/questions/18173889/cannot-access-centos-sshd-on-docker


加入我们
一起打造出让我们引以为豪、令全世界尖叫的产品。我们是由一群“想要改变现有格局”的年轻人组成的团队。我们心有猛虎、热情澎湃，不甘按部就班的平淡、酷爱恰到好处的精致。我们在时间的领域里深耕，想帮助每一个为忙碌生活所累的人、令他们的时间变得有序而自由。


浅谈几种常见 RAID 的异同
https://blog.csdn.net/Ki8Qzvka6Gz4n450m/article/details/91444893
https://wenku.baidu.com/view/a77a4b6ff56527d3240c844769eae009581ba2e0.html

RAID 0 可用于两个或更多硬盘或 SSD。目标是提高读的性能。
RAID 1 用于创建数据的自动副本。
所谓 RAID 10,其实就是磁盘阵列先 RAID 1,后 RAID 0,同理
RAID 3 是这样的：若有 N 块盘，其中拿出 1 块盘作为校验盘，剩余 N-1 块盘相当于作 RAID 0 同时读写，当 N-1 那里的其中一块盘坏掉时，可以通过校验码还原出坏掉盘的原始数据。
RAID 5 与 RAID 0 一样，数据被分成块并执行写入处理，同时把 RAID 3 的 “校验盘” 也分成块分散到所有的盘里。


一篇读懂BBU
http://bean-li.github.io/raid-bbu/

【mysql】使用脚本对mysql状态进行监控
https://www.cnblogs.com/chenpingzhao/p/5084445.html

mysql磁盘IO%util 居高不下之RAID卡 BBU Learn Cycle周期
https://www.cnblogs.com/Sunnynanbing/p/7780341.html

awk: line 2: function strtonum never defined
https://blog.csdn.net/dpppppp/article/details/53018132

公众号 Markdown 编辑器
https://lab.lyric.im/wxformat/
http://blog.didispace.com/tools/online-markdown/


道业惑：服务教育行业的专业强大的家校沟通工具
http://daoyehuo.com/index.html

记一次tornado QPS 优化
https://segmentfault.com/a/1190000010771366


Apache NiFi 与Falcon/Oozie异同
https://www.jianshu.com/p/1975a7cae634


把物理机转换为虚拟机简单方法
https://blog.csdn.net/sailor201211/article/details/22373937


要使用 dd 和 gzip 生成压缩的镜像文件，可以执行命令： #   dd   bs=512
count=[fdisk命令中最大的end数+1] if=/dev/sda | gzip -6 > /ghost.img.gz


还原时，可以执行下列命令： # gzip -dc /ghost.img.gz.gz | dd of=/dev/sda

KVM 虚拟机在物理主机之间迁移的实现
https://www.linuxidc.com/Linux/2017-03/141274.htm

实现Windows直接远程访问Ubuntu 18.04（旧版本也支持,无需安装第三方桌面,直接使用自带远程工具）
https://www.cnblogs.com/xuliangxing/p/7642650.html

Ubuntu下包含2种远程桌面的方式：VINO-Server以及VNC Server
https://www.cnblogs.com/leaven/archive/2012/03/31/2427240.html

ubuntu 16.04 tightvncserver VNC 安装配置，安装xrdp，x11vnc、Ubuntu 18.04
https://blog.csdn.net/u010953692/article/details/84979965

如何在Ubuntu 18.04上安装TeamViewer
https://www.linuxidc.com/Linux/2018-05/152282.htm

Ubuntu上装KVM：安装、初次使用
https://blog.csdn.net/c80486/article/details/42836169
https://www.jb51.net/article/142818.htm

以免万一，在这提供XP SP3 Vol版本密钥：序列号MRX3F-47B9T-2487J-KWKMF-RPWBY

sudo service network-manager restart
sudo ifdown --exclude=lo -a && sudo ifup --exclude=lo -a
ifconfig eth0 down && ifconfig eth0 up
sudo /etc/init.d/networking restart

实例讲解虚拟机3种网络模式(桥接、nat、Host-only)
https://www.cnblogs.com/ggjucheng/archive/2012/08/19/2646007.html

  桥接网络是指本地物理网卡和虚拟网卡通过VMnet0虚拟交换机进行桥接，物理网卡和虚拟网卡在拓扑图上处于同等地位，那么物理网卡和虚拟网卡就相当于处于同一个网段，虚拟交换机就相当于一台现实网络中的交换机,所以两个网卡的IP地址也要设置为同一网段。

   所以当我们要在局域网使用虚拟机，对局域网其他pc提供服务时，例如提供ftp，提供ssh，提供http服务，那么就要选择桥接模式。
   
NAT模式中，就是让虚拟机借助NAT(网络地址转换)功能，通过宿主机器所在的网络来访问公网。

   NAT模式中，虚拟机的网卡和物理网卡的网络，不在同一个网络，虚拟机的网卡，是在vmware提供的一个虚拟网络。   

CentOS 7上使用virt-manager安装虚拟机
https://www.linuxidc.com/Linux/2019-05/158766.htm


首先创建网桥并绑定

brctl addbr br0                #增加网桥
brctl addif br0 eno1           #绑定网桥跟物理网卡
ip addr del dev enO1 192.168.3.60/24  #删除物理网卡ip地址
ifconfig br0 192.168.3.60/24 up       #增加网桥ip地址并且启动
route add default gw 192.168.3.1      #重新增加默认网关   


nmcli connection show


How to setup a bridge with network-manager?
https://askubuntu.com/questions/883469/how-to-setup-a-bridge-with-network-manager


nmcli命令详解
https://blog.csdn.net/u010599211/article/details/86672940


网桥
How to add network bridge with nmcli (NetworkManager) on Linux
https://www.cyberciti.biz/faq/how-to-add-network-bridge-with-nmcli-networkmanager-on-linux/



network:
    ethernets:
        enp3s0f0:
            dhcp4: false
            dhcp6: false
    bridges:
        br0:
            addresses:
                - 192.168.1.8/24
            gateway4: 192.168.1.1
            nameservers:
                addresses: [ 192.168.1.1, 114.114.114.114 ]
                search: [ msnode ]
            interfaces:
                - enp3s0f0
    version: 2
    
nmcli connection show --active
nmcli con add ifname br0 type bridge con-name br0
nmcli con add type bridge-slave ifname enp2s0 master br0
nmcli con show
nmcli con modify br0 bridge.stp no
nmcli -f bridge con show br0
nmcli con down "helloworld_2G"


# nmcli device wifi rescan
# nmcli device wifi list
# nmcli device wifi connect SSID-Name password wireless-password


virsh list
nmcli -f bridge con show br0    

KVM 开启自启动
virsh autostart 虚拟机名    #设置随宿主机开机自启动
检查在/etc/libvirt/qemu/autostart/下会生成一个（虚拟机名.xml）文件
virsh autostart --disable 虚拟机名  #取消随宿主机开机自启动


cd /var/lib/libvirt/images/
du -sh *

KVM详解，太详细太深入了，经典
http://blog.chinaunix.net/uid-30022178-id-5749329.html


How to change port number in vino?
https://askubuntu.com/questions/69965/how-to-change-port-number-in-vino

Navigate to desktop>gnome>remote access and change your port there, don't forget to tick the use-alternative-port option so that vino uses the one you set:

How to find ip address of Linux KVM guest virtual machine


$ virsh list
$ virsh domifaddr freebsd11.1

# virsh net-list
# virsh net-info default
# virsh net-dhcp-leases default


如何在Ubuntu 18.04上使用UFW设置防火墙
https://www.linuxidc.com/Linux/2018-06/152881.htm
配置iptables、ufw端口转发
https://www.sundayle.com/iptables-forward/
kvm虚拟机端口映射
https://blog.51cto.com/ting2junshui/2045143
iPtables规则保存及加载等操作
https://blog.csdn.net/llxx1234/article/details/78758634


iptables 设置端口转发/映射
https://blog.csdn.net/light_jiang2016/article/details/79029661


vi /etc/sysctl.conf
    net.ipv4.ipv4_forward=1
    
sysctl -p


iptables -t nat -L -n --line-number
iptables -I FORWARD -m state -d 192.168.122.0/24 --state NEW,RELATED,ESTABLISHED -j ACCEPT
iptables -t nat -I PREROUTING -p tcp --dport 8010 -j DNAT --to-destination 192.168.122.23:80
iptables -t nat -I PREROUTING -p tcp -s 192.168.2.101/24 --dport 80:40000 -j DNAT --to-destination 192.168.122.23:80-40000

# 删除某个规则
iptables -t nat -L -n --line-number
iptables -t nat -D PREROUTING 1


iptables --list
iptables-save

iptables-save > /etc/iptables.rules
vi /etc/network/if-pre-up.d/iptable
    #!/bin/bash
    iptables-restore < /etc/iptables.rules
    iptables -I FORWARD -m state -d 192.168.122.0/24 --state NEW,RELATED,ESTABLISHED -j ACCEPT
    echo 1 > /proc/sys/net/ipv4/ip_forward
    
chmod +x !$


Cat /Proc/Sys/Net/Ipv4/Ip_forward 0 解决办法
https://www.cnblogs.com/wangbaobao/p/6674464.html

echo "echo 1 > /proc/sys/net/ipv4/ip_forward" >> /etc/rc.d/rc.local \
&& echo 1 > /proc/sys/net/ipv4/ip_forward \
&& chmod +x /etc/rc.d/rc.local \
&& ll /etc/rc.d/rc.local \
&& cat /proc/sys/net/ipv4/ip_forward


DNAT port range with different internal port range with Iptables
https://serverfault.com/questions/729810/dnat-port-range-with-different-internal-port-range-with-iptables

iptables -t nat -A PREROUTING -i xenbr0 -p tcp --dport 64000:65000 -j DNAT --to 172.16.10.10:61000-62000

微博分享按钮

看到一篇文章，想分享到微博，得打开微博，点新建微博，复制标题，复制网址，复制文字，上传图片等，很麻烦，可以新建一个书签，网址输入如下内容，看到好的文章，点下书签就可以分享到微博了。
javascript:window.open('http://service.weibo.com/share/share.php?url='+encodeURIComponent(location.href)+'&title='+encodeURIComponent(document.title)+'&appkey=3355748460&searchPic=true');


Vino& VNC server auto start after Ubuntu boot up
https://blog.csdn.net/flxuelee/article/details/51859878

在KVM主机和虚拟机之间共享目录
https://www.cnblogs.com/wangjq19920210/p/11303309.html

src_mnt /src 9p trans=virtio 0 0

libvirt/9p/kvm mount in fstab fails to mount at boot time
https://superuser.com/questions/502205/libvirt-9p-kvm-mount-in-fstab-fails-to-mount-at-boot-time

Don't know if it's the ideal solution, but on an Ubuntu 12.04 guest I got it to work by adding the 9p modules to the initramfs.

Added to /etc/initramfs-tools/modules:

9p
9pnet
9pnet_virtio
Then:

sudo update-initramfs -u


Sharing folder with VM through libvirt, 9p, permission denied
https://askubuntu.com/questions/548208/sharing-folder-with-vm-through-libvirt-9p-permission-denied

/etc/libvirt/qemu.conf:
    user = "root"
    group = "root"
    dynamic_ownership = 0

virsh edit ubuntu18.04
 
    <filesystem type='mount' accessmode='mapped'>
      <source dir='/kvm-data/data'/>
      <target dir='data'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x09' function='0x0'/>
    </filesystem>
    
accessmode 可以设置成：mapped，passthrough 或 none。物理机准备一下共享目录：
https://zhuanlan.zhihu.com/p/49120559

accessmode 可以设置成：mapped，passthrough 或 none。物理机准备一下共享目录：
所有虚拟机在物理机上都会以 libvirt-qemu 这个用户来跑（前面设置过 qemu.conf），所以需要保证你物理机上需要共享的路径的权限。同时 accessmode 建议设置成 mapped，这样会用文件信息里的 meta info 来保存虚拟机中的文件权限信息。    



Ubuntu中更改MySQL数据库文件目录的方法

cd /data 
mkdir mysqldb 
cp -r /var/lib/mysql /data/mysqldb/

vim /etc/mysql/my.cnf
    datadir = /data/mysqldb/mysql
chown -R mysql:mysql /data/mysqldb/mysql/

vim /etc/apparmor.d/usr.sbin.mysqld

    # /var/lib/mysql r, 
    # /var/lib/mysql/** rwk,

    /data/mysqldb/mysql/ r, 
    /data/mysqldb/mysql/** rwk,
    
/etc/init.d/apparmor restart 
/etc/init.d/mysql restart （或者使用 restart mysql）


在Ubuntu server版下装 x window 的过程如下：
apt-get install xserver-xorg 
apt-get install x-window-system-core 
dpkg-reconfigure xserver-xorg 
apt-get install gnome-core 
apt-get install gdm xscreensaver 
startx



Ubuntu 查看网关地址方法
1. ip route show

2.route -n or netstat -rn

3.traceroute


KVM 虚拟机在物理主机之间迁移的实现
https://www.ibm.com/developerworks/cn/linux/l-cn-mgrtvm2/index.html

virt-manager 管理的虚拟机配置文件在 /etc/libvirt/qemu/”your vm name.xml”。拷贝 XML 配置文件到目的主机的相同目录后，进行适当的修改，比如：与源主机相关的文件或路径等。无论你何时在 /etc/libvirt/qemu/ 中修改了虚拟机的 XML 文件，必须重新运行 define 命令，以激活新的虚拟机配置文件。

virsh define /etc/libvirt/qemu/”your vm name.xml”


How to autostart TeamViewer in Linux

Steps

Log in as either root or a user with sudo access.
Navigate to http://www.teamviewer.com and download the appropriate version of TeamViewer for your Linux Distribution.
Install the TeamViewer software.
Debian, Ubuntu, SUSE: sudo dpkg -i teamviewer_linux.deb or teamviewer_linux_x64.deb
RedHat, CentOS, Fedora: rpm -i teamviewer_linux.rpm
Once installed run TeamViewer and choose from the Menu option Extras > Options
Checkmark the Box "Start TeamViewer with system".
Please take note of the TeamViewer ID or assign this TeamViewer installation to an account.
Set a Person password (for unattended access).
Then open a terminal in order to modify the file needed to initiate TeamViewer on startup.
Assuming you use vi as your text editor:
sudo vi /etc/rc.local
By default this document does nothing, we will add a line of code above the last line that says:
exit 0
In this instance I am using TeamViewer 9 so my line will look like this:
/opt/teamviewer9/tv_bin/script/teamviewer &
Save and close the file.
Reboot the computer.


压缩 KVM 镜像
tar --use-compress-program='pigz -1' -Scvf a.qcow2.tar.gz a.qcow2 + rsync --sparse

tar --use-compress-program='pigz -1' -Scvf ubuntu18.04.qcow2.tar.gz ubuntu18.04.qcow2

rsync --ignore-existing --sparse ubuntu18.04.qcow2.tar.gz
rsync -v -z -r --inplace --progress -e ssh /path/to/local/storage/ user@remote.machine:/path/to/remote/storage/ 



Ubuntu Server 进入界面模式

apt-get install xserver-xorg 
apt-get install x-window-system-core 
dpkg-reconfigure xserver-xorg 
apt-get install gnome-core 
apt-get install gdm xscreensaver 
startx

开机不启动界面
sudo nano /etc/default/grub

    #GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
    GRUB_CMDLINE_LINUX_DEFAULT="text"
    
sudo update-grub
sudo systemctl enable multi-user.target --force
sudo systemctl set-default multi-user.target


Ubuntu 18.04 修改默认启动级别

1、切换为图形模式

sudo systemctl set-default graphical.target
或者
sudo systemctl set-default runlevel5.target
2、切换为文本命令模式

sudo systemctl set-default multi-user.target
或者
sudo systemctl set-default runlevel3.target
3、需要重启


How to install VNC on Ubuntu 18.04 (TightVNC Server)
https://www.ubuntu18.com/install-vnc-server-ubuntu-18/

sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install xfce4 xfce4-session
sudo apt-get install tightvncserver
autocutsel -s PRIMARY/CLIPBOARD -cutbuffer 0 -f
apt install autocutsel
tightvncserver
tightvncserver -kill :1


vi ~/.vnc/xstartup
    #!/bin/sh

    unset SESSION_MANAGER
    xrdb $HOME/.Xresources
    xsetroot -solid grey
    [ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
    [ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
    autocutsel &
    startxfce4 &
    
tightvncserver :1 -geometry 1366x768 -depth 24

sudo vim /etc/systemd/system/vncserver@1.service

[Unit]
Description=Start a VNC Session at startup With Desktop ID 1
After=multi-user.target network.target

[Service]
Type=forking
# IMPORTANT!!! Change 'username' to actual user that connect to the session
User=root
ExecStartPre=/bin/sh -c '/usr/bin/tightvncserver -kill :%i > /dev/null 2>&1 || :'
ExecStart=/usr/bin/tightvncserver :%i -geometry 1366x768 -depth 24
ExecStop=/usr/bin/tightvncserver -kill :%i

[Install]
WantedBy=multi-user.target

sudo systemctl daemon-reload
sudo systemctl enable vncserver@1.service


中文语言包
sudo apt-get install language-pack-zh
sudo fc-list ：lang=zh//显示ubuntu系统中的中文字体


sudo fc-cache //刷新字体缓存

安装中文字体
sudo apt-get install ttf-wqy-zenhei
sudo fc-cache -v


修改时区

tzselect
    Asia，确认之后选择中国（China)，最后选择北京(Beijing)
root@ubuntu:/# cp /usr/share/zoneinfo/Asia/Shanghai  /etc/localtime


ZOOMIT的使用方法
https://blog.csdn.net/molihuakai_118/article/details/75996398


按下快捷键(默认ctrl+1)，即可进入ZoomIt的放大模式。这时屏幕内容将放大后(默认2倍)显示。
移动光标，放大区域将随之改变。
用鼠标滚轮或者上下方向键，将改变放大比例。
关于实时放大，见下。
按下Esc键 或 鼠标右键，会退出放大模式。
在放大模式下，按上鼠标左键，将保持放大状态，启用标注功能。当然，也可以退出放大，只进行标注。

按下快捷键(默认ctrl+2)，或在放大模式下按下鼠标左键，可进入标注模式。这时，鼠标会变成一个圆形的笔点，其颜色、大小可调。
* 通过按住左ctrl键，使用鼠标滚轮或者上下箭头键调整画笔的宽度。
* 按键调整画笔颜色:r 红色;g 绿色;b 蓝色;o 橙色;y 黄色;p 粉色。

可轻松画出不同的形状:
* 按住Shift键可以画出直线;
* 按住Ctrl键可以画出长方形;
* 按住tab键可以画出椭圆形;
* shift+ctrl 可以画出箭头。

* Ctrl+Z:撤销最后的标注。
* e:擦除所有标注。
* w(白色)/k(黑色):将屏幕变成白板或黑板。
* ctrl+s:保存标注或者放大后的画面。
*屏幕打字:进入标注模式后，按't'可以进入打字模式。Esc或左键退出。鼠标滚轮或上下箭头可以改变字体大小。缺点是，不支持中文。
* 鼠标右键:退出标注模式。



生产环境中使用Docker Swarm的一些建议
https://www.cnblogs.com/fundebug/p/6823897.html



How To Install Nginx on CentOS 7
https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-centos-7

sudo yum install epel-release
sudo yum install nginx
sudo systemctl start nginx
sudo firewall-cmd --permanent --zone=public --add-service=http 
sudo firewall-cmd --permanent --zone=public --add-service=https
sudo firewall-cmd --reload
sudo systemctl enable nginx


How To Install MySQL on CentOS 7
https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-centos-7


如何在Ubuntu 18.04 LTS上安装和配置KVM
https://www.sysgeek.cn/install-configure-kvm-ubuntu-18-04/
如何在 Ubuntu 上搭建网桥
https://www.linuxprobe.com/build-bridge-ubuntu.html
查看网卡
sudo networkctl status -a


How to find ip address of Linux KVM guest virtual machine
https://www.cyberciti.biz/faq/find-ip-address-of-linux-kvm-guest-virtual-machine/

# virsh net-list
# virsh net-info default
# virsh net-dhcp-leases default
$ virsh list
$ virsh domifaddr freebsd11.1

virsh list --name | while read n 
do 
  [[ ! -z $n ]] && virsh domifaddr $n
done
$ virsh list
$ virsh dumpxml VM_NAME | grep "mac address" | awk -F\' '{ print $2}'
$ arp -an | grep 52:54:00:ce:8a:c4

$ virsh list
$ virsh domifaddr rhel7

How To Setup Bridge (br0) Network on Ubuntu Linux 14.04 and 16.04 LTS
https://www.cyberciti.biz/faq/how-to-create-bridge-interface-ubuntu-linux/


centos7 mysql数据库安装和配置
https://www.cnblogs.com/starof/p/4680083.html
CentOS 7 下 yum 安装和配置 NFS
https://qizhanming.com/blog/2018/08/08/how-to-install-nfs-on-centos-7
Centos 7 DHCP 安装和配置
https://blog.51cto.com/13701082/2339242

CentOS7命令行下快速配置网络
https://www.jianshu.com/p/eab30fb7de15
nmcli d
nmtui


修改CentOS默认yum源为国内yum镜像源
https://blog.csdn.net/inslow/article/details/54177191

CentOS Docker 安装
https://www.runoob.com/docker/centos-docker-install.html

CentOS 7 安装 PHP 7.2
https://blog.csdn.net/huaying927/article/details/98033119


压缩 KVM 的 qcow2 镜像文件
https://www.cnblogs.com/sixloop/p/8515099.html

qemu-img convert -c -O qcow2 /path/old.img.qcow2 /path/new.img.qcow2


Support for `--no-resolve-image`? #1844
https://github.com/docker/docker-py/issues/1844

image = 'local.registry.url:5000/foo/bar'
image_id = client.inspect_image(image)['Id']

# 查看端口是否监听，如果未监听，多等一会儿，约几十秒到 1分钟
while true; do netstat -tnpl | grep 10000; sleep 1; done

## 工程化 Check List

文档

- 是否有需求分析文档？
- 是否有概要设计文档？
- 是否有详细设计文档？
- 是否有运维部署文档？
- 是否有代码规范文档？
- 是否有数据库规范文档？

基础组件

- 是否有日志记录基础组件？
- 是否有数据访问基础组件？
- 是否有缓存使用基础组件？
- 是否有加密解密基础组件？
- 是否有频率限制基础组件？
- 是否有运维计数器基础组件？
- 是否有用户行为计数器基础组件？

基础服务

- 是否有通知发送服务？

过程管理

- 是否使用版本管理工具？
- 是否使用需求管理工具？
- 是否使用 BUG 跟踪工具？
- SQL 语句是否有 DBA Review？
- 代码上线前是否有安全 Review？

开发

- 是否有接口文档生成工具？
- 是否有代码风格自动检查工具？
- 是否有自动化单元测试工具？

运维

- 是否有自动部署工具？
- 自动部署工具是否有回滚功能?

运营

- 是否每天自动发送运营报表？
- 运营报表是否包含用户增长数据，用户活跃数据，用户应收数据？
- 是否有内部工具可以让运营搜索各种运营数据？



Linux GCC常用命令
https://www.cnblogs.com/ggjucheng/archive/2011/12/14/2287738.html#_Toc311642844

上述编译过程是分为四个阶段进行的，即预处理(也称预编译，Preprocessing)、编译(Compilation)、汇编 (Assembly)和连接(Linking)。

gcc -E test.c -o test.i 或 gcc -E test.c
gcc -S test.i -o test.s
gcc -c test.s -o test.o
gcc test.o -o test

GDB详解
https://www.cnblogs.com/ggjucheng/archive/2011/12/14/2288004.html


grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}'
free | grep Mem | awk '{print $3/$2 * 100.0 "%"}'
df -hl / | grep / | awk '{print $5}'

 df -hl | grep 'sda2\|sda3' | awk 'BEGIN{print "Size","Use%"} {size+=$2;percent+=$5;} END{print size,percent}' | column -t
 
 
escape dollar sign in bashscript (which uses awk)
https://stackoverflow.com/questions/18344459/escape-dollar-sign-in-bashscript-which-uses-awk

How To Use awk In Bash Scripting
https://www.cyberciti.biz/faq/bash-scripting-using-awk/

Arrays in unix shell?
https://stackoverflow.com/questions/1878882/arrays-in-unix-shell

Shell script “for” loop syntax
https://stackoverflow.com/questions/1445452/shell-script-for-loop-syntax

Remotely Execute Multi-line Commands with SSH
https://thornelabs.net/posts/remotely-execute-multi-line-commands-with-ssh.html

How to capture disk usage percentage of a partition as an integer?
https://askubuntu.com/questions/847752/how-to-capture-disk-usage-percentage-of-a-partition-as-an-integer

How to print the percentage of disk use from `df -hl`
https://unix.stackexchange.com/questions/64815/how-to-print-the-percentage-of-disk-use-from-df-hl

Linux command for percentage of memory that is free [closed]
https://stackoverflow.com/questions/10585978/linux-command-for-percentage-of-memory-that-is-free

How to get overall CPU usage (e.g. 57%) on Linux [closed]
https://stackoverflow.com/questions/9229333/how-to-get-overall-cpu-usage-e-g-57-on-linux

How to use bash $(awk) in single ssh-command?
https://stackoverflow.com/questions/14707307/how-to-use-bash-awk-in-single-ssh-command 


ssh user@localhost bash -c "' echo 222 \$( uname -a | awk "'"'{print \\\$2}'"'" ) '"
ssh user@host 'bash -s' < local_script.sh

ssh user@host <<'ENDSSH'
#commands to run on remote host
ENDSSH

runoverssh -s localscript.sh user host1 host2 host3...


写一个脚本，同时监控一个集群多台机器的CPU，内存，磁盘使用率
用于不方便使用浏览器时快速查看少量机器整体情况的场景

# cat monitor.sh
hosts=(
    "user@host1"
    "user@host2"
    "user@host3"
)

while true
do
    printf "%-15s %-15s %-10s %-10s %-20s\n" host cpu mem disk load
    for host in "${hosts[@]}"
    do
        ssh $host '
            host=$(hostname)
            cpu=$(grep "cpu " /proc/stat | awk "{usage=(\$2+\$4)*100/(\$2+\$4+\$5)} END {print usage \"%\"}")
            mem=$(free | grep Mem | awk "{print \$3/\$2 * 100.0 \"%\"}")
            disk=$(df -hl / | grep / | awk "{print \$5}")
            load=$(cat /proc/loadavg | awk "{print \$1\",\"\$2\",\"\$3}")
            printf "%-15s %-15s %-10s %-10s %-20s\n" $host $cpu $mem $disk $load
        '
    done
    sleep 1
done


# ./monitor.sh
user@host1       0.0348016%      1.74837%   3%         0.00,0.00,0.00
user@host2       0.0370607%      1.31528%   3%         0.00,0.00,0.00
user@host3       0.053345%       1.32445%   3%         0.04,0.03,0.00



定时清理
find /data/backup/ -mtime +30 -name '*.sql' -exec rm -i {} \;


Linux安装CUDA的正确姿势
https://blog.csdn.net/wf19930209/article/details/81879514

ubuntu安装tensorflow gpu版本
https://zhuanlan.zhihu.com/p/44183691

# lspci | grep -i nvidia
06:00.0 VGA compatible controller: NVIDIA Corporation GK107 [GeForce GTX 650] (rev a1)
06:00.1 Audio device: NVIDIA Corporation GK107 HDMI Audio Controller (rev a1)


按alt + ctrl + F1进入tty命令行界面,登录用户和密码
sudo service lightdm stop
sudo telinit 3


#如有输出信息则需要禁用nouveau
lsmod | grep nouveau

#禁用
cd /etc/modprobe.d
sudo vi blacklist-nouveau.conf

#在文件末尾添加如下信息
blacklist nouveau
options nouveau modeset=0

#更新 重启
sudo update-initramfs -u
sudo reboot

#重启后 再次检查 
#若无信息输出 则禁用成功
lsmod | grep nouveau

CUDA与cuDNN
https://www.jianshu.com/p/622f47f94784



Zipkin和微服务链路跟踪
https://www.cnblogs.com/danghuijian/p/9482279.html



API 网关与高性能服务最佳实践
https://www.upyun.com/opentalk/activity/51.html

GPU计算加速01 : AI时代人人都应该了解的GPU知识
https://zhuanlan.zhihu.com/p/76297133
GPU加速02:超详细Python Cuda零基础入门教程，没有显卡也能学！
https://zhuanlan.zhihu.com/p/77307505
GPU加速03:多流和共享内存—让你的CUDA程序如虎添翼的优化技术！
https://zhuanlan.zhihu.com/p/78557104
GPU加速04:将CUDA应用于金融领域，使用Python Numba加速B-S期权估值模型
https://zhuanlan.zhihu.com/p/78844923


Make the current commit the only (initial) commit in a Git repository?
https://stackoverflow.com/questions/9683279/make-the-current-commit-the-only-initial-commit-in-a-git-repository

cat .git/config  # note <github-uri>
rm -rf .git

git init
git add .
git commit -m "Initial commit"

git remote add origin <github-uri>
git push -u --force origin master


清空某文件的 git 历史
Removing and purging files from git history
https://blog.ostermiller.org/git-remove-from-history

# 某台机器
cp FILE_LIST /backup
git filter-branch --tag-name-filter cat --index-filter 'git rm -r --cached --ignore-unmatch FILE_LIST' --prune-empty -f -- --all
git push origin --force --all
git push origin --force --tags
git reflog expire --expire=now --all
git gc --aggressive --prune=now
# 其它机器
cd MY_LOCAL_GIT_REPO
git fetch origin
git rebase
git reflog expire --expire=now --all
git gc --aggressive --prune=now

查看 git 里最大的文件
git verify-pack -v .git/objects/pack/pack-*.idx | sort -k 3 -g | tail -5
git rev-list --objects --all | grep 2ab97e4d630bfd2369770d3650548356eefd583a

qcon 2019 ppt 下载
https://qcon.infoq.cn/2019/beijing/schedule?utm_source=infoq&utm_campaign=full&utm_term=0517


查看 cpu 命令 lscpu

音频云转换
https://cloudconvert.com/mp3-to-wav


Spark : how to run spark file from spark shell
https://stackoverflow.com/questions/27717379/spark-how-to-run-spark-file-from-spark-shell
:load PATH_TO_FILE
spark-shell -i file.scala




scala> csv4.take(2)
res128: Array[Array[String]] = Array(Array(xxx大酒店, aba_2069, 低星, 115, 中国, \N, 阿坝, 34.0567484663, 0.0790406185024, \N, 0, 0, 242, 129, 四川, 27, 59, 35, 72, 二星及其他, 27, 59, 6, 0.176470588235, 34, 71, 3.9779300690), Array(yyy宾馆, aba_2096, 低星, 49, 中国, \N, 阿坝, 34.0567484663, 0.0790406185024, 九寨沟沟口, 1, 0, 227, 394, 四川, 42, 74, 60, 117, 二星及其他, 38, 66, 0, 0, 47, 89, 3.9723401070))

scala> val cities = Array("广州", "北京","上海")
cities: Array[String] = Array(广州, 北京, 上海)

scala> val city_index = 6
city_index: Int = 6

scala> val comments_index = 13
comments_index: Int = 13

scala> csv4.groupBy(_(city_index)).filter(x=>cities.contains(x._1)).map(x=>(x._1, x._2.map(_(comments_index).toInt).reduce((x,y)=>x+y))).collect
res129: Array[(String, Int)] = Array((广州,627913), (北京,894305), (上海,695220))

scala>


How to flatten a List of Lists in Scala with flatten
https://alvinalexander.com/scala/how-to-flatten-list-lists-in-scala-with-flatten-method

scala> val lol = List(List(1,2), List(3,4))
lol: List[List[Int]] = List(List(1, 2), List(3, 4))
scala> val result = lol.flatten
result: List[Int] = List(1, 2, 3, 4)



API 网关的选型 测试和持续集成
https://www.upyun.com/opentalk/433.html


玩转Flume+Kafka原来也就那点事儿
https://www.jianshu.com/p/f0a08bd4f975

玩转Flume+Kafka原来也就那点事儿
https://blog.csdn.net/csdcit/article/details/79392311

这些「神秘」团队到底是做什么的？
http://www.imooc.com/article/281307

我们认为优秀的工程师或多或少有以下共同特质：

A Quick Learner
An Earnest Curiosity
Faith in Open Source
Self-driven
Get Things Done


virtual 迁移
源机器
    pip freeze --all > requirements.txt
    pip download -r requirements.txt
目标机器
    # 解压 packages.zip 到 packages 目录
    virtualenv --extra-search-dir=packages --never-download venv
    venv\Scripts\activate.bat
    pip install --no-index -f packages -r packages\requirements.txt
    python app.py


PyPI上package有好几种格式：

源文件（一般是.tar.gz或.zip文件，用pip安装，与机器架构无关，但某些package可能涉及到C/C++编译）
wheel文件（二进制文件，拓展名为.whl，用pip安装，无需编译，但与机器架构相关）
.egg文件（二进制文件，用easy_install安装，无需编译，但与机器架构相关）

APISIX 的高性能实践
https://www.upyun.com/opentalk/429.html


How to remove an item from a list in Scala having only its index?
https://stackoverflow.com/questions/18847249/how-to-remove-an-item-from-a-list-in-scala-having-only-its-index
val trunced = internalIdList.take(index) ++ internalIdList.drop(index + 1)

Scala: what is the best way to append an element to an Array?
0 +: array :+ 4

Array 插入一列
t.map(x =>x._1 +: x._2.productIterator.toArray)


val rdd1 = sc.makeRDD(Array(("Spark", "1", "30K"),("Hadoop", "2", "15K"),("Scala", "3", "25K"),("Java", "4", "10K")))
val rdd1_2 = rdd1.map(_.productIterator.toArray).map(x=>x(1) +: (x.take(1) ++ x.drop(1+1)))



scala> val rdd1 = sc.makeRDD(Array(("Spark", "1"),("Hadoop", "2"),("Scala", "3"),("Java", "4")))
rdd1: org.apache.spark.rdd.RDD[(String, String)] = ParallelCollectionRDD[17] at makeRDD at <console>:24

scala> val rdd1_2 = rdd1.map(x=>(x._2, x._1))
rdd1_2: org.apache.spark.rdd.RDD[(String, String)] = MapPartitionsRDD[18] at map at <console>:27

scala> rdd1_2.collect
res12: Array[(String, String)] = Array((1,Spark), (2,Hadoop), (3,Scala), (4,Java))

scala> val rdd2 = sc.makeRDD(Array(("1","30K"),("2","15K"),("3","25K"),("5","10K")))
rdd2: org.apache.spark.rdd.RDD[(String, String)] = ParallelCollectionRDD[19] at makeRDD at <console>:24

scala> val rdd3 = rdd1_2.join(rdd2)
rdd3: org.apache.spark.rdd.RDD[(String, (String, String))] = MapPartitionsRDD[22] at join at <console>:27

scala> rdd3.collect
res13: Array[(String, (String, String))] = Array((1,(Spark,30K)), (2,(Hadoop,15K)), (3,(Scala,25K)))

scala> val rdd4 = rdd3.map(x=>(x._1, x._2._1, x._2._2))
rdd4: org.apache.spark.rdd.RDD[(String, String, String)] = MapPartitionsRDD[23] at map at <console>:27

scala> rdd4.collect
res14: Array[(String, String, String)] = Array((1,Spark,30K), (2,Hadoop,15K), (3,Scala,25K))



scala 里一个 tuple 如何变换列的位置？toArray 后用 +：, take，drop 实现效果了，但无法再转回 tuple了。
如果用 _2, _1, _3 重新排列的话不够通用，而且有时候 tuple 有几十列，写出来代码很难看，还有没有更简单通用的办法，
最好不用sharpless。


val filter_func = (x:Array[String]) => csv2.filter(x(7)==_(14)).isEmpty


val df1 = spark.read.format("csv").option("delimiter", ",").load("hdfs://localhost:9000/comments1.csv")
val df2 = spark.read.format("csv").option("delimiter", ",").load("hdfs://localhost:9000/propertys1.csv"r)
val df3 = df1.join(df2, df1("_c7") <=> df2("_c14"))
val colnames = (0 to 60).map("c_"+_.toString).toArray
val df4 = df3.toDF(colnames: _*)
df4.repartition(1).write.format("csv").save("hdfs://localhost:9000/joined.csv")
 val csv = sc.textFile("hdfs://localhost:9000/joined.csv").map(_.split(","))
 
Ubuntu18.04安装NVIDIA显卡驱动
https://phantomvk.github.io/2019/06/29/Ubuntu_install_nVidia_Driver/

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
ubuntu-drivers devices
sudo apt install nvidia-driver-430 


"RmInitAdapter failed" with 370.23 but 367.35 works fine
dmesg | grep NVRM
pacman -Q|grep nvidia
journalctl -xe
lsmod|grep nvidia
modinfo nvidia
cat /proc/driver/nvidia/gpus/0000\:06\:00.0/information
lsinitcpio /boot/initramfs-linux.img|grep nvidia

How do I install NVIDIA and CUDA drivers into Ubuntu?
https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu

Using GPU from a docker container?
https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container


Docker 安装 TensorFlow GPU 实战
https://blog.csdn.net/u011291159/article/details/66970202
https://github.com/NVIDIA/nvidia-docker

systemctl status nvidia-docker

docker pull daocloud.io/daocloud/tensorflow:1.4.0-rc1-gpu
nvidia-docker run -i -t -p 8888:8888 daocloud.io/daocloud/tensorflow:1.4.0-rc1-gpu /bin/bash
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu

docker 
https://download.docker.com/linux/ubuntu/dists/bionic/pool/stable/amd64/
安装
    dpkg -i containerd.io_1.2.6-3_amd64.deb
    dpkg -i docker-ce-cli_19.03.4~3-0~ubuntu-bionic_amd64.deb
    dpkg -i docker-ce_19.03.4~3-0~ubuntu-bionic_amd64.deb

卸载
    sudo apt-get purge docker-ce
    sudo rm -rf /var/lib/docker

Accessing GPUs from a Docker Swarm service
http://cowlet.org/2018/05/21/accessing-gpus-from-a-docker-swarm-service.html
docker19.03使用NVIDIA显卡
https://blog.51cto.com/13447608/2437856?source=dra

Runtime options with Memory, CPUs, and GPUs
https://docs.docker.com/config/containers/resource_constraints/



压缩：时间戳差值压缩，zigzag xor,snappy
索引：LSM

成本函数

## scikit-learn

- 基于Numpy、SciPy和Matplotlib
- 适用于分类（向量机、最近邻、随机森林等）
- 回归（向量回归（SVR），脊回归，Lasso回归等）
- 聚类（K-均值、谱聚类、均值偏移等）
- 数据降维、模型选择和数据预处理等应用。

深度神经网络，顾名思义，就是包含很多隐藏层的神经网络。这就意味着深度神经网络包含很多权重和偏置，这些权重和偏置，落实到代码层面，就是很多大型浮点矩阵。所以，深度神经网络需要进行很多浮点运算，也需要较大的带宽来访问这些大型浮点矩阵。只有gpu才符合以上两点要求。

目前sklearn不使用gpu，确实是早期设计原因，它并不是为海量数据的学习建模，以及企业级应用设计的。我更倾向于把它归类于教学工具，以及用于小批量数据的初步分析。因为它功能全面，使用简单。能够快速给研究人员灵感。真正的企业级部署是不可能用sklearn的。几乎所有的算法都有更好的替代。

sklearn可以用到gpu吗，是只有神经网络能用gpu的算力？
https://www.wukong.com/answer/6502778705033560334/

Ubuntu18.04 + CUDA9.0 + cuDNN7.3 + Tensorflow-gpu-1.12 + Jupyter Notebook深度学习环境配置
https://blog.csdn.net/youngping/article/details/84207234


python 安装

sudo apt-get install python3.7
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2
python --version

sudo apt install python3-pip
sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate




import tensorflow as tf
print(tf.__version__)
tf.constant(1.)+tf.constant(2.)
tf.test.is_gpu_available()

初学者的 TensorFlow 2.0 教程
https://tensorflow.google.cn/tutorials/quickstart/beginner

高效的TensorFlow 2.0 (tensorflow2官方教程翻译)
https://www.jianshu.com/p/599c79c3a537

MNIST机器学习入门
http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html

TensorFlow教程：TensorFlow快速入门教程（非常详细）
http://c.biancheng.net/tensorflow/

新手入门第四课——PaddlePaddle快速入门
https://aistudio.baidu.com/aistudio/projectdetail/176740

深度学习开源书，基于TensorFlow 2.0实战。
https://github.com/dragen1860/Deep-Learning-with-TensorFlow-book

TensorFlow 2.0完全入门
https://www.bilibili.com/video/av69107165

TensorFlow入门：手写体数字识别
https://v.qq.com/x/page/a0884nk1tvp.html

【帅器学习/柏欣】Tensorflow2入门
https://www.bilibili.com/video/av69661987/?redirectFrom=h5


复制算法

黄文才：云智天枢AI中台架构及AI在K8S中的实践
https://cloud.tencent.com/developer/article/1505465


监控系统通过开源组件：telegraf + influxdb + grafana 搭建起来的。每个node节点会部署一个daemonset的容器monitor_agent。业务容器通过unix socket与monitor_agent通讯。monitor_agent每分钟汇总上报数据到influxdb存储，grafana通过读取influxdb进行视图展示和告警设置等。

当然业界也有其他的解决方案，比如说Prometheus(普罗米修斯)，我们做了选型的对比：

1. 它的采集器是以exporter的方式去运行的，每一个要监控的组件都有要装一个exporter，promethus再从多个exporter里面拉数据。

2.promethus自带的展示界面比较挫，即使采用promethus，也要配合grafana才能用 

3.promethus是拉模式来搜集数据的，业务要上报数据的话要安装它的一个代理pushgateway，数据先push到代理，然后再等promethus去拉，实时性和性能都比推模式略差一点 

我们采用的类似侵入式的方式，更适合我们这个平台。另外telegraf有很多扩展插件，用起来会很方便。


2007年6月，NVIDIA公司推出了CUDA (Compute Unified Device Architecture)，CUDA 不需要借助图形学API，而是采用了类C语言进行开发。

从左边图来看，CUDA软件体系分为三层：（1）CUDA函数库；（2）CUDA 运行时API；（3）CUDA驱动API。

所以应用程序和CUDA Libraries以及CUDA Runtime间不会有什么问题？

主要问题在CUDA Runtime和CUDA Driver之间。CUDA Driver库是在创建容器时从宿主机挂载到容器中，很容易出现版本问题，需要保证CUDA Driver的版本不低于CUDA Runtime版本。
容器内使用GPU:

1、GPU设备挂载到容器里

--device挂载设备到容器

特权模式privileged

2、CUDA Driver API挂载到容器

3、CUDA Runtime API和CUDA Libraries

通常跟应用程序一起打包到镜像里

4、K8S通过nvidia-container-runtime实现挂载gpu设备和cuda驱动进容器

5、K8S通过nvidia-device-plugin实现分卡。扩展资源 ，使gpu能像cpu那样分配。

 nvidia有两种GPU虚拟化的解决方案：（1）GRID；（2）MPS。第一种模式更多用于虚拟机场景，基于驱动，隔离型会做的比较强，但不开源，性能比较好。MPS是应用到容器场景的，基于软件的方式，隔离性比较弱，但也不开源。
 
 
浅谈GPU虚拟化技术（四）- GPU分片虚拟化
https://yq.aliyun.com/articles/599189

Docker 中玩转 GPU
https://blog.opskumu.com/docker-gpu.html

# export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
# docker run -it --rm $DEVICES -v /usr/lib64/nvidia/:/usr/local/nvidia/lib64 tensorflow/tensorflow:latest-gpu bash


root@b37235b80e1a:/notebooks# python
Python 2.7.12 (default, Nov 19 2016, 06:48:10)
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
>>> b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
>>> c = tf.matmul(a, b)
>>> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
2017-07-14 15:30:24.261480: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla P40, pci bus id: 0000:03:00.0)
2017-07-14 15:30:24.261516: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:1) -> (device: 1, name: Tesla P40, pci bus id: 0000:82:00.0)
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla P40, pci bus id: 0000:03:00.0
/job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: Tesla P40, pci bus id: 0000:82:00.0
2017-07-14 15:30:24.263788: I tensorflow/core/common_runtime/direct_session.cc:265] Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla P40, pci bus id: 0000:03:00.0
/job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: Tesla P40, pci bus id: 0000:82:00.0

>>> print(sess.run(c))
... ...
[[ 22.  28.]
[ 49.  64.]]

docker run --rm --name tensorflow -ti tensorflow/tensorflow:latest-gpu
python -c 'import tensorflow as tf;print(tf.test.is_gpu_available())'

root@bd0fb3758da2:~# python --version
Python 2.7.6
root@bd0fb3758da2:~# python -m tensorflow.models.image.mnist.convolutional

程序员也可以懂一点期望值管理
https://zhuanlan.zhihu.com/p/20046507

如何高效的学习掌握新技术
https://zhuanlan.zhihu.com/p/20190356

《构建之法》读后感之项目计划
https://zhuanlan.zhihu.com/p/20044649

沈4 年前
项目管理，就是带领团队不断找到项目中最模糊的部分，在一段时间内把它清晰化到一定程度，然后再找到次模糊的部分，不断迭代。项目成员和其他涉众在这个逐步清晰化的过程中，自然会越来越有信心。
是否勇于面对模糊的部分，是考察领导力的最基本的要素。

./cuda_10.0.130_410.48_linux

cd /usr/local/cuda-10.0/samples/1_Utilities/deviceQuery
make
./deviceQuery
cd ../bandwidthTest
make
./bandwidthTest



解决Could not load dynamic library 'libcudart.so.10.0'的问题
https://blog.csdn.net/u012388993/article/details/102573117

AttributeError: module 'tensorflow' has no attribute 'Session'
https://blog.csdn.net/qq_33440324/article/details/94200046

用 tf.compat.v1.Session() 代替 tf.Session()`

tf.device()指定tensorflow运行的GPU或CPU设备
https://blog.csdn.net/dcrmg/article/details/79747882

TensorFlow 2.0 教程
https://blog.csdn.net/qq_31456593/article/details/88606284

『TensorFlow2.0正式版』极简安装TF2.0正式版（CPU&GPU）教程
https://blog.csdn.net/xiaosongshine/article/details/101844926

import tensorflow as tf
version = tf.__version__
gpu_ok = tf.test.is_gpu_available()
print("tf version:",version,"\nuse GPU",gpu_ok)

禁用 2.0 行为
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tensorflow中不同的GPU使用/gpu:0和/gpu:1区分，而CPU不区分设备号，统一使用 /cpu:0


TensorFlow2.0教程-张量及其操作
https://blog.csdn.net/qq_31456593/article/details/90178004
.
Disable Tensorflow debugging information
https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

让 cat 增加语法高亮

pygmentize -g 003.py

Syntax highlighting/colorizing cat
https://stackoverflow.com/questions/7851134/syntax-highlighting-colorizing-cat

https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster

# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker

docker run --help | grep -i gpus
docker run -it --rm --gpus all ubuntu nvidia-smi

docker run --gpus all --rm --name tensorflow -ti tensorflow/tensorflow:latest-gpu
python -c 'import tensorflow as tf;print(tf.test.is_gpu_available())'


给容器按比例分配CPU
cat /sys/fs/cgroup/cpu/docker/cpu.shares
1024
So, if we have 2 containers, one for the database and one more for the web server
sudo docker run -c 614 -dit --name db postgres /postgres.sh
sudo docker run -c 410 -dit --name web nginx /nginx.sh


GPU 编排
CASE STUDY: GPU ORCHESTRATION USING DOCKER
https://codingsans.com/blog/gpu-orchestration-using-docker
Accessing GPUs from a Docker Swarm service
http://cowlet.org/2018/05/21/accessing-gpus-from-a-docker-swarm-service.html


docker service create --generic-resource "gpu=1" --constraint 'node.id == sqe18dxvimg8arkbn3elvkaep' tensorflow/tensorflow:latest-gpu


ls /dev/nvidia0


## docker gpu

确认检测到显卡

    lspci -vv | grep -i nvidia
    
Nvidia 驱动安装

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update
    ubuntu-drivers devices
    sudo apt install nvidia-driver-430
    
驱动测试

    nvidia-smi

CUDA 下载

    # 下载 10.0，不要下载最新的 10.1，因为目前 tensorflow 2.0 不支持，
    https://developer.nvidia.com/cuda-downloads
    
CUDA 安装

    # 不要安装 Nvidia 驱动，因为已经安装过了
    # 如果是双显卡，不要安装 OpenGL 库，否则可能会黑屏
    chmod +x cuda_10.0.130_410.48_linux
    ./cuda_10.0.130_410.48_linux
    
CUDA 测试

    /usr/local/cuda-10.0/bin/nvcc -V
    
    # 测试结果应为 PASS
    cd /usr/local/cuda-10.0/samples/1_Utilities/deviceQuery
    make
    ./deviceQuery
    
    # 测试结果应为 PASS
    cd ../bandwidthTest
    make
    ./bandwidthTest
    
libcudnn

    https://developer.nvidia.com/rdp/cudnn-archive

libcudnn 安装

    dpkg -i libcudnn7_7.6.3.30-1+cuda10.1_amd64.deb
    
tensorflow 2.0 安装

    # 安装 pip 和 virtualenv
    sudo apt install python3-pip
    mkdir ~/.pip
    sudo vim ~/.pip/pip.conf
        [global]
        index-url = https://mirrors.aliyun.com/pypi/simple
        [install]
        trusted-host=mirrors.aliyun.com
    sudo pip3 install -U virtualenv
    # 创建并加载虚拟环境
    virtualenv --system-site-packages -p python3 ./venv
    source ./venv/bin/activate
    # 确认 pip 为虚拟环境中的 pip
    which pip
    # 安装 2.0 版本
    pip install tensorflow-gpu==2.0.0
    
tensorflow with GPU 测试

    python -c 'import tensorflow as tf;print(tf.test.is_gpu_available())'
    
jupyter notebook 安装启动
    
    source ./venv/bin/activate
    pip install jupyter
    pip install ipykernel
    python -m ipykernel install --user --name=tensorflow
    jupyter notebook
    
jupyter notebook with GPU 测试

    # 打开 localhost:8888 新建 tenforflow 文件并运行如下代码
    import tensorflow as tf
    version = tf.__version__
    gpu_ok = tf.test.is_gpu_available()
    print("tf version:",version,"\nuse GPU",gpu_ok)
    

Docker 19.03 下载

    https://download.docker.com/linux/ubuntu/dists/bionic/pool/stable/amd64/

Docker 安装

    dpkg -i containerd.io_1.2.6-3_amd64.deb
    dpkg -i docker-ce-cli_19.03.4~3-0~ubuntu-bionic_amd64.deb
    dpkg -i docker-ce_19.03.4~3-0~ubuntu-bionic_amd64.deb
    
nvidia-container-runtime-hook 安装
    
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit       
    which nvidia-container-runtime-hook
    
    sudo systemctl restart docker
    
Docker with GPU 测试

    docker run -it --rm --gpus all ubuntu nvidia-smi
    docker run --gpus all --rm --name tensorflow -ti tensorflow/tensorflow:latest-gpu python -c 'import tensorflow as tf;print(tf.test.is_gpu_available())'
    
    
虽然 Docker 19.03 默认已经支持 GPU，但只是 docker run 支持，而 docker service 和 docker-compose 都不支持，所以还需要安装 nvidia-container-runtime 并设置。

nvidia-container-runtime 安装

    sudo apt-get install nvidia-container-runtime
    
为 Docker 增加 nvidia runtime

    sudo tee /etc/docker/daemon.json <<EOF
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
    EOF
    sudo pkill -SIGHUP dockerd

默认使用 nvidia runtime

    
    sudo mkdir -p /etc/systemd/system/docker.service.d
    # VRAM 表示显存大小，主要用于 docker service 时使用 --generic-resource 找到有显卡的节点
    sudo tee /etc/systemd/system/docker.service.d/override.conf <<EOF
    [Service]
    ExecStart=
    ExecStart=/usr/bin/dockerd -H fd:// --default-runtime=nvidia --containerd=/run/containerd/containerd.sock --node-generic-resource VRAM=8000
    EOF
    vi /etc/nvidia-container-runtime/config.toml
        # 取消下面这行的注释
        swarm-resource = "DOCKER_RESOURCE_GPU"
    
    sudo systemctl daemon-reload
    sudo systemctl restart docker

Docker Service 测试

    # 本机不使用 --gpus 参数启动容器并查看 GPU 设备
    docker run --rm --name tensorflow -ti tensorflow/tensorflow:latest-gpu ls /dev/nvidia0
    # 随机节点启动容器，启动后通过 docker service ps 找到节点，并到节点机器上 docker exec 进入容器ls /dev/nvidia0，下同
    docker service create tensorflow/tensorflow:latest-gpu-py3-jupyter
    # 指定节点启动容器
    docker service create  --constraint 'node.id == wnsgpwm1if9a7op0h9atrvbjb' tensorflow/tensorflow:latest-gpu-py3-jupyter
    # 自动选择定义了 VRAM 的节点
    docker service create --generic-resource VRAM=3000  tensorflow/tensorflow:latest-gpu-py3-jupyter
    # 设置显存自适应，以便多容器共用一块 GPU
    docker service create --env TF_FORCE_GPU_ALLOW_GROWTH=true --generic-resource VRAM=3000  tensorflow/tensorflow:latest-gpu-py3-jupyter
    # 映射 8888 端口，浏览器打开 http://localhost:8888 ，密码可进入容器后 jupyter notebook list 查看
    docker service create --publish 8888:8888 --env TF_FORCE_GPU_ALLOW_GROWTH=true \
        --generic-resource VRAM=3000  tensorflow/tensorflow:latest-gpu-py3-jupyter
    
    
    
参考链接

- https://github.com/NVIDIA/nvidia-container-runtime/
- https://github.com/NVIDIA/nvidia-docker
- https://codingsans.com/blog/gpu-orchestration-using-docker
- https://docs.docker.com/config/containers/resource_constraints/
- http://cowlet.org/2018/05/21/accessing-gpus-from-a-docker-swarm-service.html


https://zhuanlan.zhihu.com/p/52096200


锁屏
%windir%\system32\rundll32.exe user32.dll,LockWorkStation

export SYSTEMD_EDITOR=vim
systemctl edit docker


cat <<EOF >003.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

import time
def time_matmul(x):
    start = time.time()
    for loop in range(10000):
        tf.matmul(x, x)
    result = time.time() - start
    print('10000 loops: {:0}ms'.format(1000*result))


x = tf.random.uniform([1000, 1000])
assert x.device.endswith('GPU:0')
time_matmul(x)


EOF

每隔 30 分钟定时锁屏休息
SCHTASKS /Create /SC MINUTE  /MO 30 /TN 定时锁屏 /TR "%windir%\system32\rundll32.exe user32.dll,LockWorkStation"


限制 GPU 使用率
https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
How to prevent tensorflow from allocating the totality of a GPU memory?

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

TF_FORCE_GPU_ALLOW_GROWTH=true python 008.py

sudo 命令继承当前环境变量
sudo -E xxx



Free Ping and Traceroute Tool
https://www.manageengine.com/free-ping-tool/free-ping-tool-index.html



FreeFileSync(免费文件同步工具) v10.18中文版
http://www.pc6.com/softview/SoftView_46868.html

免费个人数据备份软件介绍：FreeFileSync、Syncthing
https://cloud.tencent.com/developer/news/180449


FreeFileSync我用在离线备份，定期接上移动硬盘，打开同步方案，同步一下。

Syncthing我用在在线远程备份。

FreeFileSync可以保持2个文件夹同步，支持本地文件夹、网上邻居、FTP。文件比较根据文件更新时间、大小（1T几分钟就可以检查完）或者内容（很慢，没试过）比较。可以后台自动同步。带简单的版本管理。可以建立配置文件，保存多个同步方案。使用非常简单，选择2个目录，检查不同，同步即可。


文件同步软件：filegee，FreeFileSync，goodsync，坚果云，nextcloud，seafile,syncthings


git apply是另外一种打patch的命令，其与git am的区别是，git apply并不会将commit message等打上去，打完patch后需要重新git add和git commit，而git am会直接将patch的所有信息打上去，而且不用重新git add和git commit,author也是patch的author而不是打patch的人


How to resolve ALL conflicts using HEAD, with any mergetool
https://stackoverflow.com/questions/20229222/how-to-resolve-all-conflicts-using-head-with-any-mergetool

git reset --hard HEAD
git merge -Xours origin/master

Git – Resolve Merge Conflicts
https://easyengine.io/tutorials/git/git-resolve-merge-conflicts/
git checkout --ours PATH/FILE
git checkout --theirs PATH/FILE

git checkout --conflict=merge .
git checkout --ours .


如何用git命令生成Patch和打Patch
https://www.cnblogs.com/ArsenalfanInECNU/p/8931377.html

# A
git format-patch HEAD^^
# B
git apply --stat 0002-.patch
git apply --check 0002-.patch
git apply  0002-.patch
git checkout --theirs packages/notebook-extension/src/index.ts

### 装机必备：
Everything
VNC Viewer
Putty
WinSCP
Notepad++
FileZilla
Foxmail
Git
7-zip
Python
VS Code
WPS Office
Chrome
QQ 拼音
ReplaceGoogleCDN-master

## 重装系统

备份

- chrome 收藏夹
- putty 配置
- git key
- 我的文档，图片



现在坚果云有 4 条相当清晰的产品线：面向个人用户有免费的普通版和付费的专业版，另外还有为中小企业提供的团队版，以及适合大企业使用的企业版。他们的个人用户和中小企业用户基本各占一半。

易用安全，两个坚持了 6 年的原则

在韩竹看来，坚果云的最大优势在于易用和安全。



mkdir ~/.pip/

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
pip install jupyterlab==2.0.0a1
curl -sL https://deb.nodesource.com/setup_12.x |  bash -
apt-get install nodejs


main.aed71038718d8fcdfab8.js
vendors~main.f0b5361fc5ec3d5b3d77.js

terminal should be login shell, not interactive shell
https://github.com/jupyterlab/jupyterlab/issues/3094


Ubuntu修改默认sh为bash

查看当前 shell
ps -p $$
echo $0

修改默认 shell

usermod -s /bin/bash root
sudo -u root chsh -s /bin/bash
dpkg-reconfigure bash
Change Shell To Bash in Linux / Unix
https://www.cyberciti.biz/faq/how-to-change-shell-to-bash/


Python中使用rrdtool结合Django进行带宽监控
https://blog.csdn.net/orangleliu/article/details/52851122
python3 rrdtool 使用
https://www.cnblogs.com/FRESHMANS/p/8427737.html

yum install rrdtool-devel python36-devel
pip install rrdtool


import rrdtool
import time

cur_time = str(int(time.time()))  # 获取当前Linux时间戳作为rrd起始时间
# 数据写频率--step为300秒(即5分钟一个数据点)
rrd = rrdtool.create('/root/xls/Flow.rrd', '--step', '300', '--start', cur_time,
　　　　　　　　　　　　 #定义数据源ens32_in(入流量)、ens32_out(出流量)；类型都为COUNTER(递增)；600秒为心跳值,其含义是600秒没有收到值，则会用UNKNOWN代替；0为最小值；最大值用U代替，表示不确定
                     'DS:ens32_in:COUNTER:600:0:U',　　'DS:ens32_out:COUNTER:600:0:U',
                     #RRA定义格式为[RRA:CF:xff:steps:rows]，CF定义了AVERAGE、MAX、MIN三种数据合并方式
　　　　　　　　　　　　 #xff定义为0.5，表示一个CDP中的PDP值如超过一半值为UNKNOWN，则该CDP的值就被标为UNKNOWN
　　　　　　　　　　　　 #下列前4个RRA的定义说明如下，其他定义与AVERAGE方式相似，区别是存最大值与最小值
　　　　　　　　　　　　 'RRA:AVERAGE:0.5:1:600',
                     'RRA:AVERAGE:0.5:6:700',
                     'RRA:AVERAGE:0.5:24:775',
                     'RRA:AVERAGE:0.5:288:797',
　　　　　　　　　　　　　
                     'RRA:MAX:0.5:1:600',　　　　　　# 每隔5分钟(1*300秒)存一次数据的最大值,存600笔，即2.08天
                     'RRA:MAX:0.5:6:700',　　　　　　# 每隔30分钟(6*300秒)存一次数据的最大值,存700笔，即14.58天（2周）
                     'RRA:MAX:0.5:24:775',　　　　　 # 每隔2小时(24*300秒)存一次数据的最大值,存775笔，即64.58天（2个月）
                     'RRA:MAX:0.5:444:797',　　　　　# 每隔24小时(288*300秒)存一次数据的最大值,存797笔，即797天(2年)
                     'RRA:MIN:0.5:1:600',
                     'RRA:MIN:0.5:6:700',
                     'RRA:MIN:0.5:24:775',
                     'RRA:MIN:0.5:444:797')
if rrd:
    print (rrdtool.error())
	
	
reate filename [--start|-b start time] [--step|-s step] [DS:ds-name:DST:heartbeat:min:max] [RRA:CF:xff:steps:rows]方法，创建一个后缀为rrd的rrdtool数据库，参数说明如下：

filename创建的rrdtool数据库文件名，默认后缀为.rrd；

--start指定rrdtool第一条记录的起始时间，必须是timestamp的格式；

--step指定rrdtool每隔多长时间就收到一个值，默认为5分钟；

DS用于定义数据源，用于存放脚本的结果的变量；

DST用于定义数据源类型，rrdtool支持COUNTER（递增类型）、DERIVE（可递增可递减类型）、ABSOLUTE（假定前一个时间间隔的值为0，再计算平均值）、GUAGE（收到值后直接存入RRA）、COMPUTE（定义一个表达式，引用DS并自动计算出某个值）5种，比如网卡流量属于计数器型，应该选择COUNTER；

RRA用于指定数据如何存放，我们可以把一个RRA看成一个表，保存不同间隔的统计结果数据，为CF做数据合并提供依据，定义格式为：[RRA:CF:xff:steps:rows]；

CF统计合并数据，支持AVERAGE（平均值）、MAX（最大值）、MIN（最小值）、LAST（最新值）4种方式。


Python3.6关于python-rrdtool报错的问题
https://blog.csdn.net/Mr_Yang__/article/details/81145329
https://pythonhosted.org/rrdtool/install.html#source-code


yum install -y gettext libffi pcre glib libpng freetype fontconfig pixman cairo fribidi graphite2 icu4c harfbuzz pango libtiff-devel.x86_64

wget https://oss.oetiker.ch/rrdtool/pub/rrdtool-1.5.6.tar.gz

tar zxvf rrdtool-1.5.6.tar.gz
cd rrdtool-1.5.6
./configure --prefix=/usr/local/rrdtool
make && make install

ln -s /usr/local/lib/librrd* /usr/lib/
git clone https://github.com/commx/python-rrdtool.git
cd python-rrdtool
python setup.py install


CentOS 6.X 安装中文字体
https://blog.csdn.net/wh211212/article/details/78730474
# 中文
yum install -y fontconfig mkfontscale
fc-list
fc-list :lang=zh

unzip win-fonts.zip -d /usr/share/fonts/
cd /usr/share/fonts/
mkfontscale
mkfontdir
fc-cache


export FLASK_ENV=production
flask run --port 8001


4 年 Java 经验面试总结、心得体会 - 程序员囧辉的文章 - 知乎
https://zhuanlan.zhihu.com/p/79224082

面试挂了阿里却拿到网易offer，一个三年Java程序员的面试总结！ - <em>Java</em>架构师的文章 - 知乎
https://zhuanlan.zhihu.com/p/61449992

专业技能方面
- 基础：JDK 常用类的原理、源码、使用场景。
- 设计模式：常用几种的原理、使用场景，单例、动态代理、模板、责任链等。
- 数据结构：数组、链表、栈、队列、树。
- 网络：TCP、HTTP、HTTPS、负载均衡算法。
- 框架：Spring IoC 原理、Spring AOP 原理和使用、Spring 常用的扩展点、MyBatis 的核心流程。
- 中间件：常用中间件的核心原理与最佳实践，并对其中的 1 到 2 个有深入的学习，Redis、Kafka（RocketMQ、RabbitMQ）、Dubbo、Zookeeper。
- 数据库（MySQL）：索引原理、隔离级别、锁机制、分库分表、慢 SQL 定位及优化、线上问题解决。
- Netty：NIO 原理、核心组件、I/O 多路复用（epoll）、零拷贝。
- JVM：运行时数据区、垃圾回收算法、垃圾回收器（CMS、G1）、常用配置参数、线上问题定位及解决。
- 稳定性保障：隔离、限流、熔断、降级等。
- Linux：基本命令的使用、快速定位和排查问题。
- 分布式理论：CAP、BASE、2PC、3PC、TCC。

项目方面
- 能独立完成一个复杂模块的需求分析、方案设计和最终落地实现。
- 能不断思考，寻找更优的设计和解决方案，积极优化慢 SQL、慢服务。
- 具备排查问题的能力，遇到线上问题能及时定位和修复上线，例如：数据库死锁、服务器宕机、服务器 Full GC 频繁等。
- 具备难题攻关的能力，能不断解决项目遇到的挑战，能给予初级工程师技术上的指导。
- 初步具备带领团队（1-3人左右）的能力，能合理分配需求，做好进度把控、风险评估、Code Review。


常见的如以下：

介绍下你参与度最高的项目
画下项目的架构图
如果核心流程处理到一半，服务器崩溃了，会怎么处理
项目中遇到过哪些挑战或问题，怎么解决的
项目的稳定性和可用性怎么保障
数据安全这块怎么设计
项目的技术选型，为什么选这些


Hr 面主要是了解候选人的一些通用素质，经常会问的问题如下：

介绍下自己投入最多的项目（当时我就惊了，Hr 也开始问项目了）
离职的原因
当前的薪资、绩效
当前在面试的其他公司的情况
平时有没有学习的习惯，怎么学习的，现在在学习什么
未来的规划

1. 我是谁 （1句话）
2. 我的三个亮点，最近最相关的经历？（3句话）
3. 我为什么想要这份工作？（1句话）




jupyter labextension list
jupyter kernelspec list

jupyterlab+nodejs

npm config set registry=https://registry.npm.taobao.org
npm config set prefix $HOME
npm config set unsafe-perm true
npm install -g ijavascript
ijsinstall

apt-get update
apt install r-base-core

jupyter labextension install jupyterlab-drawio
jupyter labextension install @jupyterlab/toc
jupyter labextension install jupyterlab_voyager

apt-get install git
git clone https://github.com/SpencerPark/IJava.git
cd IJava/
chmod u+x gradlew && ./gradlew installKernel


echo 'export GOPATH=~/go' >> ~/.bashrc
source ~/.bashrc
go get -u github.com/gopherdata/gophernotes
mkdir /usr/local/share/jupyter/kernels/gophernotes
cp $GOPATH/src/github.com/gopherdata/gophernotes/kernel/* /usr/local/share/jupyter/kernels/gophernotes
cd /usr/local/share/jupyter/kernels/gophernotes


A gallery of interesting Jupyter Notebooks
https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks

利器|JupyterLab 数据分析必备IDE完全指南
https://zhuanlan.zhihu.com/p/67959768


nodejs jupyterlab-extension
https://www.npmjs.com/search?q=keywords%3Ajupyterlab-extension


How to fix ‘Configuring tzdata’ interactive input when building Docker images
https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/

ENV DEBIAN_FRONTEND=noninteractive


Recursive sub folder search and return files in a list python
Recursive sub folder search and return files in a list python


How can I detect only deleted, changed, and created files on a volume? NTFS
https://stackoverflow.com/questions/7421440/how-can-i-detect-only-deleted-changed-and-created-files-on-a-volume/7459109#7459109

演练：创建传统的 Windows 桌面应用程序C++
https://docs.microsoft.com/zh-cn/cpp/windows/walkthrough-creating-windows-desktop-applications-cpp?view=vs-2017
https://docs.microsoft.com/zh-cn/windows/win32/LearnWin32/learn-to-program-for-windows

精通Windows API 范文庆/周彬彬/安靖著 中文 PDF版 [50M]
https://www.jb51.net/books/67342.html

C++ 中 Windows 编程概述
https://docs.microsoft.com/zh-cn/cpp/windows/overview-of-windows-programming-in-cpp?view=vs-2017

wtl学习总结
https://www.cnblogs.com/wliangde/p/4259281.html


VS2015支持windowsxp
https://blog.csdn.net/kl222/article/details/54376591

1. 项目菜单->项目属性->配置属性->常规->平台工具集，选择“VS2013WindowsXP(v120_xp)”;
2. 项目菜单->项目属性->配置属性->常规->MFC的使用，选择使用标准Windows库;如果选用静态库编译的话选用静态库中选用MFC。
3. 项目菜单->项目属性->配置属性->常规->字符集中使用多字节字符集或使用Unicode字符集
4. 项目菜单->项目属性->链接器->系统->子系统->控制台或窗口windows（根据你的项目类型选择），第二项版本号设成5.01。


Visual Studio 2015 - Windows XP (v140_xp) 编译工具 ucrtbased.dll缺失
https://blog.csdn.net/atceedsun/article/details/53583824



VS2017编译在XP环境下运行的程序
https://blog.csdn.net/qq_41917908/article/details/83512386


如果VS2017生成的程序在xp系统提示缺少VCRUNTIME140D.dll，这是因为程序采用了动态编译，只要进行静态编译即可解决
如下图所示，使用多线程MT就可以解决这个问题。C/C++ / 代码生成，运行库，选择多线程（/MT）

但是问题又来了，VS2017以MT方式链接编译出来的exe还是没法在xp下正常运行，这是因为PE文件中的主系统版本号，不选子系统的情况下默认是6，也就是win7，我们还得修改版本号来适配XP

多线程MT和多线程MD的区别
https://www.cnblogs.com/mod109/p/3762604.html

对于多线程MT的程序来说，其连接的是libcmt.lib,该文件属于C语言运行时库，整个lib都会连接到PE文件当中。而多线程MD的程序链接的却是类似msvcpXXX.dll，该文件属于微软运行时库.也就是说如果是多线程MD编译出来的文件运行时都会加载相应版本的运行时库，当如果找不到运行时库就会报错而无法运行，同时如果运行时库不匹配也会出现各种意料之外的崩溃或者程序根本跑不起来等情况。


QUIC协议学习记录
https://www.cnblogs.com/mod109/p/7372577.html


首先介绍一下可以打包Python代码的工具：

py2exe: 这个是知名度最高的，但是好像不太适合新手，需要各种库，用起来比较繁琐，不推荐。
pyinstaller : 可以只是生成单独的可执行程序。 最新版本为3.2.1 Stable, supports Python 2.7, 3.3–3.7。 可以自定义图标。 跨平台，兼容性良好。

cx_Freeze :
这个打包质量挺好，操作也简单。缺点是不能生产单独的可执行文件，结果就是exe文件被淹没在众多文件中，看着不高大上。

WIN32下的按钮标准控件
https://blog.csdn.net/zhangkai19890929/article/details/90069188

首先在win32下所有的控件其实都是子窗口.，所以我们收到WM_CREATE消息后，创建控件的.

“WCHAR *”类型的实参与"const char *"类型的形参不兼容
https://blog.csdn.net/u012686154/article/details/86241162
https://www.cnblogs.com/dbylk/p/3696367.html

通过查看LPCSTR的定义：

typedef _Null_terminated_ CONST CHAR *LPCSTR, *PCSTR; 
可以知道LPCSTR代表了const char *类型，它是一个指向以'\0'结尾的8位（单字节）ANSI字符数组的常量指针，

而const wchar_t *类型是一个指向'\0'结尾的16位（双字节）Unicode字符数组的常量指针.

在VS2013编译器中直接输入的字符串常量（如“abc”）默认是以const char *的格式（即ANSI编码）储存的，因此会导致类型不匹配的编译错误。

 

解决的办法有两个：

第一个方法是右击“解决方案资源管理器”中的项目，“属性→配置属性→常规→项目默认值→字符集”，默认的选项是“使用多字节字符集”，将它改为“使用Unicode字符集”即可。

这样，输入的字符串会默认以const wchar_t *格式储存。

第二个方法是使用_T宏，它在“tchar.h”中定义，它能够自动识别当前编译器字符串的储存格式并做出相应转换，避免这种类型的编译错误。

具体使用方法为：将“abc”改为_T("abc")。


[原创]Win32 编程进阶:打造自己的标准控件
https://bbs.pediy.com/thread-129283.htm

使用Win32/ATL建立窗口的过程
https://www.cnblogs.com/liangxiaofeng/p/5066026.html

有时候想写个几十kb的小程序，MFC实在是太大了，Win32有时又太麻烦，怎么办呢？用ATL写会更方便和轻量级一些 ATL和MFC不同的是，ATL并没有强制性封装WinMain函数，仅封装了WndProc，所以使用ATL写Windows程序有很高的自由度


WTL,ATL与MFC之比较
https://blog.csdn.net/jack_incredible/article/details/7962488


早期的VC++开发者们发现了MFC(Microsoft Foundation Classes) 这样一个好东东。他们发现，MFC提供了一个强大的类库，很好的满足了面向对象编程的需要。随着泛型编程技术的发展和时间的推移，慢慢地，他们慢慢觉得MFC的类库过于庞大和宽泛，而且它提供的模板库只覆盖了很有限的领域。于是，ATL应运而生。

ATL(Active Template Library),简称活动模板库。ATL的设计者们通过它来它提供快速的COM组件封装。ATL很好地体现了用模板进行编程的思想。如果模板设计得当，就很容易灵活的适应各种不同的需求，而且更容易跟进新技术的发展需求。

但是，作为泛型编程的爱好者们始终要面对的一个主要难题，那就是如何方便地进行图形用户界面（GUI）设计。因为ATL只是提供了对Win32/64窗口的低级封装，因此ATL的使用者们在用ATL进行COM开发的同时，不得不借助于MFC来进行COM组件的UI编程。因为在GUI方面，ATL确实是爱莫能助。

WTL(Windows Template Library)在体现模板编程思想的同时，对模板进行了很好的高级封装，很好的满足了UI编程的各种需求。这也是WTL设计者们的用意所在。在用WTL生成应用程序的时候不需要将DLL文件与EXE文件一起交付给用户，而且，WTL又有很好的兼容性。你可以将它与ATL, STL, VC++ 数据模板，第三方模板，或者你自己的模板这几种中的任何一种一起使用。正因为这些特点，使得当前WTL成为了大部分高级的C++开发者们进行UI设计时的首选。



https://www.nuget.org/packages/wtl


Fake Everything
Everything的原理猜想与实现
https://github.com/Artwalk/Fake-Everything


Windows平台Python编程必会模块之pywin32
https://www.cnblogs.com/achillis/p/10462585.html



使用 ps 命令查看进程启动的精确时间和启动后所流逝的时间
ps -eo pid,lstart,etime,cmd | grep nginx

查看CPU最高的进程
ps -aux --sort=-pcpu|head -10

linux锁定用户和解锁用户
https://blog.csdn.net/qq_37699336/article/details/80296670

1、禁止个别用户登录。比如禁止lynn用户登录。
passwd -l test
这就话的意思是锁定test用户，这样该用户就不能登录了。
passwd -u test
对锁定的用户lynn进行解锁，用户可登录了。

3、禁止所有用户登录。
touch /etc/nologin
除root以外的用户不能登录了！

change-journals USN
https://docs.microsoft.com/zh-cn/windows/win32/fileio/change-journals

How to get the 'NextUSN' journal entry for a VSS snapshot?
https://stackoverflow.com/questions/10544433/how-to-get-the-nextusn-journal-entry-for-a-vss-snapshot
VSS（卷影拷贝服务）与snapshot优缺点及区别
https://blog.csdn.net/frankarmstrong/article/details/77370253


关于NTFS-MFT
https://www.jianshu.com/p/8471b7f4152a

引导扇区是从NTFS文件系统的第一个扇区开始，以55 AA结尾。我们主要关注前88字节的信息，其中重要的就是“NTFS”标识、扇区大小、每簇扇区数、MFT起始簇以及MFT备份MFTMirr位置这些信息。

https://www.cnblogs.com/mwwf-blogs/archive/2015/05/04/4467687.html

How do we access MFT through C#
https://stackoverflow.com/questions/21661798/how-do-we-access-mft-through-c-sharp

NtfsReader .NET Library
https://sourceforge.net/projects/ntfsreader/

Undelete a file in NTFS
https://www.codeproject.com/Articles/9293/Undelete-a-file-in-NTFS


选择C/C++ -> 常规 -> 调试信息格式 -> 选择 程序数据库 (/Zi)


wchar_t DWORD WORD _T _L 的区
https://bbs.csdn.net/topics/350023270
32 bit windows:

wchar_t :宽字节变量类型，用于表示Unicode字符，

它实际定义在<string.h>里：typedef unsigned short wchar_t。

DWORD 四字节无符号整数， unsigned long

WORD 二字节无符号整数    unsigned short long

为了让编译器识别Unicode字符串，必须以在前面加一个“L”,如

wchar_t a[] = L"Hello!" ;


_T( )是定义在头文件tchar.h中的宏，视是否定义了_UNICODE宏而定义成： 
定义了_UNICODE：    #define _T(x) L##x 
没有定义_UNICODE： #define _T(x) x 

Converting contents of a byte array to wchar_t*
https://stackoverflow.com/questions/13887348/converting-contents-of-a-byte-array-to-wchar-t


终于理解了什么是LGPL
https://www.cnblogs.com/findumars/p/3556883.html

因 此LGPL协议的开源 代码很适合作为第三方类库被商业软件引用，但不适合希望以LGPL协议代码为基础，通过修改和衍生的方式做二次开发的商业软件采用。

假设我们使用一个名为 Lib 的库，这个库是基于 LGPL协议发布的。如果你使用 Lib.dll 做动态链接（Windows 下），好，一切 OK。无论你的程序怎么样，你都可以做你所做的事情。

如果你因某种原因必须静态链接一个基于 LGPL 协议发布的库（一下我们简称为 LGPL 库），那么，你有义务进行下面的工作：

你必须在你的文档中说明，你的程序中使用了 LGPL 库，并且说明这个库是基于 LGPL 发布的；
你必须在你的应用程序发布中包含一份 LGPL协议，通常就是那个文本文件；
你必须开放使用了 LGPL 库代码的所有代码，例如某些封装器。但是，其他使用这些封装器的代码就不需要开放了；
你必须包含你的应用程序的余下部分的目标文件（通常就是我们所说的 .o 等等），或者是其他等价的文件。源代码并不是必须的。

WinForm ListView虚拟模式加载数据 提高加载速度
https://www.cnblogs.com/kest/p/4659421.html
C# ListView双缓存代码，解决闪屏问题
https://www.cnblogs.com/94cool/archive/2013/02/06/2899995.html
BinarySearch limits for numbers within range
https://stackoverflow.com/questions/8912613/binarysearch-limits-for-numbers-within-range

倒排索引
C# Inverted Index Example
https://www.dotnetperls.com/inverted-index
https://stackoverflow.com/questions/2110142/writing-an-inverted-index-in-c-sharp-for-an-information-retrieval-application
C# algorithm for N-gram
https://stackoverflow.com/questions/3829110/c-sharp-algorithm-for-n-gram
N-gram and Fast Pattern Extraction Algorithm
https://www.codeproject.com/articles/20423/n-gram-and-fast-pattern-extraction-algorithm
http://jakemdrew.com/blog/ngram.htm

严格遵守 GPL 的代码如何商用？
https://www.zhihu.com/question/19703551

GPL 只是规定用户在获取你的程序的时候必须可以获得源代码，但并没有规定必须免费，因此理论上说，你仍然可以收取费用。不过，由于 GPL 规定你不得阻止用户再分发，因此用户完全可以从你这里买来代码之后再免费送给所有其它人，因此对于 GPL 代码想要收费发布难度是很大的，目前比较可行的办法是像 Redhat 那样，通过提供订阅和服务的方式来收费，提供一些额外的增值服务吸引用户交费。


/usr/bin/pipework br0 -i eth0 d_centos_ebms 1.1.1.1/23@2.2.2.2


Docker：使用pipework配置docker网络
https://www.cnblogs.com/1994ghj/p/5008287.html
https://my.oschina.net/guol/blog/345038

Pipework是一个Docker配置工具，是一个开源项目，由200多行shell实现。

Pipework是一个集成工具，需要配合使用的两个工具是OpenvSwitch和Bridge-utils。

FileLocator使用指南
https://blog.csdn.net/code4101/article/details/83029094
FileLocator Pro：强大高效的无索引全文搜索软件
https://blog.csdn.net/oLanLanXiaRi/article/details/48250711

文件搜索一直是大家平时最常用的功能之一。在搜索文件名方面，目前已有不少软件做到了极致，比如 Everything ( 官网 | 介绍 ) 以及 操作新颖的晚辈 Listary Pro ( 官网 | 介绍 ) 都是此领域之佳品。但有时，仅通过文件名并不足以快速找到所需文件和内容。因此，支持全文搜索的软件，也是重度知识管理、搜索用户的必备工具。这方面的解决方案中，Windows自带的索引功能，因为效率和占用资源问题，基本被用户抛弃（新版本有改进）。Google桌面及百度桌面，都基本停止更新。CDS（Copernic Desktop Search） 在国外有一定知名度，提供了免费家用版、专业版、企业版，但因为早期中文支持不佳，在国内用户极少（据称最新版本对中文支持极好）。倒是开源免费、小巧绿色的 Locat32 和开源免费、跨平台的 DocFetcher 得到了国内用户的青睐。

今天 LYcHEE 所介绍的，是另一款全文检索软件 FileLocator Pro ( 介绍 | 中文帮助 )。相比同类软件，它的特点是：支持更多格式与压缩包，搜索速度更快，无索引不占硬盘空间、支持 多种搜索规则 及 日期、属性等细节设定。FileLocator Pro 在国内的影响正大迅速扩大，相关介绍文章还可参见小众软件、精品绿色便携软件。


在docker使用devicemapper作为存储驱动时，默认每个容器和镜像的最大大小为10G。如果需要调整，可以在daemon启动参数中，使用dm.basesize来指定，但需要注意的是，修改这个值，不仅仅需要重启docker daemon服务，还会导致宿主机上的所有本地镜像和容器都被清理掉。

使用aufs或者overlay等其他存储驱动时，没有这个限制。

无论宿主机有多少个cpu或者内核，--cpu-shares选项都会按照比例分配cpu资源。另外只有一个容器时--cpu-shares选项无意义。

$ docker run -it --rm --cpu-period=100000 --cpu-quota=200000 u-stress:latest /bin/bash
这样的配置选项是不是让人很傻眼呀！100000 是什么？200000 又是什么？ 它们的单位是微秒，100000 表示 100 毫秒，200000 表示 200 毫秒。它们在这里的含义是：在每 100 毫秒的时间里，运行进程使用的 CPU 时间最多为 200 毫秒(需要两个 CPU 各执行 100 毫秒)。要想彻底搞明白这两个选项的同学可以参考：CFS BandWith Control。我们要知道这两个选项才是事实的真相，但是真相往往很残忍！还好 --cpus 选项成功的解救了我们，其实它就是包装了 --cpu-period 和 --cpu-quota。

Docker: 限制容器可用的 CPU
https://www.cnblogs.com/sparkdev/p/8052522.html
Docker: 限制容器可用的内存
https://www.cnblogs.com/sparkdev/p/8032330.html


当 CPU 资源充足时，设置 CPU 的权重是没有意义的。只有在容器争用 CPU 资源的情况下， CPU 的权重才能让不同的容器分到不同的 CPU 用量。--cpu-shares 选项用来设置 CPU 权重，它的默认值为 1024。我们可以把它设置为 2 表示很低的权重，但是设置为 0 表示使用默认值 1024。
下面我们分别运行两个容器，指定它们都使用 Cpu0，并分别设置 --cpu-shares 为 512 和 1024：

$ docker run -it --rm --cpuset-cpus="0" --cpu-shares=512 u-stress:latest /bin/bash
$ docker run -it --rm --cpuset-cpus="0" --cpu-shares=1024 u-stress:latest /bin/bash




stress工具使用指南和结果分析
https://www.cnblogs.com/muahao/p/6346775.html

stress -c 4 -t 100
stress --vm 10 --vm-bytes 1G --vm-hang 100 --timeout 100s

C# ListView实例：文件图标显示
https://blog.csdn.net/dufangfeilong/article/details/41744847

Matching strings with wildcard
https://stackoverflow.com/questions/30299671/matching-strings-with-wildcard

Super Fast String Matching in Python
https://bergvca.github.io/2017/10/14/super-fast-string-matching.html

mport re

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

print('All 3-grams in "McDonalds":')
ngrams('McDonalds')

基于python+whoosh的全文检索实现
https://blog.csdn.net/qq_21149391/article/details/79509251


https://www.how2shout.com/tools/opensource-tools-for-artificial-intelligence-ai.html

Best AI Platforms Software
https://www.g2.com/categories/ai-platforms

Top 45 Artificial Intelligence Companies
https://www.datamation.com/artificial-intelligence/top-artificial-intelligence-companies.html


https://www.predictiveanalyticstoday.com/artificial-intelligence-platforms/

PAI实现的深度学习网络可视化编辑功能-FastNerualNetwork
https://blog.csdn.net/cpongo1/article/details/89543074

刚刚，阿里重磅发布机器学习平台 PAI 3.0！
https://www.infoq.cn/article/NxTj8-xC2KIZxGiFVMMu

阿里重磅开源Blink：为什么我们等了这么久？
https://www.jianshu.com/p/37a6acfc3124


在线的caffe网络可视化工具： http://ethereon.github.io/netscope/quickstart.html

https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/


FlinkML是Flink内部的机器学习工具库。它是Flink生态圈的新组件，社区成员不断向它贡献新的算法。 
http://flink.iteblog.com/dev/libs/ml/index.html

深入理解Apache Flink核心技术
https://www.cnblogs.com/feiyudemeng/p/8998772.html


Apache Flink®生态所面临的机遇与挑战
https://yq.aliyun.com/articles/701178

比拼生态和未来，Spark和Flink哪家强？
https://zhuanlan.zhihu.com/p/42511707

Spark背后公司Databricks获2.5亿融资，估值27.5亿美元
https://blog.csdn.net/cpongo4/article/details/89119256


#删除所有未运行的容器（已经运行的删除不了，未运行的就一起被删除了）
sudo docker rm $(sudo docker ps -a -q)

#根据容器的状态，删除Exited状态的容器
sudo docker rm $(sudo docker ps -qf status=exited)


阿里云机器学习平台让机器学习不再远不可及，在平台上没有繁琐的公式和复杂的代码逻辑，用户看到的是各种分门别类被封装好的算法组件。在搭架实验的过程中，只要拖拽组件就可以快速的拼接成一个workflow。操作体验类似于搭积木，真正做到让小白用户也可以轻松玩转机器学习。“过去半个月才能搭建的一套数据挖掘实验，利用阿里云机器学习平台3个小时就可能解决”。

同时，平台的每一个实验步骤都提供可视化的监控页面，数据挖掘工程师可以实时的掌握模型的训练情况，可视化的结果评估组件也极高的提升了模型调试效率。 在深度学习黑箱透明化方面，我们也在不断的研发集成各种可视化的工具，包括开源的TensorBoard和自研的工具，为客户提供更多可参考的信息，缩短模型优化的过程。

阿里云机器学习——让人工智能触手可及
https://yq.aliyun.com/articles/72847

Code Free Machine Learning for Hadoop & Spark
https://rapidminer.com/products/radoop/

Create predictive models using the RapidMiner Studio visual workflow designer


WOFF字体没有必要再开启GZIP，因为这个字体文本本身就是压缩过的。
WOFF 2.0的mime type值，怎么说呢，有些许小争议。Google使用font/woff2，而W3C则推荐application/font-woff2。我个人的建议是这样的：在CSS base64字体表示时候使用Google的font/woff2，毕竟是web呈现；然后服务器配置走application/font-woff2。

从云厂商宕机史谈预案建设
https://www.infoq.cn/article/BdpCdzipGb10UN5RUhZ3
从二十个严重的配置故障中我们能学到什么？
https://www.infoq.cn/article/M9fY3XNJ5R53dLUNUZah
摆脱无效报警？十年运维监控报警优化经验总结
https://www.infoq.cn/article/1AofGj2SvqrjW3BKwXlN
ESLint 在中大型团队的应用实践
https://mp.weixin.qq.com/s?__biz=MjM5NjQ5MTI5OA==&mid=2651750496&idx=2&sn=2627a4bdd7e63343c8947ecf82b2349f&chksm=bd12592d8a65d03b68b25893759f62e044d91ec5dbe1cabac6246eaa61528c2ca8575d6aeef1&scene=21#wechat_redirect

超大规模服务的故障“弹性自愈”
https://www.infoq.cn/article/7NOuK1frElxWNYqEP0WJ

随着云计算的发展，多云和混合云逐渐成为新常态，Kubernetes 逐渐成为企业和云的新的交互界面。从架构的角度，Kubernetes 一方面起到了异构 IAAS 的 Control Plane 的作用，一方面也对底层基础设施提出了更高的要求，无论是立体的安全风险防控，还是 10 倍的应用密度和调用频率的提升。另一方面，企业的基础设施从也单机房向全球多地域演进，如何实现全球多地域的统一管理也是需要考虑的问题。本次分享，会介绍下阿里云的上万个 Kubernetes 的最佳实践，如何从自动化到 AIOps，如何打造云原生信息高速公路。

中台建设如火如荼。沉淀出一套可以复用的能力可帮助企业解决“重复造轮子”的问题，突破增长瓶颈，提升效率。本专题将就如何搭建中台以及中台建设过程中将会遇见哪些挑战等话题进行讨论。

随着微服务技术迅猛发展，Service Mesh 技术也逐渐走向成熟，成为当下最吸引眼球的技术热点，本专题将聚焦微服务创新与实践。

上云已经成为确定性的趋势，如何用好云？在云的环境下应该怎么做应用架构？本专题将探讨云时代的架构演进历程。

软件正在改变世界，而工程师改变软件。技术日新月异，怎样构建一个可持续发展的工程师能力成长模型促进职业发展？将是本专题重点关注的。


技术团队管理是一个很广泛的话题，是大多数研发同学职业生涯的必经之路，有明确汇报关系的技术团队管理，有项目维度轻量的技术团队管理，有小团队间小管理半径的技术团队管理，有大团队大管理半径松散的技术团队管理，甚至单纯为了攻克某个技术难题临时组建的小技术团队也需要管理，我觉得技术管理是一个技术人员必备的技能，所以在日常工作中，应该有意无意的去锻炼自己的技术管理能力，从小处入手，一步一个脚印，等将来真有这种机会的时候，就能信手拈来，把握住机会。

对于已经走上管理岗位的技术人员，就需要时刻牢记你的职责是什么，但基本离不开两点：团队目标的达成和团队的成长。目标的达成是一个团队存在的原因和根本，目标如果达不成，团队都可能面临解散的风险，所以，业务型团队的目标就是持续的赋能业务而不是把技术做到超越业务，技术研究型团队的目标就是持续的技术研究而不是偏离主线去做其他研究，而这些都相应的有可衡量的指标，努力带领团队达成指标是要务，与此同时，团队的成长是时刻要考虑的事情，团队的成长是目标完成过程中持续的补给，是定海神针，是每个团队成员最在乎的事，团队的成长就包括团队梯队的搭建和团队成员的个人成长，好的梯队能让团队更稳固更健康，而关注成员个人成长让个体对团队更忠诚。

技术团队管理就像一座秘密花园，当你试着去了解，去探索，去感悟，你会发现，真香。

云计算从最开始的想法到步入现实，从虚拟化到现在火热的云原生，新的概念层出不穷，比如容器、Kubernetes、Serverless、Service Mesh，更多是对更优雅灵活的技术架构的追求与探索，随着这些技术落地到生产环境，我们也逐渐步入了云时代的深水区。从 IaaS 到 SaaS 的性能都面对着不同的挑战与问题，也有了许多出色的解决方案。通过我们专题，希望可以让听众收获：

1. 云上的应用与服务如何通过优化表现出更佳的性能，如更稳定更短的延迟，更大的吞吐量；

2. 云上的应用与服务如何充分地利用硬件性能，如调度与混合部署，软硬一体。

不会数学统计没关系——5分钟教你轻松掌握箱线图
https://www.sohu.com/a/218322591_416207

在箱线图中，箱子的中间有一条线，代表了数据的中位数。箱子的上下底，分别是数据的上四分位数（Q3）和下四分位数（Q1），这意味着箱体包含了50%的数据。因此，箱子的高度在一定程度上反映了数据的波动程度。上下边缘则代表了该组数据的最大值和最小值。有时候箱子外部会有一些点，可以理解为数据中的“异常值”。


纯CSS实现左右拖拽改变布局大小
https://blog.csdn.net/github_35631540/article/details/92063430


优秀！ resize-bar的opacity：0保证resize-bar透明放在最底层 resize-line的points-event：none保证不影响到resize-bar的事件响应 .resize-bar:hover~.resize-line 保证鼠标悬停resize-bar的时候resize-line做出响应

当拖动之前将iframe隐藏，换成显示一个div的虚线框，拖拽完成之后隐藏虚线框且将iframe显示到虚线框位置
https://bbs.csdn.net/topics/370130464

解决方案:拖动的时候,使用一个透明的div或其他元素,把这个iframe遮住.
当拖动之前将iframe隐藏，换成显示一个div的虚线框，拖拽完成之后隐藏虚线框且将iframe显示到虚线框位置



linux下怎么退出vi编辑器，按esc没有用；vim recording
https://blog.csdn.net/hellocsz/article/details/82593482


virt-install \
    --accelerate \
    --name xp \
    --ram 2048 \
    --vcpus=2 \
    --controller type=scsi,model=virtio-scsi \
    --disk path=/var/lib/libvirt/images/xp.qcow2,size=50,sparse=true,cache=none,bus=virtio \
    --cdrom=/home/action/xp.iso \
    --graphics vnc,listen=0.0.0.0,port=5904 \
    --network bridge=br0

virsh edit xp    
    
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2' cache='none'/>
      <source file='/var/lib/libvirt/images/xp.qcow2'/>
      <target dev='vda' bus='virtio'/>
      <boot order='1'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x06' function='0x0'/>
    </disk>
    <disk type='file' device='cdrom'>
      <driver name='qemu' type='raw'/>
      <source file='/home/action/xp.iso'/>
      <target dev='hda' bus='ide'/>
      <readonly/>
      <boot order='2'/>
      <address type='drive' controller='0' bus='0' target='0' unit='0'/>
    </disk>

virsh start xp
virsh reboot xp
virsh shutdown xp
virsh destroy  xp



https://docs.fedoraproject.org/en-US/Fedora/18/html/Virtualization_Administration_Guide/sect-Attaching_and_updating_a_device_with_virsh.html

Docker管理工具-Swarm
https://www.cnblogs.com/bigberg/p/8761047.html

Swarm 主要功能

集群管理与Docker Engine集成:使用Docker Engine CLI来创建一个你能部署应用服务到Docker Engine的swarm。你不需要其他编排软件来创建或管理swarm。
分散式设计：Docker Engine不是在部署时处理节点角色之间的差异，而是在运行时扮演自己角色。你可以使用Docker Engine部署两种类型的节点，管理器和worker。这意味着你可以从单个磁盘映像构建整个swarm。
声明性服务模型： Docker Engine使用声明性方法来定义应用程序堆栈中各种服务的所需状态。例如，你可以描述由消息队列服务和数据库后端的Web前端服务组成的应用程序。
伸缩性：对于每个服务，你可以声明要运行的任务数。当你向上或向下缩放时，swarm管理器通过添加或删除任务来自动适应，以保持所需状态。
期望的状态协调：swarm管理器节点持续监控群集状态，并调整你描述的期望状态与实际状态之间的任何差异。 例如，如果设置运行一个10个副本容器的服务，这时worker机器托管其中的两个副本崩溃，管理器则将创建两个新副本以替换已崩溃的副本。 swarm管理器将新副本分配给正在运行和可用的worker。
多主机网络：你可以为服务指定覆盖网络(overlay network)。 当swarm管理器初始化或更新应用程序时，它会自动为容器在覆盖网络(overlay network)上分配地址。
服务发现：Swarm管理器节点为swarm中的每个服务分配唯一的DNS名称，并负载平衡运行中的容器。 你可以通过嵌入在swarm中的DNS服务器查询在swarm中运行中的每个容器。
负载平衡：你可以将服务的端口暴露给外部的负载均衡器。 在内部，swarm允许你指定如何在节点之间分发服务容器。
安全通信：swarm中的每个节点强制执行TLS相互验证和加密，以保护其自身与所有其他节点之间的通信。 你可以选择使用自签名根证书或来自自定义根CA的证书。
滚动更新：在上线新功能期间，你可以增量地应用服务更新到节点。 swarm管理器允许你控制将服务部署到不同节点集之间的延迟。 如果出现任何问题，你可以将任务回滚到服务的先前版本。

Swarm在scheduler节点（leader节点）运行容器的时候，会根据指定的策略来计算最适合运行容器的节点，目前支持的策略有：spread, binpack, random。

　　1）Random 顾名思义，就是随机选择一个Node来运行容器，一般用作调试用，spread和binpack策略会根据各个节点的可用的CPU, RAM以及正在运 行的容器的数量来计算应该运行容器的节点。

　　2）Spread 在同等条件下，Spread策略会选择运行容器最少的那台节点来运行新的容器，binpack策略会选择运行容器最集中的那台机器来运行新的节点。 使用Spread策略会使得容器会均衡的分布在集群中的各个节点上运行，一旦一个节点挂掉了只会损失少部分的容器。

　　3）Binpack Binpack策略最大化的避免容器碎片化，就是说binpack策略尽可能的把还未使用的节点留给需要更大空间的容器运行，尽可能的把容器运行在 一个节点上面。


Docker Swarm集群搭建教程
http://c.biancheng.net/view/3178.html

内置的 Swarm 安全机制
Swarm 集群内置有繁多的安全机制，并提供了开箱即用的合理的默认配置——如 CA 设置、接入 Token、公用 TLS、加密集群存储、加密网络、加密节点 ID 等。


Docker Swarm 包含两方面：一个企业级的 Docker 安全集群，以及一个微服务应用编排引擎。

集群方面，Swarm 将一个或多个 Docker 节点组织起来，使得用户能够以集群方式管理它们。

Swarm 默认内置有加密的分布式集群存储（encrypted distributed cluster store）、加密网络（Encrypted Network）、公用TLS（Mutual TLS）、安全集群接入令牌 Secure Cluster Join Token）以及一套简化数字证书管理的 PKI（Public Key Infrastructure）。我们可以自如地添加或删除节点。

编排方面，Swarm 提供了一套丰富的 API 使得部署和管理复杂的微服务应用变得易如反掌。通过将应用定义在声明式配置文件中，就可以使用原生的 Docker 命令完成部署。

此外，甚至还可以执行滚动升级、回滚以及扩缩容操作，同样基于简单的命令即可完成。


docker service create --name redis --mode global  redis:3.0.6

docker开启远程访问
https://blog.csdn.net/lynx7/article/details/84789682

vim /lib/systemd/system/docker.service
    ExecStart=/usr/bin/dockerd -H unix:///var/run/docker.sock -H tcp://0.0.0.0:2375
systemctl daemon-reload
systemctl restart docker

export DOCKER_HOST="tcp://0.0.0.0:2375"
docker ps
sudo docker -H tcp://0.0.0.0:2375 pssudo docker -H tcp://0.0.0.0:2375 ps


请教个 docker 的问题，我搞了个测试的 swarm 集群， 3 个 manager 节点，停掉 leader 节点后会自动选取出一个新的  leader 节点，但我要访问这个集群的话应该如何访问才能不会访问到挂掉的 manager 节点 ip 上呀？想到有四种方案
A、客户端缓存 3 个 master ip，哪个能通访问哪个
B、用 etcd 搭一个配置中心，把 swarm leader ip 存起来，再写个健康检查程序定时更新 leader ip，最后客户端从 etcd 读取 swarm ip
C、搭一个 DNS 服务器，写个健康检查程序定时更新 swarm ip 的 A 记录，最后客户端通过域名访问 swarm ip
D、客户端通过单一的 ip 访问 swarm，写一个监控程序当发现 swarm leader 漂移后告警，然后人工修改客户端配置并热更新程序。

从架构复杂度，可维护性，自动化的角度看最优的选择应该选哪个方案呀？

客户端新增 nginx 配置，如下

```
cat <<EOF >/etc/nginx/conf.d/docker.conf
upstream devdocker {
    server 192.168.1.200:2375 max_fails=1 fail_timeout=10s;
    server 192.168.1.201:2375 max_fails=1 fail_timeout=10s;
    server 192.168.1.202:2375 max_fails=1 fail_timeout=10s;
}

log_format dockerlog '$time_iso8601 $status $request_time $upstream_response_time $remote_addr'
                  ' $http_host $upstream_addr "$request" $body_bytes_sent "$http_referer" '
                  '"$http_user_agent" "$http_x_forwarded_for"';
server {
    listen 2375;
    server_name  devdocker;

    access_log  logs/devdocker.access.log dockerlog;
    error_log  logs/devdocker.error.log;

    location / {
        proxy_pass  http://devdocker;

        proxy_set_header   Host             $host;
        proxy_set_header   X-Real-IP        $remote_addr;
        proxy_set_header   X-Forwarded-For  $proxy_add_x_forwarded_for;
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
   }
}
EOF
nginx -t && nginx -s reload
```

客户端修改 hosts

```
echo '127.0.1.1   devdocker' >> /etc/hosts
```

客户端测试访问 docker 集群，并查看请求日志

```
docker -H devdocker node ls
tail /usr/share/nginx/logs/devdocker.access.log
```

登录到 192.168.1.200，201，203 任意一台服务器关掉 docker 服务 `systemctl stop docker`，
再次在客户端多次测试访问 docker 集群，不会有任何失败访问。



在Windows环境中使用Nginx, Consul, Consul Template搭建负载均衡和服务发现服务
https://www.cnblogs.com/lwqlun/p/8835867.html

Nginx+upstream针对后端服务器容错的配置说明
https://www.cnblogs.com/kevingrace/p/8185218.html

浅析AnyCast网络技术
https://www.cnblogs.com/zafu/p/9168617.html

最后，针对Anycast 做如下总结：

 

优点：

Anycast可以零成本实现负载均衡，无视流量大小。

Anycast是天然的DDOS防御措施，体现了大事化小，小事化了的解决方法。

部署Anycast可以获得设备的高冗余性和可用性。

Anycast适用于无连接的UDP，以及有连接的TCP协议。

 

缺点：

Anycast严重依赖于BGP的选路原则，在整个Internet网络拓扑复杂的情况下，会导致次优路由选择。


选择存储驱动并正确地配置在 Docker 环境中是一件重要的事情，特别是在生产环境中。

下面的清单可以作为一个参考指南，帮助我们选择合适的存储驱动。同时还可以参阅 Docker 官网上由 Linux 发行商提供的最新文档来做出选择。
Red Hat Enterprise Linux：4.x版本内核或更高版本 + Docker 17.06 版本或更高版本，建议使用 Overlay2。
Red Hat Enterprise Linux：低版本内核或低版本的 Docker，建议使用 Device Mapper。
Ubuntu Linux：4.x 版本内核或更高版本，建议使用 Overlay2。
Ubuntu Linux：更早的版本建议使用 AUFS。
SUSE Linux Enterprise Server：Btrfs。

log_format为nginx设置日志格式
https://blog.csdn.net/wanchaopeng/article/details/93205621

MySQL报错 Error_code: 1045
http://blog.itpub.net/20893244/viewspace-2137789/
mysql主从同步
https://www.jianshu.com/p/80f30029cdf5

MySQL主从仅同步指定库
https://www.cnblogs.com/new-journey/p/11319527.html

有两种方式，1.在主库上指定主库二进制日志记录的库或忽略的库：

vim  /etc/my.cnf
    ...
    binlog-do-db=xxxx   二进制日志记录的数据库
    binlog-ignore-db=xxxx 二进制日志中忽略数据库
    以上任意指定其中一行参数就行，如果需要忽略多个库，则添加多行
    ...<br>重启mysql
 2.在从库上指定复制哪些库或者不负责哪些库

#编辑my.cnf，在mysqld字段添加如下内容：
  
replicate-do-db    设定需要复制的数据库
replicate-ignore-db 设定需要忽略的复制数据库
replicate-do-table  设定需要复制的表
replicate-ignore-table 设定需要忽略的复制表
replicate-wild-do-table 同replication-do-table功能一样，但是可以通配符
replicate-wild-ignore-table 同replication-ignore-table功能一样，但是可以加通配符
  
mysql主从同步 binlog-do-db replicate-do-db
https://blog.csdn.net/z69183787/article/details/70183284  

binlog_do_db是指定binlog日志记录那些库的二进制日志。replicate_do_db则在slave库中指定同步那些库的binlog日志。

而应该采用拷贝文件的方式，请按如下操作步骤：

先在主服务器上锁定所有的表，以免在复制过程中数据发生变化：

mysql> flush tables with read lock;

然后在主服务器上查询当前二进制文件的文件名及偏移位置：

mysql > show master status;

然后停止主服务器上的MySQL服务：

shell> mysqladmin -u root shutdown

再拷贝数据文件：

shell> tar -cvf /tmp/mysql-snapshot.tar .

拷贝完别忘了启动主服务上的MySQL服务了。

然后把数据文件应用到从服务器上，再次启动slave的时候使用，记得启动时加上skip-slave-start选项，使之不会立刻去连接master，再在从服务器上设置相关的二进制日志信息：

  
#修改后重启mysql

Mysql从库重建
https://blog.csdn.net/devotedwife/article/details/81915072

主库
mysqldump -u*** -p*** -h db-master --default-character-set=utf8 --master-data=2 --single-transaction --databases DB1 DB2 DB3  > product_data_backup_20170515.sql

注意–master-data=2 –single-transaction的配合使用，前一个参数会在开始导出时锁全表，记录当前的binlog文件和位置，然后释放锁，然后在同一个事务中导出数据，以保证一致性。

从库
stop slave;
从库先drop需要同步的数据库，然后source product_data_backup_20170515.sql导入数据；

不要修改从库的表结构以及数据，避免同步冲突失败；
不要滥用set global sql_slave_skip_counter,这会跳过某些同步sql，可能导致数据不一致；
附件类数据尽量不要直接存储在数据库中，备份和恢复时会特别慢

从未在从服务器上手动更新过数据，但还是可能遇到“Error: 1062 Duplicate entry”错误，具体原因不详，可能是MySQL本身的问题。遇到这类问题的时候，从服务器会停止复制操作，我们只能手动解决问题，具体的操作步骤如下：

mysql> set global sql_slave_skip_counter = 1;
mysql> start slave;

同样的操作可能需要进行多次，也可以设置自动处理此类操作，在从服务器的my.cnf里设置：

slave-skip-errors=1062

## mysql 主从同步

主节点

vi /etc/mysql/mysql.conf.d/mysqld.cnf

    [mysqld]
    log_bin=master60
    server_id=60
    binlog_format="mixed"
    bind-address            = 0.0.0.0

systemctl restart mysql
systemctl status mysql

mysqldump -uroot -p --default-character-set=utf8 --master-data=2 --single-transaction --databases db1 db2 db3  > backup.sql

mysql -uroot -p

    show master status\G;
    #授权给从服务器，单个数据库授权无效，必须设置*.*
    grant replication slave on *.* to repluser@'%' identified by '123456'; 
    show grants for 'repluser';

从节点

mysql -uroot -p
    stop slave;
    drop database db1;
    drop database db2;
    drop database db3;
    
# grep 'CHANGE MASTER TO MASTER_LOG_FILE' backup.sql
-- CHANGE MASTER TO MASTER_LOG_FILE='master60.000002', MASTER_LOG_POS=154;
    
mysql -uroot -p <backup.sql


    
mysql -uroot -p
    show databases;

vi /etc/mysql/mysql.conf.d/mysqld.cnf
    [mysqld]
    server_id=128
    replicate-do-db="db1"
    replicate-do-db="db2"
    replicate-do-db="db3"

systemctl restart mysqld
systemctl status mysql

mysql -uroot -p

    # master_log_file 与主库binlog日志名相同，master_log_pos 偏移量与主库相同
    change master to master_host='192.168.1.60', master_user='repluser', master_password='123456', master_log_file='master60.000002', master_log_pos=154;
    start slave;
    show slave status\G;


看到 Slave_IO_Running: Yes，Slave_SQL_Running: Yes 表示成功
如果 Slave_IO_Running 不为 Yes，可能是没连上主库，如下方式定为

tail -f /var/log/mysql/error.log
perror 1045	
mysql -urepluser -h 192.168.1.60 -p -P3306

Slave_SQL_Running 不为 yes 表示执行中继日志的 sql 命令时出错，需要停掉slave，在从库新建相关表,重启slave

测试：

主库

create table t1(id int );#
insert into t1 values(1);

从库

select * from t1;


MySQL 双主模式，两台机器互为主从，A 修改自动同步 B，B 修改自动同步 A。
为了防止并发引起的自增主键冲突，虽然双主，但不双写，平时只写 A。
当确认 A 宕机后，手工或自动把 vip 指向 B，或域名指向 B，或修改应用配置指向 B。
这样配置后，除了 A 切到 B 时会有短暂业务影响外，还有没有别的坑？


MySQL 双主问题集
https://www.cnblogs.com/GO-NO-1/p/10218304.html

MySQL双主一致性架构优化
https://www.jianshu.com/p/e2296eff932e

主库高可用，主库一致性，一些小技巧：

双主同步是一种常见的保证写库高可用的方式

设置相同步长，不同初始值，可以避免auto increment生成冲突主键

不依赖数据库，业务调用方自己生成全局唯一ID是一个好方法

shadow master保证写库高可用，只有一个写库提供服务，并不能完全保证一致性

内网DNS探测，可以实现在主库1出现问题后，延时一个时间，再进行主库切换，以保证数据一致性


请教下各位大佬，关于 mysql 的双主复制问题
https://www.v2ex.com/t/579370

不是 DBA，但是了解一些 msyql。
这个是比较野的多主方案吧。看样子是两个节点互为主从。这么用的比较少。

mysql 多主现在生产环境用的一般有 PXC/MGC，或者 MGR 方案。
pxc 和 MGC 其实同样的东西，都是 Percona 主导的技术。
MGR 是比较新的方案，没怎么了解。
应用层都不用重新设计。

多主一般强调一致性，同时在所有的节点做写入操作。保证所有的节点数据一致。
缺点明显，集群性能估计是单节点的 60%左右。如果有个节点性能差，会直接拖后腿。

mysql5.6配置semi_sync
https://www.cnblogs.com/caibird2005/p/4311544.html

mysql的replication协议是异步的，虽然异步效率、性能很好，但是却无法保证主从数据一致性,
如果master crash，已经commit的事务不会被传送到任何的slave上，
从mysql5.5之后，mysql为了保证主从库数据一致性，引进了semi-sync功能，
semi-sync意思是MASTER只需要接收到其中一台SLAVE的返回信息，就会commit；否则需等待直至切换成异步再提交。

优点：
当事务返回客户端成功后，则日志一定在至少两台主机上存在。
MySQL的Semi-sync适合小事务，且两台主机的延迟又较小，则Semi-sync可以实现在性能很小损失的情况下的零数据丢失。

缺点：
完成单个事务增加了额外的等待延迟，延迟的大小取决于网络的好坏。

PXC、MGR、MGC集群之新建、备份、恢复操作步骤 
http://www.sohu.com/a/339960973_610509

PXC的原理
https://www.cnblogs.com/zengkefu/p/5678279.html

即将开源的新一代MySQL高可用组件：MySQL Plus
https://blog.csdn.net/n88lpo/article/details/80015256


mysql 从库升级为主库的步骤
https://blog.csdn.net/weixin_33759269/article/details/92350851

1、进入主库，设置只读；

SET GLOBAL read_only=1;

2、进入从库，等同步完成后，暂停同步，并设置读写；


stop slave;
SET GLOBAL read_only=0;
reset slave all;

-- RESET SLAVE ALL是清除从库的同步复制信息、包括连接信息和二进制文件名、位置
-- 从库上执行这个命令后，使用show slave status将不会有输出。

3、修改配置文件连接到新的主库上。
人人都能看懂的LSTM
https://zhuanlan.zhihu.com/p/32085405

LSTM内部主要有三个阶段：

1. 忘记阶段。这个阶段主要是对上一个节点传进来的输入进行选择性忘记。简单来说就是会 “忘记不重要的，记住重要的”。
2. 选择记忆阶段。这个阶段将这个阶段的输入有选择性地进行“记忆”。
3. 输出阶段。这个阶段将决定哪些将会被当成当前状态的输出。

基于神经网络（LSTM）的股票数据分析
https://zhuanlan.zhihu.com/p/55841946

不同数据集，正确率和损失率有很大不同。在创业板300+只股票数据中，LSTM模型正确率可高达88.97%，损失率可以低至3.64%。在主板股票数据中，正确率为77.53%，损失率为7.97%。在中小板数据中，正确率为76.81%，损失率为8.63%。

正确率：预测结果正确数 / 测试数据总数 * 100%
损失率：（预测涨幅为正 and 实际涨幅为负） / 测试数据总数 * 100%

AI For Trading： LSTM模型预测三天后个股收盘价
https://zhuanlan.zhihu.com/p/94889284



规格型号：超融合计算节点：8台、华为/FusionCube HCI、18.9万元/台；虚拟化软件：1套、华为/FusionSphere、13.15万元/套；云管套件：1套、华为/ManageOne、17.65万元/套；等

华为FusionCube超融合基础设施融合计算、存储、网络、虚拟化、管理于一体，具有高性能、低时延和快速部署等特点，并内置华为自研分布式存储引擎，深度融合计算和存储，消除性能瓶颈，灵活扩容，支持业界主流数据库和业界主流虚拟化软件。

基于教育信息化2.0的指导，遵循大数据的智慧校园建设标准，通过大数据技术，打通学校“人、财、物”的基础数据，以学校基本对象（学生/教师/资产/科研成果/招生就业等）为业务数据基础，整合梳理多源日志、行为等数据构建学生画像、教师画像等，通过人工智能方法，结合高校场景，实现数据信息全面，过程可见，智能辅助，决策科学的目标，提升学校的教学、科研、人才培养、后勤服务等多方面需求。
1、业务大数据、日志大数据及有关社会大数据的采集和存储。对接校内各应用系统获取各类业务数据、异构系统设备的日志数据，结合社会大数据资源，采用大数据管理技术进行统一存储，为数据的挖掘和分析打好基础。
2、大数据分析应用，以学生画像为纽带，日志行为数据为入口，通过AI行为建模，挖掘和发现数据中隐含的、未知的、极具潜在应用价值的信息和规律，实现面向领导、面向老师、面向学生、面向网络等多角色的大数据应用需求的可视化交付，并且支持大屏、WEB、H5、分析报告等多样化的成果交付，为我校的教务管理、科研管理、学生管理等各项工作提供决策和指导。



云原生技术公开课：
https://edu.aliyun.com/roadmap/cloudnative
大数据学习路线
https://edu.aliyun.com/roadmap/bigdata
数据库学习路线
https://edu.aliyun.com/roadmap/database
人工智能学习路线
https://edu.aliyun.com/roadmap/ai

如何快速学习Tableau Desktop
https://www.jianshu.com/p/0adf8fea3351

Apache Superset(孵化)是一个现代的、企业级的商业智能web应用程序。
https://www.jianshu.com/p/4a1c213a8b1c

给大数据分析师的一双大礼: Apache Kylin和Superset
https://www.jianshu.com/p/828fa6f45ff3

阿里巴巴大数据竞赛
https://github.com/sunnotes/Ali-Data-Mining

Free Data Mining Tools
http://www.rdatamining.com/resources/tools

Top 15 Best Free Data Mining Tools: The Most Comprehensive List
https://www.softwaretestinghelp.com/data-mining-tools/
https://opensourceforu.com/2017/03/top-10-open-source-data-mining-tools/


CC中英字幕 - Weka在数据挖掘中的运用（Data Mining with Weka）
https://www.bilibili.com/video/av45489204?from=search&seid=13827735967963188412


哪些事情是weka能做的但是spss无法做到的？
https://www.zhihu.com/question/20985683/answer/16819027

初试weka数据挖掘
https://www.cnblogs.com/hxsyl/p/3307343.html


python数据挖掘orange
https://blog.csdn.net/pipisorry/article/details/52845804


完成 Orange3 数据挖掘 汉化版
https://blog.csdn.net/err2008/article/details/89000962

# Install some build requirements via your system's package manager
sudo apt install virtualenv build-essential python3-dev

# Create a separate Python environment for Orange and its dependencies ...
virtualenv --python=python3 --system-site-packages orange3venv
# ... and make it the active one
source orange3venv/bin/activate

# Install Qt dependencies for the GUI
pip install PyQt5 PyQtWebEngine

# Install Orange
pip install orange3

1.3.2 sklearn自带的小数据集
自带的小数据集

名称	数据包调用方式	适用算法
鸢尾花数据集	load_iris()	分类
乳腺癌数据集	load_bread_cancer()	二分类任务
手写数字数据集	load_digits()	分类
糖尿病数据集	load_diabetes()	回归
波士顿房价数据集	load_boston()	回归
体能训练数据集	load_linnerud()	多变量回归


Orange数据挖掘工具介绍
https://blog.csdn.net/SunChao3555/article/details/84975783

轻量级BI工具Superset的搭建与使用
https://www.jianshu.com/p/b02fcea7eb5b

开源数据挖掘工具Orange简介
https://blog.csdn.net/Tulongf/article/details/23992007

Orange,RapidMiner,Weka,JHepWork,KNIM,五个免费开源的数据挖掘软件
https://blog.csdn.net/bruce__ray/article/details/49699461

26种数据挖掘软件比较及介绍
https://blog.csdn.net/Tulongf/article/details/23994233

几经折腾，终于完成Orange3数据挖掘新版的汉化工作！需要合作的加Q：726008，Orange3群：681586766

Orange是一款底层基于C++，并且提供了Python接口的开源数据挖掘工具。与Sklearn，pyml这 类数据挖掘包相比，Orange的历史更加悠久，在上面实现的算法也更加丰富，此外，除了以python模块的形式使用之外，Orange还提供了GUI，可以用通过预先 定义好的多种模块组成工作流来完成复杂的数据挖掘工作。

Orange的发起最早可以追溯到1997年WebLab会议，在这个会议上人们提到了构建一个灵活的实验基准以便大家可以将自己的算法，实验结果放在上面，这些想法最终 催生了Orange项目。

Orange包含大量标准或非标准的机器学习和数据挖掘的算法，以及常规数据读写和操作，其底层的核心是由C++来实现的，同时Orange也利用python脚本来快速 实现新算法的原型以及一些不太要求执行时间的功能。


docker swarm 端口开放
The network ports required for a Docker Swarm to function properly are:

TCP port 2376 for secure Docker client communication. This port is required for Docker Machine to work. Docker Machine is used to orchestrate Docker hosts.
TCP port 2377. This port is used for communication between the nodes of a Docker Swarm or cluster. It only needs to be opened on manager nodes.
TCP and UDP port 7946 for communication among nodes (container network discovery).
UDP port 4789 for overlay network traffic (container ingress networking).

### 技能树：初级后台

C 基础

- 编写，编译，运行 hello world。
- 实现 atoi 函数。
- 复制一个文件。
- 对一个整型数组进行冒泡排序。
- 对一个整型数组进行二分查找。
- 用结构数组表示一个成绩列表｛姓名，科目，成绩｝，求出指定科目平均成绩最高的姓名。

Linux

- 安装系统
- 分区，格式化，挂载
- 配置动态、静态网络IP，DNS，网关，子网掩码
- 使用 top 查看系统运行情况
- 使用 ifconfig, ip 查看本机网络配置
- 使用 uname 查看本机信息，
- 添加新用户，加入组，切换用户
- 使用 ulimit 修改描述符最大限制
- touch，mkdir，cd，pwd，ls，cp, mv，rm 进行文件管理
- 熟悉文件查看命令：cat, head, more, less, tail
- 使用 find 查找文件
- 使用 grep 查找文本
- 使用 sort 排序命令输出
- 使用 diff 命令比较文件
- 使用 wc 统计输出行数
- chown 和 chmod 修改文件权限
- 开启 openssh 服务，指定允许远程登录的用户
- 指定拥有 sudo 权限的用户
- 编写简单 shell 脚本：if, while, $?
- 使用管道组合多个命令
- 对标注输入，标准输出，标准错误进行重定向
- 持续监控日志输出
- 使用 scp 和 rsync 在多台机器间同步文件
- 配置 ssh 免密登录
- 使用 netstat, ss 查看端口监听列表
- 使用 ps 查看进程列表
- 使用 ping, curl, nc 测试远程服务是否正常
- 使用 free 查看内存使用情况，定位内存占用大的进程
- 查看 CPU 使用情况，定位 CPU 使用高的进程
- 查看磁盘使用情况，定位磁盘占用大的目录
- 查看网络流量情况
- 使用包管理工具安装软件，修改源到国内镜像
- 启动，停止，禁用，启用某个服务
- 使用防火墙开放或关闭某个端口: ufw, iptables
- 使用 rc.local 设置开机自启动脚本
- 使用 crontab 设置定时任务
- 使用 vi 编辑配置文件
- 修改 .bashrc 里的环境变量，使用 alias 添加别名
- 使用 ln 建立文件软连接
- 使用 shell 进行数学运算
- 使用 tar，zip, unzip 解压，压缩文件
- 挂接，弹出 U 盘


git

- 配置密钥，创建仓库，克隆仓库，.git/config 改动
- git status，代码拉取，代码提交，代码推送，修改最后一次 commit 信息
- git rm， git mv, .gitignore 使用
- 代码对比：工作区对比，暂存区对比，分支间对比，历史对比
- 解决冲突：使用 A， 使用 B，两者合并
- 新建分支，合并分支，删除分支，删除远程分支，常见分支规划（dev, testing, master, hotfix, feat）
- 查看历史：查看历史改动，指定某文件改动，提取指定历史版本
- 撤销修改：撤销修改的文件，撤销暂存的文件，撤销已提交改动，撤销已推送改动
- 使用书架暂存: git stash
- git rebase -i
- 恢复文件：git reflog


HTML

- HTML 基本文档：doctype, lang, meta，head, body
- 引入 css, js
- 排版相关元素：h1-h5, p，a，img，br, hr，span，strong，code，pre，q
- 列表相关元素：ul, ol, dl，li，dd
- 布局相关元素: div，span，table
- 表格相关元素：table, tr, td, col
- 表单相关元素：form，input, select，textarea，button
- input 类型：text, password, checkbox, radio, hidden，color，date，time，number
- h5 验证属性：required ，max，min，minlength，maxlength
- 实现用户注册页面：姓名，年龄，职业，爱好，手机号


CSS

- 常见选择器：id, class，后代选择器，子选择器，属性选择器，伪类选择器
- 传统布局：position，display，width, height，margin, padding, border, overflow, float
- flex 布局：flex-flow， justify-content，align-items，flex，align-self
- 装饰类规则：字体，字号，行高，颜色，背景色，链接，列表，阴影，圆角
- 常见布局实现：图标和文字水平对齐，文字垂直水平居中，div 垂直水平居中，左右分栏（左定宽右弹性），左中右分栏（左右定宽，中间弹性），水平菜单，下拉菜单
- Bootstrap 使用：表格，表单，辅助类，小图标，按钮组，导航条，标签页，面包屑，分页，警告框，面板，轮播，模态框


PHP

- 变量，分支，循环，函数，字符串操作，数组操作，时间操作，文件操作，面向对象
- 理解 empty，is_null，isset
- 读取 get, post, cookie，header，文件上传等数据
- 设置 HTTP 应答码， header, cookie，输出 json，HTML，文件下载
- 数据校验，XSS 过滤，SQL 注入过滤
- 配置并使用 session
- 使用正则表达式 preg
- 访问数据库：mysqli，PDO
- 访问网络：curl
- 调试拍错：die，exit，var_dump, print_r，debug_print_backtrace ，error_log，file_put_contents, error_reporting
- 使用命名空间和自动加载，使用 composer 安装第三方组件
- 使用 MVC 框架：理解 model, view, controller, templeate，library，helper 的职责
- 使用 HMVC modules 进行模块化开发
- 配置 php.ini 和 php-fpm.conf


MySQL

- 增删改查，多表关联，分组统计
- 导入导出数据库或数据表
- 创建用户，指定权限
- 查看表结构，添加列，修改列
- 查看执行计划，添加有效索引


Nginx

- 配置 PHP-fpm
- 配置静态目录，启用文件索引，压缩，过期时间
- 配置反向代理，设置必要的转发头
- 配置虚拟目录为 php，反向代理或静态目录
- 配置 https 证书


Javascript

- 基本语法：分支，循环，函数，数组，hash
- DOM API，BOM API
- jquery 选择器，dom 操作，css 操作，事件操作
- jquery ajax
- underscore, async 使用


网络

- 常见 HTTP 方法：GET, PUT, POST, DELETE, OPTION
- 常见 HTTP 应答码：100, 200, 301, 302, 400, 401, 403, 404, 413，500
- 常见请求头：Accept，Accept-Encoding，Host，UserAgent，Cookie，Connection，Referer
- 常见应答头: Connection，Content-Type，Content-Length，Server, Transfer-Encoding，Set-Cookie
- 浏览器开发者工具网络标签
- 常用网络工具：ifconfig, ip, lsof, netstat,ss, ping, traceroute，mtr，telnet, curl, wget, nc，tcpdump, nslookup, dig, whois, nload, iftop
- 防火墙相关：iptables, ufw


问题定位

- 信息查看工具使用：uptime, top, htop, vmstat, iostat, sar
- 优雅重启服务：php-fpm, nginx，gunicorn, supervisord
- 系统日志查看
- crontab 日志查看
- Nginx 日志配置及查看
- PHP/fpm 日志配置及查看
- 应用服务日志查看
- MySQL 慢日志配置及查看


Linux常用网络工具总结
https://blog.csdn.net/li_101357/article/details/70256411

本文总结了Linux中的常用的网络工具，其中包括

网络配置相关：ifconfig、ip
路由相关：route、netstat、ip
查看端口工具：netstat、lsof、ss、nc、telnet
下载工具：curl、wget、axel
防火墙：iptables、ipset
流量相关：iftop、nethogs
连通性及响应速度：ping、traceroute、mtr、tracepath
域名相关：nslookup、dig、whois
web服务器：python、nginx
抓包相关：tcpdump
网桥相关：ip、brctl、ifconfig、ovs


图解HTTP（六）—— HTTP请求头（首部）
https://blog.csdn.net/alexshi5/article/details/80379086

Git鲜为人知的四个命令：bisect，blame，reflog和提交范围
https://baijiahao.baidu.com/s?id=1598885936030644678&wfr=spider&for=pc


TensorSpace是一套用于构建神经网络3D可视化应用的框架。 开发者可以使用 TensorSpace API，轻松创建可视化网络、加载神经网络模型并在浏览器中基于已加载的模型进行3D可交互呈现。 TensorSpace可以使您更直观地观察神经网络模型，并了解该模型是如何通过中间层 tensor 的运算来得出最终结果的。 TensorSpace 支持3D可视化经过适当预处理之后的 TensorFlow、Keras、TensorFlow.js 模型。
https://github.com/tensorspace-team/tensorspace/blob/master/README_zh.md


机器学习指机器通过统计学算法，对大量的历史数据进行学习从而生成经验模型，利用经验模型指导业务。目前机器学习主要在以下方面发挥作用：

营销类场景：商品推荐、用户群体画像、广告精准投放
金融类场景：贷款发放预测、金融风险控制、股票走势预测、黄金价格预测
SNS关系挖掘：微博粉丝领袖分析、社交关系链分析
文本类场景：新闻分类、关键词提取、文章摘要、文本内容分析
非结构化数据处理场景：图片分类、图片文本内容提取OCR
其它各类预测场景：降雨预测、足球比赛结果预测
笼统地讲，机器学习可以分为三类：

有监督学习（Supervised Learning）：指每个样本都有对应的期望值，通过模型搭建，完成从输入的特征向量到目标值的映射。典型的案例就是回归和分类问题。
无监督学习（Unsupervised Learning）：指在所有的样本中没有任何目标值，期望从数据本身发现一些潜在的规律，例如一些简单的聚类。
增强学习（Reinforcement Learning）：相对来说比较复杂，是指一个系统和外界环境不断地交互，获得外界反馈，然后决定自身的行为，达到长期目标的最优化。其中典型的案例就是阿法狗下围棋，或者无人驾驶。

PAI底层支持多种计算框架：有流式算法框架Flink，基于开源版本深度优化的深度学习框架TensorFlow，支持千亿特征千亿样本的大规模并行化计算框架Parameter Server，同时也兼容Spark、PYSpark、MapReduce等业内主流开源框架。

PAI平台提供：PAI-STUDIO（可视化建模和分布式训练）、PAI-DSW（notebook交互式AI研发）、PAI-AutoLearning（自动化建模）、PAI-EAS（在线预测服务）四套服务，每个服务既可单独使用，也可相互打通。用户可以从数据上传、数据预处理、特征工程、模型训练、模型评估，到最终的模型发布到离线或者在线环境，一站式完成建模，有效的提升开发效率。在数据预处理方面，PAI跟阿里云DataWorks（一站式大数据智能云研发平台）也是无缝打通的，支持SQL、UDF、UDAF、MR等多种数据处理开发方式，灵活性较高。在PAI平台上训练模型，生成的模型可以通过EAS部署到线上环境，整个实验流程支持周期性调度，可以发布到DataWorks与其它上下游任务节点打通依赖关系，另外调度任务区分生产环境以及开发环境，可以做到数据安全隔离。

一站式的机器学习平台意味着只要训练数据准备好（存放到OSS或MaxCompute中），用户就不需要额外的迁移工作，所有的建模工作都可以通过PAI来实现。

DataWorks
DataWorks是一个提供了大数据OS能力、并以all in one box的方式提供专业高效、安全可靠的一站式大数据智能云研发平台。 同时能满足用户对数据治理、质量管理需求，赋予用户对外提供数据服务的能力。


全生命周期数据应用开发
从数据开发到算法开发，从服务开发到应用开发，闭环涵盖数据业务全流程。


下一代大数据云研发平台
提供离线、实时、机器学习Studio满足大数据全业务场景。

完美支持数据中台
为全域数据汇聚与融合加工、数据治理与分享提供温床，助力企业完美升级数据体系。

全智能化体验
引入SQL智能编辑器、智能基线监控、数据质量监控、数据保护伞，赋能AI时代必备能力。

覆盖大数据全业务场景的功能体系


数据集成
供复杂网络环境下、丰富的异构数据源之间数据高速稳定的数据移动及同步能力。

 
    多数据源快速上云
    支持多库、多表整体数据上云的快捷配置。


    多种配置方式
    同时兼容可视化向导模式、复杂配置的脚本模式以及API模式创建数据集成任务。


    多种同步方式
    支持实时、历史数据的批量、增量同步，同步速度可以打满万兆网卡。


    任意数据源、任意网络环境数据抽取
    支持任意结构化、非结构化、半结构化的数据传输；同时可配置Agent至自有跳板机，实现对内网环境数据源的抽取与同步。

 
数据开发
构建项目->解决方案->业务流程三级结构，帮助用户获得更加清晰的开发逻辑。

 
    多引擎工作流混编
    以DataStudio为核心的多引擎混编工作流，串联跨引擎数据节点开发，每个类型的引擎数据节点都有对应的Studio进行开发。


    SQL智能编辑器
    提供SQL格式化、智能补齐、关键字高亮、错误提示、SQL内部结构等人性化功能，带来更顺滑的SQL开发体验。


    科学规范的项目模式
    提供开发、生产环境隔离的“标准项目模式”，将更稳定的生产环境带给用户。


    业务流程与解决方案
    从业务视角管理整体工作流，将同类业务组织为解决方案，实现沉浸式开发。

 
数据治理
保障数据定时产出、有效产出，让数据满足企业“存、通、用”的高标准数据管理要求。

 
    数据质量监控
    提供对多种异构数据源的质量校验、通知、管理能力。


    任务智能监控
    通过简单配置赋予智能监控系统自行决策“是否报警、何时报警、如何报警、给谁报警”的能力，以实现复杂工作流的全链路监控。

 
数据安全
提供可视化数据权限申请审批流程，并一些类诸如敏感数据分级、访问行为识别、数据脱敏、风险识别的数据审计能力。

 
    数据权限申请与审批
    开发者可在线批量发起数据权限申请，管理者酌情进行审批，实现流程可控与可视化，利于事后审计与追溯。


    敏感数据智能识别
    基于自学习的模型算法，自动识别企业拥有的敏感数据，并以直观的形式展示具体类型、分布、数量等信息；同时支持自定义类型的数据识别


    精准的数据分级分类
    支持自定义分级信息功能，满足不同企业对数据等级管理需要


    灵活的数据脱敏
    提供丰富多样、可配置的数据脱敏方式，无论是存储环节的静态脱敏，还是使用环节的动态脱敏


    用户异常操作风险监控和审计
    利用多维度关联分析及算法，主动发现异常风险操作，提供预警以及可视化一站式审计

 
数据服务
基于Serverless为企业搭建统一的数据服务总线，帮助企业统一管理对内对外的API服务。

 
    Serverless构建方式
    告别传统构建API的开发、运维流程，仅需关注API本身逻辑即可在web页面完成配置，并支持弹性扩展，运维成本为0。


    过滤器与函数
    灵活变换API返回结果数据结构，适配各类业务系统要求。


    服务编排
    支持将多个数据服务API串联为工作流，实现复杂业务请求逻辑。


    简单管理API生命周期
    基于web页面可完成API发布、管理、运维、售卖的全生命周期管理，助力用户简单、快速、低成本、低风险地实现微服务聚合、前后端分离、系统集成的工作。


    一键打通商业模式
    支持一键将API发布至阿里云市场进行售卖，直接将数据能力变现。

 
应用开发
实现在线Web轻量化开发能力，提供丰富的前端组件，通过自由拖拽即可简单快速搭建前端应用。

 
    托管Web应用开发
    无需下载安装本地IDE和配置维护环境变量，只需一个浏览器，即可在办公室、家或任何可以连接网络的地方，进行您的数据开发工作。


    功能完备的编辑器
    提供智能提示、补全代码并提供修复建议，让您轻松地编写、运行和调试项目。


    在线调试
    在线调试具有本地IDE所有的断点类型和断点操作，支持线程切换、过滤，支持变量值查看、监视，支持远程调试和热部署。


    协同编辑
    支持多人同时在线编辑同一个工程的同一个文件，提高工作效率。


    插件体系
    支持业务插件、工具插件和语言插件三种插件。

 
机器学习
阿里云机器学习平台（PAI）集数据处理、建模、离线预测、在线预测为一体，向用户提供更简易的操作体验。

 
    良好的交互设计
    通过对底层的分布式算法封装，提供拖拉拽的可视化操作环境，让数据挖掘的创建过程像搭积木一样简单。


    优质、丰富的机器学习算法
    提供经过阿里大规模业务锤炼而成的基础的聚类、回归类等算法与文本分析、特征处理等复杂算法。


    规格支持主流深度学习框架
    包含Tensorflow、Caffe、MXNet三款主流的机器学习框架，底层提供M40型号的GPU卡进行训练。

    
可视化模型

javascript workflow builder
    
10+ JavaScript libraries to draw your own diagrams (2019 edition)
https://modeling-languages.com/javascript-drawing-libraries-diagrams/    

https://jsplumbtoolkit.com/demos.html
https://gojs.net/latest/index.html

html5 javascript workflow diagram generator [closed]
https://stackoverflow.com/questions/20190581/html5-javascript-workflow-diagram-generator

jsplumb 中文基础教程
https://wdd.js.org/jsplumb-chinese-tutorial/#/

jsplumb实现流程图
https://www.jianshu.com/p/a3cd623cdbb7
https://github.com/wangduanduan/visual-ivr

开源HTML5拓扑图绘制工具？
https://www.zhihu.com/question/41026400

百度脑图核心——kityminder-editor 本地化改造
https://www.jianshu.com/p/9b53499d9031


Machine learning tools in JavaScript
https://github.com/mljs/ml
https://ml5js.org/

Top Machine Learning Libraries for Javascript 
https://www.kdnuggets.com/2016/06/top-machine-learning-libraries-javascript.html

orange3-web
https://github.com/biolab/orange3/issues/1419

Node-RED
Low-code programming for event-driven applications
https://nodered.org/

Business Intelligence (BI) in Python, OLAP
https://github.com/mining/mining

老师和学生都喜欢 Orange3
https://orange.biolab.si/home/teachers_and_students_love_it/


Browserify：浏览器加载Node.js模块
http://javascript.ruanyifeng.com/tool/browserify.html

browserify -r through -r ./my-file.js:my-module > bundle.js


计算矩阵的相关系数时，为什么提示说标准差是0
 
应该是有某一列的数字完全一样吧


b.reduce(function(pre, cur, i) { pre[a[i]] = cur; return pre }, {})

jsPlumb使用学习-在线流程设计器demo参考说明
https://blog.csdn.net/hexin8888/article/details/83992816

基于 vue 和 element-ui 实现的表单设计器，使用了最新的前端技术栈，内置了 i18n 国际化解决方案，可以让表单开发简单而高效。
https://github.com/GavinZhuLei/vue-form-making/blob/master/README.zh-CN.md

vue内引入jsPlumb流程控制器（一）
https://blog.csdn.net/weixin_30872671/article/details/94822539

Graphlib JS 学习笔记
https://blog.csdn.net/qq_32773935/article/details/81236461

Div Background Image Z-Index Issue
https://stackoverflow.com/questions/10507871/div-background-image-z-index-issue

发现一个 CSS BUG，一个正常position 的 div a 无论如何设置 z-index，都不能在一个 position: relative 并且设置了 background-image 的div b 上面。
b 去掉 background-image 就可以了，或者 a 得设置 position: relative，怪不得 css 难掌握，这 z-index 有时候毫无作用呀。

webpack
https://www.jianshu.com/p/2ce732125376

npm install=npm i。在git clone项目的时候，项目文件中并没有 node_modules文件夹，项目的依赖文件可能很大。直接执行，npm会根据package.json配置文件中的依赖配置下载安装。
-global=-g，全局安装，安装后的包位于系统预设目录下
--save=-S，安装的包将写入package.json里面的dependencies，dependencies：生产环境需要依赖的库
--save-dev=-D，安装的包将写入packege.json里面的devDependencies，devdependencies：只有开发环境下需要依赖的库


webpack 引入 bootstrap(一)
https://www.cnblogs.com/wyxxj/p/7381050.html


Bootstrap4与Bootstrap3不同
https://blog.csdn.net/drl_blogs/article/details/89305729

webpack-dev-server，iframe与inline的区别
https://www.cnblogs.com/videring/articles/7641555.html

深入解析webpack-dev-server的用法
https://www.jianshu.com/p/bbb55217d124


Managing jQuery plugin dependency in webpack
https://stackoverflow.com/questions/28969861/managing-jquery-plugin-dependency-in-webpack

vue引入bootstrap——webpack
https://blog.csdn.net/wild46cat/article/details/77662555