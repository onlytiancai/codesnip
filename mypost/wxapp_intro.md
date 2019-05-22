# 官方零基础入门教程

- [小程序开发指南](https://developers.weixin.qq.com/ebook?action=get_post_info&docid=0008aeea9a8978ab0086a685851c0a)

---

# 小程序优势

- 快速的加载
- 更强大的能力
- 原生的体验
- 易用且安全的微信数据开放
- 高效和简单的开发

---

# 起步

- 申请账号
	- 需要邮箱
	- 获取到 appid
	- 如果只体验不发布，可用试用帐号
- 安装开发工具
	- 文件树、代码区、预览区、调试区
	- 扫描预览默认的 QuickStart 项目

---

# 代码构成

- JSON 配置
	- app.json，project.config.json，page.json
- WXML 界面
	- 新标签：page，view, block, text
	- 数据绑定语法 {{}}，自定义属性 wx:if, wx:for 
	- 自定义组件
- WXSS 样式
	- iphonx 6 上 1rpx = 0.5px = 1物理像素
	- app.wxss，page.wxss
- JS 逻辑	
	- es6
	- Page({data: {}, onLoad(){}})
	- wx.xxx API

---
# 开发流程

- 整体的产品体验有一个清晰的规划和定义
- 交互图或者手稿描绘小程序的界面交互
- 完成WXML+WXSS还原设计稿
- 梳理出 data 部分，填充 WXML 的模板语法
- 完成JS逻辑部分

---
# Flex布局

- 传统网页开发，我们用的是盒模型，通过 display:inline | block | inline-block、 position、float来实现布局，缺乏灵活性且有些适配效果难以实现。
- flex的概念最早是在2009年被提出，目的是提供一种更灵活的布局模型，方便适配不同大小的内容区域。

- .container: flex-direction，flex-wrap，justify-content，align-items，align-content 
- .item：order，flex-shrink，flex-grow，flex-basis，flex，align-self 
- [示例](https://developers.weixin.qq.com/ebook?action=get_post_info&docid=00080e799303986b0086e605f5680a)