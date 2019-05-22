Page({
  data: {
    TodayList: [],
    Today: "",
    input: ""
  },

  onLoad: function() {
    var d = new Date();
    this.setData({
      Today: d.getFullYear() + '-' + (d.getMonth() + 1) + '-' + d.getDate() 
    });

    this.loadData();
  },



  save: function() {
    wx.setStorageSync('TodayList', this.data.TodayList);
    this.loadData();
  },

  loadData: function() {
    var todo = wx.getStorageSync('TodayList');
    if (todo) {
      this.setData({
        TodayList: todo,
        completedCount: todo.filter(function(x){return x.completed}).length,
      });
    }
  },
  
  // 实时获取用户输入文字
  addInput: function (e) {
    this.setData({
      input: e.detail.value
    });
  },
  
  // 回车后添加任务
  addTodo: function(e) {
    // 添加一条数据
    this.data.TodayList.unshift({
      description: this.data.input,
      completed: false
    });

    // 保存并刷新
    this.save();

    // 清空输入框
    this.setData({
      input: ''
    });
  },

  //删除任务
  removeTodo: function(e) {
    var index = e.currentTarget.id;
    this.data.TodayList.splice(index, 1);
    this.save();
  },

  //改变任务状态
  toggleTodo: function(e) {
    var index = e.currentTarget.id;
    var todo = this.data.TodayList;
    todo[index].completed = !todo[index].completed;
    this.save();
  },
})