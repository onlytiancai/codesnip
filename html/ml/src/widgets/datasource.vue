<template>
  <div>
    <h3>数据集名称：{{name}} </h3>
    数据行数：{{rows}}
    <table class="table table-striped">
        <tr>
          <th>#</th>
          <th>列名</th>
        </tr>
      <tr v-for="(item,i) in cols"><td>{{i}}</td><td>{{item}}</td></tr>  
    </table>
  </div>
</template>

<script>
require("bootstrap/dist/css/bootstrap.min.css");
// TODO：为啥引用 bootstrap3 没有 CSS 效果
//require('bootstrap3/dist/css/bootstrap.min.css')
export default {
  data: function() {
    return {
      count: 0
    };
  }
};
</script>