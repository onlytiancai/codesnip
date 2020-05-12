<template>
    <div>
        X 轴：
          <select  v-model="xAxis"  @change="drawLine()" >
            <option v-for="(item,index) in fields" :key="index" :value='index'>{{item}}</option>  
       </select>  
        Y 轴：
          <select  v-model="yAxis"  @change="drawLine()" >
            <option v-for="(item,index) in fields" :key="index" :value='index'>{{item}}</option>  
       </select>  
    </div>
</template>
<script>
import Vue from "vue";
import echarts from "echarts";
Vue.prototype.$echarts = echarts;
var colorarrays = [
  "#2F9323",
  "#D9B63A",
  "#2E2AA4",
  "#9F2E61",
  "#4D670C",
  "#BF675F",
  "#1F814A",
  "#357F88",
  "#673509",
  "#310937",
  "#1B9637",
  "#F7393C"
];
export default {
  name: "hello",
  data() {},
  mounted() {
    this.drawLine();
  },
  methods: {
    drawLine() {
      const { dataset, fields, classes, classLabels, xAxis, yAxis } = this;
      const main = document.getElementById("main");

      let myChart = this.$echarts.init(main);

      const option = {
        xAxis: {},
        yAxis: {},
        series: [
          {
            data: dataset.map(function(x) {
              return [x[xAxis], x[yAxis]];
            }),
            type: "scatter",
            itemStyle: {
              normal: {
                color: function(e) {
                  const c = classes[e.dataIndex];
                  const colorIndex = classLabels.indexOf(c);
                  return colorarrays[colorIndex];
                }
              }
            }
          }
        ]
      };

      myChart.setOption(option);
    }
  }
};
</script>

