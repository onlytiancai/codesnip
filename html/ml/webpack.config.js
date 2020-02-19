const path = require('path')
const webpack = require('webpack')
const VueLoaderPlugin = require('vue-loader/lib/plugin');

module.exports = {
  // 内嵌调试
  devtool: 'eval-source-map',
  // 入口
  entry: path.join(__dirname, './src/main.js'),

  //出口
  output: {
    path: path.join(__dirname, './dist'),
    publicPath: "/assets/",
    filename: 'bundle.js'
  },

  //模式
  mode: 'development',
  resolve: {
    alias: {
      'vue': 'vue/dist/vue.js'
    }
  },
  module: {
    rules: [
      {
        test: /\.vue$/,
        loader: 'vue-loader',
      },
      {
        test: /\.css$/,
        loaders: ["style-loader", "css-loader"]
      },
      {
        test: /\.(png|jpe?g|gif|svg)(\?.*)?$/,
        loader: 'url-loader',
      },
      {
        test: /\.(woff2?|eot|ttf|otf)(\?.*)?$/,
        loader: 'url-loader',
      },
      // 配置js的loader
      {
        test: /\.js$/,
        use: {
          // 打包编译ES6语法
          loader: "babel-loader",
          options: {
            presets: [
              // 用于将高级语法转换为低级语法
              "@babel/preset-env"
            ]
          }
        },
        exclude: /node_modules/,
      }
    ]
  },
  plugins: [
    new VueLoaderPlugin(),
    new webpack.ProvidePlugin({
      $: "jquery",
      jQuery: "jquery",
      "window.jQuery": "jquery",
      "window.$": "jquery",
    })
  ]
}