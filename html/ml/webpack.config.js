const path = require('path')
const webpack = require('webpack')

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
  module: {
    rules: [
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
      }
    ]
  },
  plugins: [
    new webpack.ProvidePlugin({
      $: "jquery",
      jQuery: "jquery",
      "window.jQuery": "jquery",
      "window.$": "jquery",   
    })
  ]
}