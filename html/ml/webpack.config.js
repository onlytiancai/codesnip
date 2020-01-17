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
        publicPath:"/assets/",
        filename: 'bundle.js'
    },

    //模式
    mode: 'development',
    module: {
        rules: [
          {
            test: /\.css$/,
            loaders: ["style-loader","css-loader"]
          },
          {
            test: /\.(jpe?g|png|gif)$/i,
            loader:"file-loader",
            options:{
              name:'[name].[ext]',
              outputPath:'assets/images/'
              //the images will be emited to dist/assets/images/ folder
            }
          }
        ]
      },
      plugins: [
        /* Use the ProvidePlugin constructor to inject jquery implicit globals */
        new webpack.ProvidePlugin({
            $: "jquery",
            jQuery: "jquery",
            "window.jQuery": "jquery'",
            "window.$": "jquery"
        })
      ]
}