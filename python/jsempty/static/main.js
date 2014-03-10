seajs.config({
    alias: {
        "$": "jquery/1.8.3/jquery-debug",
        "jquery": "jquery/1.8.3/jquery-debug",
        "underscore": "underscore/1.4.2/underscore-debug",
        "backbone": "backbone/0.9.2/backbone-debug",
        "mustache": "mustache/0.5.0/mustache-debug",
        "cookie": "cookie/1.0.2/cookie-debug",
        "moment": "moment/1.7.2/moment-debug"
    },
    preload: ['seajs/plugin-text']
});

seajs.use(['./static/dnspod.js' ]);
