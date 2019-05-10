requirejs.config({
    baseUrl: 'js/lib',
    paths: {
        vue: 'https://cdn.staticfile.org/vue/2.4.2/vue.min',
        vue_resource: 'https://cdn.staticfile.org/vue-resource/1.5.1/vue-resource.min',
        jquery: 'https://cdn.staticfile.org/jquery/3.4.1/jquery.min',
        bootstrap: 'https://cdn.staticfile.org/twitter-bootstrap/3.4.1/js/bootstrap.min',
    },
    "shim": {
        "bootstrap": ["jquery"]
    }
});