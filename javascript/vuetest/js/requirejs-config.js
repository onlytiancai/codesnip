requirejs.config({
    baseUrl: 'js/lib',
    paths: {
        Vue: 'https://cdn.staticfile.org/vue/2.4.2/vue.min',
        VueResource: 'https://cdn.staticfile.org/vue-resource/1.5.1/vue-resource.min',
        VueRoute: 'https://cdn.staticfile.org/vue-router/3.0.6/vue-router.min',
        jquery: 'https://cdn.staticfile.org/jquery/3.4.1/jquery.min',
        Bootstrap: 'https://cdn.staticfile.org/twitter-bootstrap/3.4.1/js/bootstrap.min',
    },
    "shim": {
        "Bootstrap": ["jquery"]
    }
});