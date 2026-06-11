上传

    cat ~/.cos.conf
    cd /Volumes/data/coscdn
    coscmd upload -rs --delete ./onnx-community /
    coscmd upload -rs --delete ./libs /
    curl -I http://cdn.ihuhao.com/libs/chart.js/4.4.1/chart.umd.min.js