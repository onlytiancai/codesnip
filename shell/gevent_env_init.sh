appname="myapp"
python_url="http://www.python.org/ftp/python/2.7.2/Python-2.7.2.tgz"
easy_install_url="http://peak.telecommunity.com/dist/ez_setup.py"
pip_url="https://raw.github.com/pypa/pip/master/contrib/get-pip.py"
python_path="/usr/local/${appname}/python"
python_cmd=${python_path}"/bin/python"
easy_install_cmd=${python_path}"/bin/easy_install"
pip_cmd=${python_path}"/bin/pip"
virtualenv_cmd=${python_path}"/bin/virtualenv"
env_path=~/.${appname}/
env_python=${env_path}"/bin/virtualenv"
env_pip=${env_path}"/bin/pip"
CPUCOUNT=`cat /proc/cpuinfo | grep processor | wc -l`

function install_if_notfound(){
    if [ ! -n `rpm -qa | grep {$1}` ]; then
        echo $1 not found
        zypper -n install $1
    else
        echo $1 found.
    fi
}

function install_python_deppend(){
    install_if_notfound gcc
    install_if_notfound make
    install_if_notfound sqlite-devel 
    install_if_notfound readline-devel 
    install_if_notfound zlib-devel 
    install_if_notfound openssl-devel 
}
install_python_deppend
function install_python(){
    if [ ! -e ${python_path} ];then
        echo ${python_path} not found
        if [ ! -e Python*.tgz ];then
            wget ${python_url}
        fi
        tar xf Python*.tgz
        cd Python*
        ./configure --prefix=${python_path}
        make -j$CPUCOUNT
        make install
        cd ..
    else
        echo ${python_path} found.
    fi 
}
install_python
function install_virtualenv(){
    if [ ! -e ${easy_install_cmd} ];then
        curl -O ${easy_install_url} 
        ${python_cmd} ez_setup.py
    else
        echo easy_install found.
    fi
    if [ ! -e ${pip_cmd} ];then
        curl -O ${pip_url} 
        ${python_cmd} get-pip.py
    else
        echo pip found.
    fi
    if [ ! -e ${virtualenv_cmd} ];then
        ${pip_cmd} install virtualenv
    else
        echo virtualenv found.
    fi
}
install_virtualenv
function create_env(){
    if [ ! -e ${env_path} ];then
        echo create virtual env 
        ${virtualenv_cmd} --no-site-packages ~/.$appname
    else
        echo ${env_path} found.
    fi
}
create_env

function install_to_env(){
    if [ -z "`find ${env_path} -name "*$1*"`" ];then
        ${env_pip} install $1
    else
        echo $1 found.
    fi
}
install_to_env greenlet
install_to_env gevent
install_to_env web.py
