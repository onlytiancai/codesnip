#!/usr/bin/env bash

shopt -s extglob
set -o errtrace
set -o errexit


LINUX_BIN_NAME="logsend"
DARWIN_BIN_NAME="logsend_darwin"
REPO="https://github.com/ezotrank/logsend"
TMP_DIR="/tmp"
INSTALL_DIR="/usr/local/bin"

download(){
  latest_release_tag=$(curl https://api.github.com/repos/ezotrank/logsend/releases|grep 'tag_name'|head -n1|awk '{print $2}'|sed -e 's/,//g'|sed -e 's/"//g')
  if [ "$(uname)" = "Darwin" ]; then
    os_package=$DARWIN_BIN_NAME
  else
    os_package=$LINUX_BIN_NAME
  fi
  url="$REPO/releases/download/$latest_release_tag/$os_package.gz"
  curl -L $url > "$TMP_DIR/logsend.gz" && (cd $TMP_DIR && gunzip -f logsend.gz)
}

install(){
  if [ -n "$1" ]; then
    INSTALL_DIR=$1
  fi
  cp -f $TMP_DIR/logsend $INSTALL_DIR && chmod 755 $INSTALL_DIR/logsend
  echo "Logsend installed to $INSTALL_DIR/logsend"
}

logsend_install(){
  download
  install $1
}

logsend_install $@
