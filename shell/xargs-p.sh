fun() {
    echo func: $1, $2
}

# 需要导出，才能在 xargs 中使用
export -f fun 

seq -f "%04g" 0 $1 | xargs -P 10 -I{} bash -c 'fun "$0" "$1"' {} $2
