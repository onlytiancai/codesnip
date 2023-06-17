jupyter
```
jupyter notebook --ip=0.0.0.0 --no-browser
jupyter --path
vi ~/.local/share/jupyter
jupyter --data-dir
>>> from jupyter_core.paths import jupyter_path
>>> print(jupyter_path('nbconvert'))
['/home/action/.local/share/jupyter/nbconvert', '/usr/local/share/jupyter/nbconvert', '/usr/share/jupyter/nbconvert']
ls /usr/local/lib/python3.6/dist-packages/nbconvert/templates/html/
basic.tpl  celltags.tpl  full.tpl  mathjax.tpl  slides_reveal.tpl
cp /usr/local/lib/python3.6/dist-packages/nbconvert/templates/html/* /home/action/.local/share/jupyter/nbconvert/templates/html/
vi /home/action/.local/share/jupyter/nbconvert/templates/html/basic.tpl
cp /usr/local/lib/python3.6/dist-packages/nbconvert/templates/skeleton/display_priority.tpl /home/action/.local/share/jupyter/nbconvert/templates/html/
cp /usr/local/lib/python3.6/dist-packages/nbconvert/templates/skeleton//null.tpl /home/action/.local/share/jupyter/nbconvert/templates/html/
jupyter nbconvert --to html ./编程.ipynb

```


编程

```
内容：
1. 数字四则混合运算，表达式树，循环，判断
2. 列表求最大，最小，排序，过滤，转换，累加
3. 字典，word count
6. 函数，抽象，组合
7. 元组，表格，增添改查

模式：
提问，回答，挑战

问题
- 20以内能被3整除的数的总和
- 3x-4=20，求x
- 求圆周率，求函数的导数
- 开根号，求质数
- 多元方程求解，线性规划
```

git
```
git config --global core.quotepath false
```
