"=============================================================================
"基本配置
"http://edyfox.codecarver.org/html/_vimrc_for_beginners.html
"=============================================================================

set nocp "不再VI兼容模式下运行

" Tab related
set ts=4 "tab宽度为4个字符
set sw=4 "自动缩进为4个空格
set smarttab "按tab键插入4个空格
set et "编辑时将tab替换成空格
set ambiwidth=double"防止特殊字符无法显示

" Format related
"set tw=78 "光标超过78列这行
set lbr "不再单词中间断行
set fo+=mB "打开断行模块对亚洲语言的支持
set nu "打开行号
" Indent related
set cin "打开c风格的自动缩进
set ai "打开普通文件类型的缩进
set cino=:0g0t0(susj1 "设置c缩进风格的选项

" Editing related
set backspace=indent,eol,start "退格键可删除缩进，上一行和之前插入的字符
set whichwrap=b,s,<,>,[,] "方向导航键可以在多行间进行
set mouse=c "支持鼠标
set selectmode= "不是用selectmode
set mousemodel=popup "右键单机窗口，弹出快捷菜单
set keymodel= "不使用shift+方向键选择文本
set selection=inclusive "选择文本时包含光标所在位置

" Misc
set wildmenu "命令模式下按tab弹出菜单
"set spell "打开拼写检查

" Encoding related
set encoding=utf-8 "当前字符编码为UTF-8
set langmenu=zh_CN.UTF-8 "使用中文菜单并使用utf-8编码
set fileencoding=utf-8
language message zh_CN.UTF-8 "使用中文提示信息且utf-8
"编码的自动识别
set fileencodings=ucs-bom,utf-8,cp936,gb18030,big5,euc-jp,euc-kr,latin1

" File type related
filetype plugin indent on "开启文件类型自动识别，启用文件类型插件及自动缩进

" Display related
set ru "打开状态栏标尺
set sm "显示括号配对情况
set hls "高亮显示被搜索的字符
if (has("gui_running"))
    set guioptions+=b "界面模式下添加水平滚动条
    colo torte "设置颜色方案
    set nowrap "设置不自动换行
else
    colo ron "设置颜色方案
    "set wrap "设置终端模式下自动换行
endif
syntax on "启用语法提示

if (has("win32"))
    if (has("gui_running"))
        set guifont=Consolas:h11:cANSI
        set guifontwide=NSimSun:h9:cGB2312
    endif
else

    if (has("gui_running"))
        set guifont=Bitstream\ Vera\ Sans\ Mono\ 11
    endif
endif
"=============================================================================
"个性化配置
"=============================================================================
if(has("win32"))
    autocmd! bufwritepost *_vimrc source $VIM\_vimrc "vimrc保存后自动加载
else
    autocmd! bufwritepost *.vimrc source ~\.vimrc "vimrc保存后自动加载
endif

"设置自动补全
set completeopt=longest,menu
"=======括号自动补全
:inoremap ( ()<ESC>i
:inoremap ) <c-r>=ClosePair(')')<CR>
:inoremap { {}<ESC>i
:inoremap } <c-r>=ClosePair('}')<CR>
:inoremap [ []<ESC>i
:inoremap ] <c-r>=ClosePair(']')<CR>

function! ClosePair(char)
    if getline('.')[col('.') - 1] == a:char
        return "\<Right>"
    else
        return a:char
    endif
endf

"打开javascript折叠
let b:javascript_fold=1
"打开javascript对dom,html和css的支持
let javascript_enable_domhtmlcss=1

"设置','为leader快捷键
let mapleader = ","
let g:mapleader = ","

"设置快速保存和退出
"快速保存为,s
"快速退出（保存）为,w
"快速退出（不保存）为,q
nmap <leader>s :w!<cr>
nmap <leader>w :wq!<cr>
nmap <leader>q :q!<cr>

"nmap <C-t>   :tabnew<cr>
nmap <C-p>   :tabprevious<cr>
nmap <C-n>   :tabnext<cr>
nmap <C-k>   :tabclose<cr>
nmap <C-Tab> :tabnext<cr>

"au GUIENTER * simalt ~x "打开窗口最大化
set guioptions-=m
set guioptions-=T

"=============================================================================
"字典设置
"=============================================================================

"加载javascript字典
au FileType javascript set dict +=$HOME.'/.vim/dict/javascript.dict'
au FileType css set dict+=$HOME.'/.vim/dict/css.dict'

"=======自动加载php函数
au FileType php call AddPHPFuncList()
function! AddPHPFuncList()
    set dictionary-=$HOME.'/.vim/dict/php_funclist.txt'
                \dictionary+=$HOME.'/.vim/dict/php_funclist.txt'
    set complete-=k complete+=k
endfunction

"=============================================================================
"插件设置
"SuperTab,SnipMate,NeoComplCache,NERDTree,Taglist,BufExplorer,Template
"=============================================================================

"=======SuperTab
let g:SuperTabRetainCompletionType=2 "记住上次补全方式，直到推出插入模式
let g:SuperTabDefaultCompletionType="<C-X><C-O>" "设置按tab后默认补全方式
"======NeoComplCache
let g:neocomplcache_enable_at_startup = 1 "自动启动
let g:neocomplcache_enable_smart_case = 1 "智能匹配大小写
let g:neocomplcache_enable_camel_case_completion = 1 "启用首字母大写匹配
let g:neocomplcache_disable_auto_complete = 0

autocmd FileType css setlocal omnifunc=csscomplete#CompleteCSS 
autocmd FileType html,markdown setlocal omnifunc=htmlcomplete#CompleteTags 
autocmd FileType javascript setlocal omnifunc=javascriptcomplete#CompleteJS 
autocmd FileType python setlocal omnifunc=pythoncomplete#Complete 
autocmd FileType xml setlocal omnifunc=xmlcomplete#CompleteTags 

"======NERDTree
map <F8> :NERDTreeToggle<CR> "f10切换树目录

"======Taglist
let Tlist_Ctags_Cmd = '/usr/bin/ctags' "ctag的命令
let Tlist_Show_One_File = 1 "不同时显示多个文件的tag，只显示当前文件的
let Tlist_Exit_OnlyWindow = 1 "如果taglist窗口是最后一个窗口，则退出vim
let Tlist_Use_Right_Window = 1 "在右侧窗口中显示taglist窗口 
map <silent> <F9> :TlistToggle<cr> "f9切换taglist窗口

"======BufExplorer
nmap <f2>  :BufExplorer<cr> "f2显示buff列表

"======pydoc,pyflakes,pydiction
let g:pydiction_location = $HOME.'/.vim/ftplugin/pydiction/complete-dict'
let g:pydiction_menu_height = 20
let g:pydoc_cmd = "pydoc"
map <S-F2> :!pyflakes %<cr>

" NERD_commenter
map <c-h> ,c<space>

"set cursorcolumn
"set cursorline

vmap <leader>tc :Align = #<cr>

" https://github.com/tpope/vim-pathogen
call pathogen#infect()

" https://github.com/nvie/vim-flake8
let g:flake8_ignore="E501,W293,E701"
let g:flake8_max_complexity=10

" 79标尺
set colorcolumn=79
