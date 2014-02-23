if !has('python')
    echo "Error: Required vim compiled with +python"
    finish
endif

function! Reddit()
 
python << EOF
try:
    import vim
    del vim.current.buffer[:]
    vim.current.buffer[0] = 80*"-"
    vim.current.buffer.append("Helloword")
    vim.current.buffer.append(80*"-")
except Exception, e:
    print e
 
EOF
endfunction
command! -nargs=0 Reddit call Reddit()
