生成 PDF

    pandoc test.md --pdf-engine=xelatex --toc --toc-depth=3 -V mainfont="PingFang SC" -o output.pdf

使用模板

    mkdir -p ~/.pandoc/templates/
    git clone https://github.com/Wandmalfarbe/pandoc-latex-template.git
    cp ~/src/pandoc-latex-template/template-multi-file/* ~/.pandoc/templates
    pandoc test.md --pdf-engine=xelatex --toc --toc-depth=3 -V mainfont="PingFang SC" --template eisvogel -o output.pdf
