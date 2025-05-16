// Markdown面板功能
document.addEventListener('DOMContentLoaded', function() {
    // 默认的格言
    const defaultQuote = `> "专注是成功的关键，而休息是持久的秘诀。"
- 保持专注，一次只做一件事
`;

    // 初始化Markdown面板
    function initMarkdownPanel() {
        const markdownPanel = document.getElementById('markdown-panel');
        if (!markdownPanel) return;
        
        // 获取保存的内容或使用默认内容
        const savedContent = localStorage.getItem('markdownContent') || defaultQuote;
        
        // 创建面板内容
        const panelHeader = document.createElement('div');
        panelHeader.className = 'markdown-panel-header';
        
        const panelTitle = document.createElement('h2');
        panelTitle.className = 'markdown-panel-title';
        panelTitle.textContent = '个人便签';
        
        const editButton = document.createElement('button');
        editButton.className = 'markdown-panel-edit';
        editButton.innerHTML = '✏️';
        editButton.title = '编辑内容';
        
        panelHeader.appendChild(panelTitle);
        panelHeader.appendChild(editButton);
        
        const markdownContent = document.createElement('div');
        markdownContent.className = 'markdown-content';
        markdownContent.innerHTML = marked.parse(savedContent);
        
        const markdownEditor = document.createElement('div');
        markdownEditor.className = 'markdown-editor';
        
        const textarea = document.createElement('textarea');
        textarea.value = savedContent;
        textarea.placeholder = '在这里输入Markdown格式的文本...';
        
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'markdown-editor-buttons';
        
        const saveButton = document.createElement('button');
        saveButton.className = 'btn';
        saveButton.textContent = '保存';
        
        const cancelButton = document.createElement('button');
        cancelButton.className = 'btn';
        cancelButton.textContent = '取消';
        
        buttonContainer.appendChild(cancelButton);
        buttonContainer.appendChild(saveButton);
        
        markdownEditor.appendChild(textarea);
        markdownEditor.appendChild(buttonContainer);
        
        markdownPanel.appendChild(panelHeader);
        markdownPanel.appendChild(markdownContent);
        markdownPanel.appendChild(markdownEditor);
        
        // 添加事件监听器
        editButton.addEventListener('click', () => {
            markdownEditor.classList.add('active');
            markdownContent.style.display = 'none';
        });
        
        cancelButton.addEventListener('click', () => {
            markdownEditor.classList.remove('active');
            markdownContent.style.display = 'block';
            textarea.value = localStorage.getItem('markdownContent') || defaultQuote;
        });
        
        saveButton.addEventListener('click', () => {
            const content = textarea.value;
            localStorage.setItem('markdownContent', content);
            markdownContent.innerHTML = marked.parse(content);
            markdownEditor.classList.remove('active');
            markdownContent.style.display = 'block';
        });
    }
    

        initMarkdownPanel();

});