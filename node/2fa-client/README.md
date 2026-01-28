部署

    vi .env
    VITE_BASE_PATH=/2fa/

    pnpm build      
    rsync -avP dist/ ihuhao:/home/ubuntu/src/html/2fa

开发

    pnpm dev:all

本地测试子目录部署

    pnpm build
    cd dist
    mkdir 2fa
    mv assets 2fa    
    mv index.html 2fa
    mv vite.svg 2fa  
    npx serve        