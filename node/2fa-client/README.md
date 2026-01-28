部署

    vi .env
    VITE_BASE_PATH=/2fa/

    pnpm build      
    rsync -avP dist/ ihuhao:/home/ubuntu/src/html/2fa

开发

    pnpm dev:all