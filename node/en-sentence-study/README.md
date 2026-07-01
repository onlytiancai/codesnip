生成视频

    cd scripts/video
    pnpm exec tsx scripts/render.ts scripts/output/5.json

检查 json 文件

    参考 scripts/output/README.md 检查 scripts/output/7.json

生成相关文件，预览视频

    cd scripts/video
    pnpm exec tsx scripts/render.ts scripts/output/7.json --phase desc      # 只生成 desc JSON
    pnpm exec tsx scripts/render.ts scripts/output/7.json --phase assets    # 只生成场景插图
    pnpm exec tsx scripts/render.ts scripts/output/7.json --phase audio     # 只生成所有卡片音频

    sed -i '' 's/[0-9]*\.draft\.json/8.draft.json/' src/Root.tsx
    pnpm studio

测试

    pnpm exec remotion studio --props "$(jq -c '{desc: .}' scripts/desc/8.draft.json)"

    pnpm exec remotion render src/index.ts EnSentenceVideo out/test.mp4 --props "$(jq -c '{desc: .}' scripts/desc/8.draft.json)"