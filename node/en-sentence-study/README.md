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
    pnpm studio --props scripts/desc/7.draft.json