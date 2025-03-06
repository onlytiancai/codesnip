import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env['ARK_API_KEY'],
  baseURL: 'https://ark.cn-beijing.volces.com/api/v3',
});

const args = process.argv;
if (!args[2] || args[2].trim() === '') {
    console.error('第二个参数（通常是脚本路径）不存在或者为空，程序退出。');
    process.exit(1);
}

const sentence = args[2];
console.log('sentence', sentence)

const prompt = `
你是一位英语专家，负责为用户提供的英语句子进行语法分析。
假设用户提供的英语句子是：The man who lives next door is my teacher.

请在<中文翻译>标签内将英语句子翻译成中文，如：
<中文翻译>
住在隔壁的人是我的老师。
</中文翻译>

请在<单词列表>标签内解释句子中的重点单词，如：
<单词列表>
- live: v.  /lɪv; laɪv/ 生存，活着；生活在；
- teacher: n. /ˈtiːtʃər/ 教师
</单词列表>

请按照以下步骤进行语法分析：
1. 用多层括号和缩进在<句子成分>标签的代码块内清晰表示出从句和修饰关系。例如：
<句子成分>
The man 
    (who lives next door) 
is my teacher.
</句子成分>

2. 详细介绍句子中的每个成分，包括主语、谓语、宾语、定语、状语、补语等，说明它们在句子中的作用和语法功能。
请在在<句子成分介绍>标签内详细介绍句子的各个成分，如：

<句子成分介绍>
- **The man**：在主句中作主语，表示句子所描述动作或状态的执行者或主体，这里指特定的那个男人。
- **who lives next door**：是一个定语从句，修饰先行词“the man”。
    - 其中“who”是关系代词，在定语从句中作主语，用来引导定语从句并指代先行词“the man”；
    - “lives”是定语从句中的谓语动词，表示“居住”的动作；
    - “next door”是地点状语，说明居住的地点，它在定语从句中修饰“lives”，表明居住的位置是隔壁。
- **is**：是主句中的谓语动词，属于系动词，起到连接主语和表语的作用，构成主系表结构。
- **my teacher**：在主句中作表语，用于说明主语“the man”的身份，即他是“我的老师”。
</句子成分介绍>
请确保分析准确、全面，对句子成分的介绍清晰易懂。
如果用户询问英语语法无关的问题，拒绝回答。
如果用户发送的文字不是英语，请提示用户重新输入。

请仔细阅读以下英语句子，按照要求进行详细分析。
英语句子：
<english_sentence>
${sentence}
</english_sentence>
`

async function main() {
  // Streaming:
  console.log('----- streaming request -----')
  const stream = await openai.chat.completions.create({
    messages: [
      { role: 'user', content: prompt },
    ],
    model: 'doubao-1-5-pro-32k-250115',
    stream: true,
  });
  for await (const part of stream) {
    process.stdout.write(part.choices[0]?.delta?.content || '');
  }
  process.stdout.write('\n');
}

main();
