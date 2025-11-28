# other
pip install huggingface_hub
HF_ENDPOINT=https://hf-mirror.com hf download bigscience/mt0-large --local-dir ./mt0-large

export HF_ENDPOINT=https://hf-mirror.com
pip install -q peft transformers datasets

Transformers 快速入门
https://huggingface.co/docs/transformers/quicktour

peft notebooks
https://huggingface.co/spaces/PEFT/causal-language-modeling

因果模型（Causal language modeling）微调教程
https://huggingface.co/docs/transformers/tasks/language_modeling

因果模型（Causal language modeling）微调 notebooks
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb

微调相关：
- Fine-Tuning Large Language Models with LoRA: A Practical Guide https://www.christianmenz.ch/programming/fine-tuning-large-language-models-with-lora-a-practical-guide/?utm_source=chatgpt.com
- In-depth guide to fine-tuning LLMs with LoRA and QLoRA https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora?utm_source=chatgpt.com
- Efficient Fine-tuning with PEFT and LoRA https://heidloff.net/article/efficient-fine-tuning-lora/?utm_source=chatgpt.com 
    - https://www.philschmid.de/fine-tune-flan-t5-peft
    - https://huggingface.co/blog/4bit-transformers-bitsandbytes
- Fine-Tuning Large Language Models (LLMs) https://towardsdatascience.com/fine-tuning-large-language-models-llms-23473d763b91/
- Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA https://huggingface.co/blog/4bit-transformers-bitsandbytes

## 摘录

RAG 和 fine tune
- 使用 RAG 时，我们不会对模型进行任何操作，它就是原始模型，数据只是随请求一起传递。而微调则不同，它使用额外的领域特定数据重新训练模型，因此我们会更改模型权重，最终得到一个包含领域特定知识的新模型。
- 你可以创建两个低秩矩阵，然后将它们相乘得到一个大矩阵。
- 微调后的模型可以通过提供定制化功能，从多方面提升业务绩效。微调模型有助于您根据自身需求和知识进行定制。
- 您可以使用 RAG 管道来定制模型，但有时知识量过于庞大，仅靠嵌入和相似性搜索是不够的，这时就需要通过微调进行定制。
- 微调不仅可以提高基础模型的性能，而且较小的（微调后的）模型通常在其训练的任务集上优于较大的（更昂贵的）模型 
- 一个主要问题是，LLM 的上下文窗口有限。因此，对于需要庞大知识库或领域特定信息的任务，模型的性能可能欠佳[1]。微调模型可以通过在微调过程中“学习”这些信息来避免这个问题。
- 微调模型通常有三种方法 ：自监督学习、监督学习和强化学习。

QLoRA

- QLoRA 使用 4 位量化来压缩预训练语言模型。然后冻结语言模型的参数，并将数量相对较少的可训练参数以低秩适配器的形式添加到模型中。
- 在微调过程中，QLoRA 通过冻结的 4 位量化预训练语言模型将梯度反向传播到低秩适配器。训练过程中，只有 LoRA 层会更新参数。
- QLoRA 拥有两种数据类型：一种是用于存储基础模型权重的存储数据类型（通常为 4 位 NormalFloat），另一种是用于执行计算的计算数据类型（16 位 BrainFloat）。
- QLoRA 将存储数据类型中的权重解压缩为计算数据类型，以执行前向和反向传播，但仅计算使用 16 位 bfloat 的 LoRA 参数的权重梯度。权重仅在需要时才进行解压缩，因此在训练和推理过程中内存占用保持较低水平。
- load a model in 4bit using NF4 quantization below with double quantization with the compute dtype bfloat16 for faster training:
- A rule of thumb is: use double quant if you have problems with memory, use NF4 for higher precision, and use a 16-bit dtype for faster finetuning. 
