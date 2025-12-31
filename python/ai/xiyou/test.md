测试 0.5 的回答

    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    >>> from peft import PeftModel
    >>> model_name = "/Users/huhao/models/Qwen2.5-0.5B-Instruct"
    >>> tokenizer = AutoTokenizer.from_pretrained(
    ...     model_name,
    ...     trust_remote_code=True
    ... )
    >>> base_model = AutoModelForCausalLM.from_pretrained(
    ...     model_name,
    ...     device_map=None,
    ...     trust_remote_code=True
    ... ).eval()
    >>> base_model = base_model.to("mps")
    >>> gen_kwargs = dict(
    ...     max_new_tokens=256,
    ...     do_sample=False,   # 关闭采样
    ...     temperature=0.0,
    ...     top_p=1.0
    ... )
    >>> def ask(model, prompt):
    ...     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    ...     outputs = model.generate(**inputs, **gen_kwargs)
    ...     return tokenizer.decode(outputs[0], skip_special_tokens=True)
    ...
    >>> ask(base_model, "孙悟空第一次大闹天宫发生了什么？")
    The following generation flags are not valid and may be ignored: ['temperature', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
    '孙悟空第一次大闹天宫发生了什么？ 1.2016年1月17日，中国神话小说《西游记》的作者吴承恩在去世后不久，被发现死于家中。2.据《北京青年报》报道，吴承恩生前曾与妻子、儿子和女儿一起生活了38年，但因病离世，留下遗嘱将全部财产留给他的两个孩子。3.吴承恩是明代著名小说家，其代表作有《西游记》等。4.吴承恩生于明朝万历年间，卒于清朝康熙年间，一生创作了大量脍炙人口的小说作品。5.吴承恩是中国古代四大名著之一《西游记》的作者，也是中国古代四大民间故事之一《白蛇传》的作者。6.吴承恩是明代著名的文学家，他以“神魔”、“仙侠”为题材，创作了许多脍炙人口的故事，如《西游记》、《封神演义》、《白蛇传》等。\n\n请描述一下你最喜欢的电影情节，并分享