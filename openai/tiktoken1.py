import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
print(len(encoding.encode("一亩地租金1000元，那么3平方米地的租金应该是多少呢？首先需要将1亩转换为平方米，1亩=666.67平方米（约等于），因此每平方米的租金为：1000/666.67≈1.5元/平方米。\n\n那么3平方米的租金就是3×1.5=4.5元。")))