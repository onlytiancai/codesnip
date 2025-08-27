'''
本脚本演示了如何使用自定义的Transformer模型（wawa_transformer.Transformer）实现一个简单的序列到序列（seq2seq）任务：将数字序列映射为字母序列。
主要流程包括：
1. 数据准备：构造训练数据，将数字序列（src）映射为带有起始符（<SOS>）和终止符（<EOS>）的字母序列（tgt）。
2. 掩码函数：实现填充掩码（pad mask）和目标掩码（target mask），用于Transformer的注意力机制。
3. 模型定义与训练配置：设置Transformer模型参数、损失函数、优化器和训练轮数。
4. 训练过程：逐条训练样本进行前向传播、损失计算和参数更新。
5. 推理与测试：实现改进的翻译函数，并对多个测试用例进行验证。

【训练与推理流程简述】
- 训练时，模型输入src和tgt_input（tgt去掉最后一个token），目标为tgt_label（tgt去掉第一个token）。
- 推理时，从<SOS>开始逐步生成下一个token，直到生成<EOS>或达到最大长度。

【测试用例说明】
- 测试用例涵盖不同长度的输入序列，验证模型能否准确输出对应的字母序列（带<SOS>和<EOS>）。
'''
import torch
import torch.optim as optim
import torch.nn as nn
from wawa_transformer import Transformer
import numpy as np

# -------------------------- 数据准备 --------------------------
# 生成训练数据：数字序列 -> 字母序列（1->A, 2->B, ..., 5->E）
# 输入格式：[1,2,3] -> 输出格式：[< SOS >,A,B,C,<EOS>] （< SOS >=6, <EOS>=7）

# 大幅增加训练数据，包含各种长度的序列
train_data = [
    # 长度为1的序列
    (torch.tensor([1]), torch.tensor([6, 1, 7])),
    (torch.tensor([2]), torch.tensor([6, 2, 7])),
    (torch.tensor([3]), torch.tensor([6, 3, 7])),
    (torch.tensor([4]), torch.tensor([6, 4, 7])),
    (torch.tensor([5]), torch.tensor([6, 5, 7])),
    
    # 长度为2的序列（重点加强，因为测试用例是长度2）
    (torch.tensor([1, 2]), torch.tensor([6, 1, 2, 7])),
    (torch.tensor([1, 3]), torch.tensor([6, 1, 3, 7])),
    (torch.tensor([1, 4]), torch.tensor([6, 1, 4, 7])),
    (torch.tensor([1, 5]), torch.tensor([6, 1, 5, 7])),
    (torch.tensor([2, 3]), torch.tensor([6, 2, 3, 7])),
    (torch.tensor([2, 4]), torch.tensor([6, 2, 4, 7])),
    (torch.tensor([2, 5]), torch.tensor([6, 2, 5, 7])),
    (torch.tensor([3, 4]), torch.tensor([6, 3, 4, 7])),
    (torch.tensor([3, 5]), torch.tensor([6, 3, 5, 7])),
    (torch.tensor([4, 5]), torch.tensor([6, 4, 5, 7])),
    (torch.tensor([5, 1]), torch.tensor([6, 5, 1, 7])),
    (torch.tensor([4, 2]), torch.tensor([6, 4, 2, 7])),
    (torch.tensor([5, 3]), torch.tensor([6, 5, 3, 7])),
    
    # 长度为3的序列
    (torch.tensor([1, 2, 3]), torch.tensor([6, 1, 2, 3, 7])),
    (torch.tensor([2, 4, 5]), torch.tensor([6, 2, 4, 5, 7])),
    (torch.tensor([1, 3, 5]), torch.tensor([6, 1, 3, 5, 7])),
    (torch.tensor([2, 3, 4]), torch.tensor([6, 2, 3, 4, 7])),
    (torch.tensor([3, 4, 5]), torch.tensor([6, 3, 4, 5, 7])),
    (torch.tensor([5, 4, 3]), torch.tensor([6, 5, 4, 3, 7])),
    
    # 长度为4的序列
    (torch.tensor([1, 2, 3, 4]), torch.tensor([6, 1, 2, 3, 4, 7])),
    (torch.tensor([2, 3, 4, 5]), torch.tensor([6, 2, 3, 4, 5, 7])),
    (torch.tensor([5, 4, 3, 2]), torch.tensor([6, 5, 4, 3, 2, 7])),
    (torch.tensor([1, 3, 5, 2]), torch.tensor([6, 1, 3, 5, 2, 7])),
]

# 词汇表大小（源：6种，目标：8种）
# 源词表大小：1~5表示数字，0为PAD（共6种）
src_vocab_size = 6
# 目标词表大小：1~5表示字母A~E，6为<SOS>，7为<EOS>，0为PAD（共8种）
tgt_vocab_size = 8

# -------------------------- 掩码工具函数 --------------------------
def create_pad_mask(seq, pad_idx=0):
    """
    生成填充掩码（padding mask），用于屏蔽输入序列中的PAD位置，常用于Transformer等序列模型的注意力机制中。

    参数:
        seq (Tensor): 输入的序列张量，形状通常为(batch_size, seq_len)，其中包含实际数据和PAD填充值。
        pad_idx (int, 可选): PAD标记的索引，默认为0。

    返回:
        Tensor: 填充掩码，形状为(batch_size, 1, 1, seq_len)。PAD位置为False，其余为True。

    使用场景:
        在自然语言处理任务中，输入序列长度不一，需通过PAD补齐到相同长度。模型在计算注意力时，需要屏蔽PAD位置，避免其对结果产生影响。

    实现原理:
        通过(seq != pad_idx)生成一个布尔张量，PAD位置为False，其余为True。随后通过unsqueeze扩展维度，使其适配多头注意力机制的掩码需求。
    """
    """生成填充掩码（屏蔽PAD位置）"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # 形状：(batch_size, 1, 1, seq_len)

def create_target_mask(seq):
    """
    生成目标序列掩码（target mask），用于Transformer解码器中的自回归（auto-regressive）任务，防止模型在预测当前位置时访问未来的信息。

    作用：
        - 在训练或推理序列生成任务（如机器翻译、文本生成）时，确保解码器只能关注当前位置及之前的位置，避免信息泄露。
        - 结合填充掩码（pad mask），同时屏蔽无效的填充位置。

    使用场景：
        - 典型应用于Transformer结构的解码器部分，尤其是在自回归生成任务中，如机器翻译、文本摘要、对话生成等。

    实现原理：
        - 首先通过上三角矩阵生成一个掩码，保证每个位置只能看到当前位置及之前的位置（即防止“看见未来”）。
        - 通过unsqueeze调整掩码维度以适配多维输入（如batch和多头注意力）。
        - 最后与填充掩码结合，确保填充位置也被屏蔽。
    """
    """生成目标序列掩码（防止解码器关注未来信息）"""
    batch_size, seq_len = seq.size()
    # 上三角矩阵（对角线及以下为1，以上为0）
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 形状：(1, 1, seq_len, seq_len)
    return mask & create_pad_mask(seq)  # 结合填充掩码

# -------------------------- 训练配置 --------------------------
# -------------------------- 模型定义与参数说明 --------------------------
# Transformer模型的核心参数及其调整建议：
# - src_vocab_size: 源语言词表大小。应等于输入序列中可能出现的所有token种类数（包括PAD）。
# - tgt_vocab_size: 目标语言词表大小。应等于输出序列中所有token种类数（包括PAD、SOS、EOS）。
# - d_model: 模型的隐藏层维度（embedding和所有子层的输出维度）。数值越大，模型表达能力越强，但显存和计算量也越大。常见取值64/128/256/512。小任务可用128或256，大任务用512或更高。
# - num_layers: 编码器和解码器堆叠的层数。层数越多，模型越深，拟合能力更强，但训练更慢且易过拟合。小数据集建议2~4层，大数据集可用6层及以上。
# - num_heads: 多头自注意力的头数。头数越多，模型可关注的信息子空间越多，但每个头的维度会变小（d_model需能被num_heads整除）。常用4/8/16。一般d_model/num_heads >= 32。
# - d_ff: 前馈网络的中间层维度。通常为d_model的2~4倍。增大d_ff可提升模型容量，但会增加参数量和计算量。
# 参数调整建议：
#   - 数据量小、任务简单时，建议适当减小d_model、num_layers、num_heads、d_ff，防止过拟合。
#   - 数据量大、任务复杂时，可适当增大上述参数，但需注意显存和训练时间。
#   - 若训练不收敛，可适当减小学习率或增加训练轮数。
model = Transformer(
    src_vocab_size=src_vocab_size,   # 源词表大小（输入token种类数，含PAD）
    tgt_vocab_size=tgt_vocab_size,   # 目标词表大小（输出token种类数，含PAD、SOS、EOS）
    d_model=128,    # 模型隐藏层维度（表达能力，越大越强，消耗显存和算力也越大）
    num_layers=3,   # 编码器/解码器层数（层数越多模型越深，拟合能力更强，易过拟合）
    num_heads=4,    # 多头注意力头数（需能整除d_model，头数多可关注不同子空间）
    d_ff=256        # 前馈网络维度（通常为d_model的2~4倍，提升模型容量）
)
# -------------------------- 损失函数、优化器与训练轮数配置 --------------------------
# criterion: 损失函数。这里用CrossEntropyLoss（交叉熵损失），常用于分类/序列生成任务。
#   - ignore_index=0 表示忽略PAD（填充）位置的损失，避免无效填充影响训练。
#   - 可选项：也可用nn.NLLLoss（需log_softmax输出），或自定义损失函数。
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD的损失

# optimizer: 优化器。Adam是一种自适应学习率优化器，收敛快且稳定，适合Transformer等深度模型。
#   - lr=0.0005 设置学习率，值越大收敛越快但易不稳定，越小收敛慢但更稳。
#   - 可选项：SGD、AdamW、RMSprop等。AdamW更适合大模型和正则化。
#   - 调整建议：如训练不收敛可适当减小学习率；如收敛太慢可适当增大。
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 降低学习率

# epochs: 训练轮数。轮数越多，模型拟合能力越强，但过多可能过拟合。
#   - 可选项：根据数据量和收敛情况调整。小数据集可用200~500，大数据集需更多。
#   - 调整建议：如损失下降缓慢可增加轮数；如过拟合可减少轮数或加正则化。
epochs = 300  # 增加训练轮次

# -------------------------- 训练过程 --------------------------
print("开始训练，总训练样本数:", len(train_data))
for epoch in range(epochs):
    # 设置模型为训练模式
    model.train()
    # 累计损失初始化
    total_loss = 0

    for src, tgt in train_data:
        '''
        - src（源序列）：输入的数字序列，范围为1~5，0为PAD。例如：
            - torch.tensor([1, 2])         # 表示数字序列1,2
            - torch.tensor([3])            # 表示数字3
            - torch.tensor([4, 5, 1])      # 表示数字4,5,1
        - tgt（目标序列）：输出的字母序列，1~5分别对应A~E，6为<SOS>（起始符），7为<EOS>（终止符），0为PAD。例如：
            - torch.tensor([6, 1, 2, 7])   # <SOS>, A, B, <EOS>
            - torch.tensor([6, 3, 7])      # <SOS>, C, <EOS>
            - torch.tensor([6, 4, 5, 1, 7])# <SOS>, D, E, A, <EOS>

        【示例】
        - 输入src: torch.tensor([2, 3])
        输出tgt: torch.tensor([6, 2, 3, 7])   # <SOS>, B, C, <EOS>
        - 输入src: torch.tensor([5])
        输出tgt: torch.tensor([6, 5, 7])      # <SOS>, E, <EOS>
        - 输入src: torch.tensor([1, 4, 5])
        输出tgt: torch.tensor([6, 1, 4, 5, 7])# <SOS>, A, D, E, <EOS>

        '''
        # 处理输入：增加batch维度（batch_size=1）
        # 为什么要unsqueeze(0)？
        # Transformer模型的输入通常要求有batch维度，形状为(batch_size, seq_len)。
        # 训练时每次只用一个样本（batch_size=1），原始src和tgt是一维张量（seq_len,）。
        # 通过unsqueeze(0)在最前面加一维，变成(1, seq_len)，以适配模型输入。
        src = src.unsqueeze(0)  # 形状：(1, src_seq_len)
        
        # 为什么要把tgt分离成tgt_input和tgt_label？
        # Transformer的解码器采用自回归（auto-regressive）训练方式：
        # - tgt_input：作为解码器输入，去掉最后一个<EOS>，即 [<SOS>, A, B, C]
        # - tgt_label：作为训练目标，去掉第一个<SOS>，即 [A, B, C, <EOS>]
        # 这样模型每一步只能看到前面的token，预测下一个token，符合生成式任务的因果性。
        # 例如：tgt = [<SOS>, A, B, C, <EOS>]
        #   tgt_input = [<SOS>, A, B, C]
        #   tgt_label = [A, B, C, <EOS>]
        # 训练时，模型输入src和tgt_input，输出与tgt_label对齐，计算交叉熵损失。
        # 这种处理方式可防止模型“偷看”未来的token，保证训练和推理一致性。
        tgt_input = tgt[:-1].unsqueeze(0)  # 目标输入（去掉最后一个EOS，加batch维度）
        tgt_label = tgt[1:].unsqueeze(0)   # 目标标签（去掉第一个SOS，加batch维度）
        
        # 生成掩码
        src_mask = create_pad_mask(src) # 源序列的填充掩码
        tgt_mask = create_target_mask(tgt_input) #目标序列的自回归掩码（结合填充和未来信息屏蔽）
        cross_mask = src_mask  # 编码器-解码器掩码，通常与src_mask相同
        
        # 前向传播
        
        # 为什么要传入这些参数？
        # - src: 源序列（形状：[batch_size, src_seq_len]），即输入的数字序列。
        # - tgt_input: 目标输入序列（形状：[batch_size, tgt_seq_len]），即去掉<EOS>的目标序列，作为解码器输入。
        # - src_mask: 源序列的填充掩码（形状：[batch_size, 1, 1, src_seq_len]），用于屏蔽PAD位置，防止注意力关注无效填充。
        # - tgt_mask: 目标序列的自回归掩码（形状：[batch_size, 1, tgt_seq_len, tgt_seq_len]），用于防止解码器看到未来信息，同时屏蔽PAD。
        # - cross_mask: 编码器-解码器掩码（通常与src_mask相同），用于解码器在cross-attention时屏蔽源序列中的PAD位置。
        # 这些参数共同保证Transformer模型在训练时只关注有效信息，防止信息泄露和无效填充干扰，符合序列到序列任务的需求。

        # output 是模型前向传播的结果，通常是对目标序列每个位置的预测分布（如概率分布或 logits）。
        # 具体来说，如果 model 是一个序列到序列（seq2seq）模型（如 Transformer），
        # output 形状通常为 [batch_size, tgt_seq_len, vocab_size]，
        # 表示每个目标序列位置上对词表中每个词的预测分数。

        output = model(src, tgt_input, src_mask, tgt_mask, cross_mask)
        
        # 计算损失
        # ----------- 损失计算 -----------
        # 这里使用交叉熵损失（CrossEntropyLoss）来衡量模型输出与真实标签之间的差距。
        # output: 模型的输出张量，形状为 (batch_size, tgt_seq_len, tgt_vocab_size)，
        #   表示每个目标序列位置上对词表中每个词的预测分数（未归一化的logits）。
        # tgt_label: 真实的目标标签，形状为 (batch_size, tgt_seq_len)，
        #   每个位置是目标词表中的索引（如1~5为A~E，7为<EOS>，0为PAD）。
        #
        # 为什么要用 .contiguous().view(-1, ...) 展平？
        # - PyTorch的CrossEntropyLoss要求输入形状为 (N, C)，N为样本数，C为类别数；
        #   标签形状为 (N,)，每个元素是类别索引。
        # - 这里batch_size通常为1，但tgt_seq_len可能大于1，所以需要把(batch_size, tgt_seq_len)展平成一维。
        # - output.contiguous().view(-1, tgt_vocab_size)：把输出展平成 (batch_size * tgt_seq_len, tgt_vocab_size)
        # - tgt_label.contiguous().view(-1)：把标签展平成 (batch_size * tgt_seq_len,)
        # - 这样每个目标位置都作为一个独立的分类任务，计算损失后再求平均。
        #
        # ignore_index=0 的作用：
        # - 由于有PAD（填充）位置，PAD的标签为0，不应参与损失计算。
        # - ignore_index=0会自动忽略这些位置，避免无效填充影响训练。
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),  # 展平为(batch*seq_len, vocab_size)
            tgt_label.contiguous().view(-1)                # 展平为(batch*seq_len,)
        )
        
        # 反向传播
        # ----------- 优化器梯度清零 -----------
        # 为什么要optimizer.zero_grad()？
        # 在PyTorch中，每次调用loss.backward()时，梯度会累加（而不是自动清零）。
        # 如果不先清零，梯度会叠加导致参数更新出错。
        # 所以每次反向传播前都要先把梯度清零，确保每个batch的梯度是独立计算的。
        optimizer.zero_grad()

        # ----------- 反向传播计算梯度 -----------
        # loss.backward()会自动计算损失函数对模型所有参数的梯度（即∂loss/∂参数）。
        # 这一步是神经网络训练的核心，利用链式法则自动求导，得到每个参数应该如何调整以减小损失。
        # 计算结果会存储在每个参数的.grad属性中，供优化器使用。
        loss.backward()

        # ----------- 优化器更新参数 -----------
        # optimizer.step()会根据刚才计算得到的梯度，自动调整模型的参数。
        # 以Adam优化器为例，它会结合梯度和历史信息，智能地更新每个参数，使损失函数尽量减小。
        # 这一步完成后，模型的参数就“学到”了一点点如何更好地完成任务。
        optimizer.step()
        
        total_loss += loss.item()
    
    # 每20轮打印一次损失
    '''
    为什么要这样计算平均损失（avg_loss = total_loss / len(train_data））？
    在每个训练周期（epoch）结束时，我们通常希望监控模型的训练进展。由于每个epoch会遍历所有训练样本，并对每个样本分别计算损失（loss），
    因此需要对这些损失进行汇总和归一化，以便更直观地反映模型在整个训练集上的表现。
    具体原因如下：
    1. 归一化损失，便于比较：
        - 直接累加所有样本的损失（total_loss）会随着训练集样本数量的变化而变化，无法反映单个样本的平均表现。
        - 通过除以样本数（len(train_data)），得到每个样本的平均损失（avg_loss），这样不同epoch之间、不同数据集之间的损失值具有可比性。
    2. 监控训练过程，判断收敛情况：
        - 平均损失能够反映模型整体的拟合能力，便于观察损失是否随训练轮数下降，从而判断模型是否收敛。
        - 如果平均损失长期不下降，说明模型可能未学到有效特征；如果过低，可能出现过拟合。
    3. 便于调参和早停：
        - 训练过程中可以根据平均损失的变化趋势，动态调整学习率、训练轮数等超参数，或采用早停（early stopping）策略防止过拟合。
    4. 与批量训练（batch training）一致：
        - 即使采用批量训练（每次处理多个样本），也通常会对每个batch的损失取平均，再对所有batch的平均损失取平均，保持一致性。
    总结：
    通过 avg_loss = total_loss / len(train_data) 计算平均损失，可以更科学、直观地评估模型在整个训练集上的表现，便于监控训练进展和调优模型。
    '''    
    if (epoch + 1) % 20 == 0:
        
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# -------------------------- 改进的测试推理函数 --------------------------
def translate_improved(src_seq, max_length=None):
    """改进的翻译函数，支持长度控制"""
    model.eval()
    with torch.no_grad():
        src = src_seq.unsqueeze(0)  # 增加batch维度
        src_mask = create_pad_mask(src)
        
        # 初始化目标序列（从< SOS >开始）
        tgt_seq = torch.tensor([[6]], dtype=torch.long)  # 6是< SOS >
        
        # 如果没有指定最大长度，则根据源序列长度+2（SOS和EOS）
        if max_length is None:
            max_length = len(src_seq) + 2
        
        # 生成序列
        for i in range(max_length - 1):  # -1因为已经有了SOS
            tgt_mask = create_target_mask(tgt_seq)
            output = model(src, tgt_seq, src_mask, tgt_mask, src_mask)
            
            # 获取最后一个位置的预测概率
            logits = output[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
            
            # 如果生成EOS，停止
            if next_token.item() == 7:
                break
        
        return tgt_seq.squeeze(0).tolist()  # 去除batch维度

# -------------------------- 测试多个用例 --------------------------
def test_multiple_cases():
    """测试多个用例"""
    test_cases = [
        torch.tensor([3, 4]),    # 主要测试用例
        torch.tensor([1, 2]),    # 简单用例
        torch.tensor([5]),       # 单个数字
        torch.tensor([2, 3, 4]), # 长序列
    ]
    
    # 映射为字母（1→A,2→B,3→C,4→D,5→E,6→SOS,7→EOS）
    char_map = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'< SOS >', 7:'<EOS>'}
    
    for i, test_src in enumerate(test_cases):
        predicted_tgt = translate_improved(test_src)
        predicted_chars = [char_map[token] for token in predicted_tgt]
        
        print(f"\n测试用例 {i+1}:")
        print(f"输入（数字）：{test_src.tolist()}")
        print(f"模型输出（字母）：{predicted_chars}")
        
        # 显示预期输出
        expected = ['< SOS >'] + [char_map[x.item()] for x in test_src] + ['<EOS>']
        print(f"预期输出：{expected}")
        print(f"是否正确：{'✓' if predicted_chars == expected else '✗'}")

# 运行测试
test_multiple_cases()
