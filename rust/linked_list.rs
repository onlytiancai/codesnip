fn main() {
    // 链表底层存储，长度 100 的数组
    // 每个元素是一个(char, u32) tuple 类型
    // .0 存储数据, 为 ' ' 表示空闲节点
    // .1 指向下一个节点， 为 0 表示节点末尾
    let mut arr = [(' ', 0);100];

    // 构建字符链表
    let s = String::from("我是中国人");
    for (i, item) in s.chars().enumerate() {
        // TODO：寻找空闲节点索引
        arr[i] = (item, i+1);
    }

    // 遍历链表并打印
    let mut node = arr[0];
    while node.1 != 0 {
        print!("{}", node.0);
        node = arr[node.1];
        if node.1 == 0 {
            println!("{}", node.0);
        }
    }
}
