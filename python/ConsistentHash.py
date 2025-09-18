import hashlib
from sortedcontainers import SortedDict

# 模拟一个哈希环
class ConsistentHash:
    def __init__(self, nodes=None, replicas=100):
        """
        :param nodes: 初始节点列表
        :param replicas: 每个物理节点对应的虚拟节点数量
        """
        self.replicas = replicas
        self.ring = SortedDict()
        self.physical_nodes = set()
        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key):
        """将键哈希为一个32位整数"""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) & 0xFFFFFFFF

    def add_node(self, node):
        """添加一个物理节点及其虚拟节点"""
        self.physical_nodes.add(node)
        for i in range(self.replicas):
            virtual_node = f"{node}#{i}"
            hash_val = self._hash(virtual_node)
            self.ring[hash_val] = node
        print(f"✅ 已添加节点: {node} ({self.replicas} 个虚拟节点)")

    def remove_node(self, node):
        """移除一个物理节点及其虚拟节点"""
        if node not in self.physical_nodes:
            print(f"❌ 错误：节点 {node} 不存在")
            return
            
        self.physical_nodes.remove(node)
        for i in range(self.replicas):
            virtual_node = f"{node}#{i}"
            hash_val = self._hash(virtual_node)
            if hash_val in self.ring:
                del self.ring[hash_val]
        print(f"❌ 已移除节点: {node}")

    def get_node(self, key):
        """获取数据应该存储的节点"""
        if not self.ring:
            return None
        
        hash_val = self._hash(key)
        
        # 找到第一个哈希值大于或等于 key 哈希值的虚拟节点
        # 如果没有找到，则回到环的起点
        idx = self.ring.bisect_left(hash_val)
        if idx == len(self.ring):
            idx = 0
            
        return self.ring.values()[idx]

# ----------------- 演示过程 -----------------

# 1. 初始化哈希环，有三个节点
print("--- 步骤1：初始化三个节点 ---")
nodes = ["NodeA", "NodeB", "NodeC"]
consistent_hash = ConsistentHash(nodes=nodes)

# 2. 模拟数据分配，并记录初始节点
print("\n--- 步骤2：分配1000个数据并记录初始节点 ---")
initial_data_mapping = {}
initial_distribution = {}
for i in range(1000):
    key = f"data_{i}"
    node = consistent_hash.get_node(key)
    initial_data_mapping[key] = node
    initial_distribution.setdefault(node, 0)
    initial_distribution[node] += 1

# 打印初始数据分布情况
total_data = sum(initial_distribution.values())
print("初始数据分布情况：")
for node, count in initial_distribution.items():
    print(f"  {node}: {count} 个 ({count/total_data:.2%})")

# 3. 扩容：增加一个新节点 D
print("\n--- 步骤3：扩容，添加一个新节点 NodeD ---")
consistent_hash.add_node("NodeD")

# 4. 重新分配数据并统计迁移量
print("\n--- 步骤4：重新分配数据并统计迁移量 ---")
new_distribution = {}
migrated_count = 0

for i in range(1000):
    key = f"data_{i}"
    old_node = initial_data_mapping[key]
    new_node = consistent_hash.get_node(key) # 此时get_node会使用新的哈希环
    
    if new_node != old_node:
        migrated_count += 1
        
    new_distribution.setdefault(new_node, 0)
    new_distribution[new_node] += 1

# 打印扩容后的数据分布和迁移情况
print("扩容后数据分布情况：")
for node, count in new_distribution.items():
    print(f"  {node}: {count} 个 ({count/total_data:.2%})")

print(f"\n📢 扩容后需要迁移的数据量： {migrated_count} 个")
print(f"📢 迁移率： {migrated_count/total_data:.2%}")
