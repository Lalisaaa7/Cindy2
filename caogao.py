import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("is cuda available:", torch.cuda.is_available())
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ---------- 全局 K-mer 词表 ----------
GLOBAL_VOCAB = {}

class SimpleDiffusionGenerator(nn.Module):
    """
    模拟扩散生成正样本的框架，针对蛋白质图节点特征生成结合位点样本
    """
    def __init__(self, input_dim, noise_dim=32):
        super(SimpleDiffusionGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 64)
        self.fc2 = nn.Linear(64, input_dim)

    def forward(self, noise):
        x = F.relu(self.fc1(noise))
        return torch.sigmoid(self.fc2(x))  # 输出模拟 one-hot 特征

class DiffusionModel:
    """
    包裹生成器模型，负责学习已有正类分布并生成新样本
    """
    def __init__(self, input_dim, device='cpu'):
        self.generator = SimpleDiffusionGenerator(input_dim).to(device)
        self.device = device

    def train_on_positive_samples(self, all_data):
        """
        用已有正类样本训练生成器模型（仅一次）
        """
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # 收集所有 label == 1 的特征
        positive_vectors = []
        for data in all_data:
            x = data.x  # shape: [L, D]
            y = data.y  # shape: [L]
            pos_feats = x[y == 1]
            if pos_feats.shape[0] > 0:
                positive_vectors.append(pos_feats)

        if not positive_vectors:
            print("⚠️ 没有可用于训练的正类样本")
            return

        positive_data = torch.cat(positive_vectors, dim=0).to(self.device)

        # 训练生成器（简单训练几轮）
        self.generator.train()
        for epoch in range(10):
            noise = torch.randn_like(positive_data[:, :32])
            generated = self.generator(noise)
            loss = criterion(generated, positive_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch == 9:
                print(f"✅ 生成器训练完成，最后loss={loss.item():.4f}")

    def generate_positive_sample(self, num_samples=10):
        """
        使用训练好的生成器生成正样本节点特征
        """
        self.generator.eval()
        noise = torch.randn((num_samples, 32)).to(self.device)
        with torch.no_grad():
            generated = self.generator(noise)
        return generated


# ---------- K-mer Encoding ----------
def extract_kmer_features(sequence, k=3):
    kmer_list = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer not in GLOBAL_VOCAB:
            GLOBAL_VOCAB[kmer] = len(GLOBAL_VOCAB)
        kmer_list.append(GLOBAL_VOCAB[kmer])

    one_hot = torch.zeros(len(kmer_list), len(GLOBAL_VOCAB))
    for i, idx in enumerate(kmer_list):
        one_hot[i, idx] = 1
    return one_hot


# ---------- Graph Construction ----------
def sequence_to_graph(sequence, label, k=3, window_size=5):
    x = extract_kmer_features(sequence, k)  # shape: [L', D]
    y = torch.tensor([int(i) for i in label[k - 1:]], dtype=torch.long)

    num_nodes = x.size(0)
    assert len(y) == num_nodes, f"label length mismatch after k-mer: {len(label)} vs {num_nodes}"

    edge_index = []
    for i in range(num_nodes):
        for j in range(i - window_size, i + window_size + 1):
            if i != j and 0 <= j < num_nodes:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=y)


# ---------- TXT 文件读取 ----------
def parse_txt_file(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    data_list = []
    buffer = []

    for line in lines:
        if line.startswith('>'):
            if len(buffer) == 3:
                name = buffer[0][1:]
                seq = buffer[1]
                label = buffer[2]
                if len(seq) == len(label):
                    data = sequence_to_graph(seq, label)
                    data.name = name
                    data.source_file = os.path.basename(path)
                    data_list.append(data)
                else:
                    print(f"⚠️ Skipping {name}: length mismatch ({len(seq)} vs {len(label)})")
            buffer = [line]
        else:
            buffer.append(line)

    if len(buffer) == 3:
        name = buffer[0][1:]
        seq = buffer[1]
        label = buffer[2]
        if len(seq) == len(label):
            data = sequence_to_graph(seq, label)
            data.name = name
            data.source_file = os.path.basename(path)
            data_list.append(data)
        else:
            print(f"⚠️ Skipping {name}: length mismatch ({len(seq)} vs {len(label)})")

    return data_list

def build_global_vocab(folder_path, k=3):
    vocab = set()
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            path = os.path.join(folder_path, filename)
            with open(path, 'r') as f:
                lines = f.readlines()
            for i in range(len(lines)):
                if lines[i].startswith('>') and i + 1 < len(lines):
                    seq = lines[i + 1].strip()
                    for j in range(len(seq) - k + 1):
                        vocab.add(seq[j:j + k])
    return {kmer: idx for idx, kmer in enumerate(sorted(vocab))}


# ---------- 数据加载入口 ----------
def load_raw_dataset(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            file_data = parse_txt_file(file_path)
            all_data.extend(file_data)
    return all_data


# ---------- 数据集划分逻辑 ----------
def split_dataset_by_filename(all_data):
    train_data = [d for d in all_data if 'Train' in d.source_file]
    test_data  = [d for d in all_data if 'Test' in d.source_file]

    if len(train_data) == 0 or len(test_data) == 0:
        print("❌ No matching files found. Check that filenames include 'Train' or 'Test'")
        return [], [], []

    # 再从训练集中划分验证集
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data


def sequence_to_onehot(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWXY"
    aa_dict = {aa: i for i, aa in enumerate(amino_acids)}
    onehot = torch.zeros((len(sequence), len(amino_acids)))
    for i, aa in enumerate(sequence):
        if aa in aa_dict:
            onehot[i, aa_dict[aa]] = 1
    return onehot


# ---------- 数据平衡逻辑 ----------
def augment_with_diffusion(diff_model, data_list):
    augmented_data = []
    for data in data_list:
        num_pos = int((data.y == 1).sum().item())
        if num_pos == 0:
            continue  # 跳过没有正类的样本

        new_x = diff_model.generate_positive_sample(num_samples=num_pos)

        # 使用同样的图结构和标签结构
        new_data = Data(
            x=new_x,
            edge_index=data.edge_index.clone(),
            y=torch.ones_like(data.y)  # 全1标签
        )
        new_data.name = data.name + "_gen"
        new_data.source_file = data.source_file
        augmented_data.append(new_data)
    return augmented_data



def balance_dataset(original_data, augmented_data):
    """
    将原始数据与生成的正类样本合并，生成平衡的数据集。
    """
    balanced_data = original_data + augmented_data
    return balanced_data


# ---------- GCN 模型 ---------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x




# ---------- 训练函数 ---------------
def train_model(balanced_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    使用平衡后的数据集来训练模型，使用GCN（图卷积网络）作为示范。
    """
    print("Training model with balanced dataset...")

    loader = DataLoader(balanced_data, batch_size=32, shuffle=True)

    # 定义超参数
    in_channels = balanced_data[0].x.shape[1]  # 输入特征维度
    hidden_channels = 64  # 隐藏层维度
    out_channels = 2  # 输出类别数（假设是二分类问题）
    lr = 1e-3  # 学习率
    epochs = 50  # 训练轮次

    # 初始化模型、优化器和损失函数
    model = GCN(in_channels, hidden_channels, out_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()  # 分类任务的损失函数

    # 将模型转移到 GPU（如果有）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 开始训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in loader:
            batch = batch.to(device)  # 将batch转移到GPU（如果有）
            optimizer.zero_grad()

            # 前向传播
            out = model(batch.x, batch.edge_index)

            # 计算损失和准确率
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            # 计算准确率
            pred = out.argmax(dim=1)  # 获取最大值的索引作为预测类别
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs

            total_loss += loss.item()

        # 输出训练过程中的信息
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(loader):.4f} - Accuracy: {correct / total:.4f}")

    print("Training completed!")


# ------------- 主函数 ---------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = './Raw_data'

    global GLOBAL_VOCAB
    GLOBAL_VOCAB = build_global_vocab(folder, k=3)

    data_all = load_raw_dataset(folder)
    train_data, val_data, test_data = split_dataset_by_filename(data_all)


    if len(train_data) == 0:
        print("❌ 无有效训练数据")
        return

    # ✅ 初始化并训练扩散生成器
    feature_dim = train_data[0].x.shape[1]
    diff_model = DiffusionModel(input_dim=feature_dim, device=device)
    diff_model.train_on_positive_samples(train_data)

    # ✅ 使用扩散生成器生成新样本
    augmented_train_data = augment_with_diffusion(diff_model, train_data)
    balanced_train_data = balance_dataset(train_data, augmented_train_data)

    # ✅ 训练 GCN 模型
    train_model(balanced_train_data)



if __name__ == '__main__':
    main()
