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

# ---------- Simple Diffusion Generator ----------
class SimpleDiffusionGenerator(nn.Module):
    def __init__(self, input_dim, noise_dim=32):
        super(SimpleDiffusionGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 64)
        self.fc2 = nn.Linear(64, input_dim)

    def forward(self, noise):
        x = F.relu(self.fc1(noise))
        return torch.sigmoid(self.fc2(x))

def balance_dataset(original_data, augmented_data):
    """
    将原始数据与生成的正类样本合并，生成平衡的数据集。
    """
    balanced_data = original_data + augmented_data
    return balanced_data

class DiffusionModel:
    def __init__(self, input_dim, device='cpu'):
        self.generator = SimpleDiffusionGenerator(input_dim).to(device)
        self.device = device

    def train_on_positive_samples(self, all_data, batch_size=256, epochs=10):
        """
        用已有正类样本训练生成器模型（支持分批训练，防止OOM）
        """
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # 收集所有正类样本特征
        positive_vectors = []
        for data in all_data:
            pos_feats = data.x[data.y == 1]
            if pos_feats.size(0) > 0:
                positive_vectors.append(pos_feats.cpu())  # 先留在 CPU
        if not positive_vectors:
            print("⚠️ 没有可用于训练的正类样本")
            return

        full_pos_data = torch.cat(positive_vectors, dim=0)  # 留在 CPU
        data_size = full_pos_data.size(0)

        print(f"✅ 正类样本总量：{data_size}，使用 batch_size={batch_size} 训练扩散生成器")

        self.generator.train()
        for epoch in range(epochs):
            perm = torch.randperm(data_size)
            epoch_loss = 0
            for i in range(0, data_size, batch_size):
                indices = perm[i:i + batch_size]
                batch_data = full_pos_data[indices].to(self.device)
                noise = torch.randn((batch_data.size(0), 32)).to(self.device)

                generated = self.generator(noise)
                loss = criterion(generated, batch_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Generator Loss: {epoch_loss:.4f}")

            # 清理显存防止碎片堆积
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def generate_positive_sample(self, num_samples=10):
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
    x = extract_kmer_features(sequence, k)
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

# ---------- 数据读取 ----------
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

# ---------- 数据入口 ----------
def load_raw_dataset(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            file_data = parse_txt_file(file_path)
            all_data.extend(file_data)
    return all_data

def split_dataset_by_filename(all_data):
    train_data = [d for d in all_data if 'Train' in d.source_file]
    test_data = [d for d in all_data if 'Test' in d.source_file]
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data

# ---------- 扩散增强 ----------
def augment_with_diffusion(diff_model, data_list, device):
    augmented_data = []
    for data in data_list:
        num_pos = int((data.y == 1).sum().item())
        if num_pos == 0:
            continue

        new_x = diff_model.generate_positive_sample(num_samples=num_pos)
        new_x = new_x.cpu()  # 先放 CPU 再转回 GPU 统一管理

        #  构建一个简单线性图：0-1-2-3-...-n
        edge_index = []
        for i in range(num_pos - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])  # 双向

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        #  构造 y 全 1，大小等于 num_pos
        new_y = torch.ones(num_pos, dtype=torch.long)

        new_data = Data(x=new_x, edge_index=edge_index, y=new_y).to(device)
        new_data.name = data.name + "_gen"
        new_data.source_file = data.source_file
        augmented_data.append(new_data)
    return augmented_data

# ---------- GCN 模型 ----------
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

# ---------- 训练 ----------
def train_model(balanced_data, device):
    print("Training model with balanced dataset...")
    loader = DataLoader(balanced_data, batch_size=8, shuffle=True)
    in_channels = balanced_data[0].x.shape[1]
    model = GCN(in_channels, hidden_channels=64, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(50):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_nodes
            total_loss += loss.item()

        acc = correct / total
        print(f"Epoch {epoch+1}/50 - Loss: {total_loss/len(loader):.4f} - Acc: {acc:.4f}")

        #  保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f" Saved best model at epoch {epoch+1} (Acc: {acc:.4f})")

    return model

def test_model(model, test_data, device):
    print("\n Evaluating on test set...")
    model.eval()
    loader = DataLoader(test_data, batch_size=1, shuffle=False)

    total, correct = 0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_nodes

    print(f" Test Accuracy: {correct / total:.4f}")


# ---------- 主函数 ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = './Raw_data'
    global GLOBAL_VOCAB
    GLOBAL_VOCAB = build_global_vocab(folder, k=3)
    data_all = load_raw_dataset(folder)
    train_data, val_data, test_data = split_dataset_by_filename(data_all)
    train_data = [d.to(device) for d in train_data]
    val_data = [d.to(device) for d in val_data]
    test_data = [d.to(device) for d in test_data]
    if len(train_data) == 0:
        print(" 无有效训练数据")
        return
    feature_dim = train_data[0].x.shape[1]
    diff_model = DiffusionModel(input_dim=feature_dim, device=device)
    diff_model.train_on_positive_samples(train_data)
    augmented_train_data = augment_with_diffusion(diff_model, train_data, device)
    balanced_train_data = balance_dataset(train_data, augmented_train_data)
    model = train_model(balanced_train_data, device)

    #  重新加载最优模型（保险起见）
    model.load_state_dict(torch.load('best_model.pt'))

    #  在测试集上评估
    test_model(model, test_data, device)
if __name__ == '__main__':
    main()
