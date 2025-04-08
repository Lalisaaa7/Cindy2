# main.py

import torch
from diffusion_gnn_mask import train_model, evaluate_model
from utils import process_train_dataset, process_val_dataset  # 用你已有的
import random

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = './Graph/646.pt'

    print("🔄 Loading dataset...")
    data = torch.load(data_path, weights_only=False)


    # 划分训练集与验证集
    random.seed(42)
    num_train = int(0.7 * len(data))
    train_raw = random.sample(data, num_train)
    val_raw = [x for x in data if x not in train_raw]

    train_data = process_train_dataset(train_raw)
    val_data = process_val_dataset(val_raw)

    # 参数设置
    in_dim = train_data[0].x.shape[1]  # 特征维度
    hidden_dim = 64
    T = 1000
    epochs = 50
    lr = 1e-3

    print("🚀 Starting training...")
    model, betas, alphas, alphas_bar = train_model(
        dataset=train_data,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        T=T,
        epochs=epochs,
        lr=lr,
        device=device
    )

    print("✅ Training done. Now evaluating on validation set...")
    evaluate_model(model, val_data, alphas, alphas_bar, T, device)

if __name__ == '__main__':
    main()
