"https://github.com/annahdo/implementing_activation_steering.git"

import torch, json, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

device = 'cuda' if torch.cuda.is_available() else 'cpu'
name = "nntsuzu/kikka-2b-base"

model = AutoModelForCausalLM.from_pretrained(
    name, 
    dtype="bfloat16", 
    trust_remote_code=True
).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(name)


def load_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_hook(storage_list):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # shape: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
        storage_list.append(hidden[:, -1, :].detach().cpu())
    return hook


def get_vector(standard_dataset, character_dataset, layers:str):
    std_activations, std_handles = {}, []
    chara_activations, chara_handles = {}, []
    final_vectors = {}
    
    layer_start, layer_end = map(int, layers.split('-'))
    target_layers = list(range(layer_start, layer_end + 1))
    diffs = {layer: [] for layer in target_layers}

    for (std_idx, std_item), (chara_idx, chara_item) in zip(enumerate(standard_dataset), enumerate(character_dataset)):
        input_std = tokenizer(std_item, tokenize=True, return_tensors="pt", add_generation_prompt=False).to(model.device)
        input_chara = tokenizer(chara_item, tokenize=True, return_tensors="pt", add_generation_prompt=False).to(model.device)

        # standard input to model
        for layer_idx in target_layers:
            std_activations[layer_idx] = []
            layer_module = model.model.language_model.layers[layer_idx]
            handle = layer_module.register_forward_hook(get_hook(std_activations[layer_idx]))
            std_handles.append(handle)
        with torch.no_grad():
            model(**input_std)
        for h in std_handles:
            h.remove()

        # character input to model
        for layer_idx in target_layers:
            chara_activations[layer_idx] = []
            layer_module = model.model.language_model.layers[layer_idx]
            handle = layer_module.register_forward_hook(get_hook(chara_activations[layer_idx]))
            chara_handles.append(handle)
        with torch.no_grad():
            model(**input_chara)
        for h in chara_handles:
            h.remove()

        # 差分計算
        for layer_idx in target_layers:
            if std_activations[layer_idx] and chara_activations[layer_idx]:
                vec_std = std_activations[layer_idx][0]  # [1, hidden_dim]
                vec_chara = chara_activations[layer_idx][0]
                diff = vec_chara - vec_std
                diffs[layer_idx].append(diff)

    # vector計算
    for layer_idx, diff_list in diffs.items():
        if not diff_list:
            print(f"Warning: The data for the {layer_idx} layer is empty, skipped.")
            continue
        
        # [num_samples, hidden_dim]
        X = torch.cat(diff_list, dim=0).float().numpy()
        
        # PCAで第1主成分を抽出
        pca = PCA(n_components=1)
        pca.fit(X)
        
        # 方向ベクトル
        direction = pca.components_[0]  # [hidden_dim]
        
        # 符号の調整: 平均との内積が正になるようにする
        mean_diff = np.mean(X, axis=0)
        if np.dot(direction, mean_diff) < 0:
            direction = -direction
        
        final_vectors[layer_idx] = torch.tensor(direction, dtype=torch.float16)
        print(f"layer {layer_idx}: 分散説明率 {pca.explained_variance_ratio_[0]:.4f}")
    
    return final_vectors


def save_vectors(vectors):
    torch.save(vectors, "./character_vector_manual.pt")
    print("✅ saved to ./character_vector_manual.pt")
    
    # 統計情報を保存
    with open("./character_vector_states.txt", 'w', encoding='utf-8') as f:
        f.write("# character Vector Statistics\n\n")
        f.write(f"Total layers: {len(vectors)}\n")
        f.write(f"Layer indices: {sorted(vectors.keys())}\n\n")
        for layer_idx in sorted(vectors.keys()):
            vec = vectors[layer_idx]
            f.write(f"Layer {layer_idx}:\n")
            f.write(f"  Shape: {vec.shape}\n")
            f.write(f"  Norm: {vec.norm().item():.4f}\n")
            f.write(f"  Mean: {vec.mean().item():.6f}\n")
            f.write(f"  Std: {vec.std().item():.6f}\n\n")
    
    print("✅ 統計情報保存完了: ./character_vector_states.txt")


std = load_dataset("./happiness.json")
chara = load_dataset("./irony.json")
vector = get_vector(standard_dataset=std, character_dataset=chara, layers="13-14")
save_vectors(vector)