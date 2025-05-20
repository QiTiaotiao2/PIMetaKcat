# import os
# import pandas as pd
# import numpy as np
# import torch
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel, T5EncoderModel
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from Bio.PDB import PDBParser
# import config
#
# def save_enzyme_embeddings(df, enzyme_extractor, output_path, device="cpu"):
#     """
#     批量处理酶序列，保存为 .npy 文件。
#     """
#     embeddings = []
#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing enzyme sequences"):
#         seq = row['seq']
#         embedding = enzyme_extractor.extract(seq)
#         embeddings.append(embedding)
#     embeddings = np.array(embeddings)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     np.save(output_path, embeddings)
#     print(f"Enzyme embeddings saved to {output_path}")
#     return embeddings
#
# def load_enzyme_embeddings(file_path):
#     """
#     从 .npy 文件中加载酶嵌入。
#     """
#     embeddings = np.load(file_path)
#     print(f"Enzyme embeddings loaded from {file_path}")
#     return embeddings
#
# class EnzymeMultiModelEmbeddingExtractor:
#     """
#     使用 ESM2（GPU）和 ProtT5（CPU）两种模型提取酶序列特征，并拼接。
#     """
#     def __init__(self,
#                  esm2_model_name: str = "facebook/esm2_t36_3B_UR50D",
#                  prot5_model_name: str = "Rostlab/prot_t5_xl_uniref50",
#                  device: str = "cuda"):
#         # ESM2 部分 (GPU)
#         self.esm2_tokenizer = AutoTokenizer.from_pretrained(
#             esm2_model_name,
#             do_lower_case=False,
#             resume_download=True
#         )
#         self.esm2_model = AutoModel.from_pretrained(
#             esm2_model_name,
#             ignore_mismatched_sizes=True,
#             resume_download=True
#         )
#         self.esm2_model.eval()
#         if hasattr(self.esm2_model, "pooler"):
#             self.esm2_model.pooler = None
#         self.gpu_device = device
#         self.esm2_model.to(self.gpu_device)
#
#         # ProtT5 部分 (CPU)
#         self.prot5_tokenizer = AutoTokenizer.from_pretrained(
#             prot5_model_name,
#             do_lower_case=False,
#             use_fast=False,
#             resume_download=True
#         )
#         self.prot5_model = T5EncoderModel.from_pretrained(
#             prot5_model_name,
#             resume_download=True
#         )
#         self.prot5_model.eval()
#         if hasattr(self.prot5_model, "pooler"):
#             self.prot5_model.pooler = None
#         self.cpu_device = "cpu"
#         self.prot5_model.to(self.cpu_device)
#
#     def extract(self, seq: str) -> np.ndarray:
#         # ============== ESM2 特征提取 ==============
#         inputs_esm2 = self.esm2_tokenizer(
#             seq,
#             return_tensors="pt",
#             add_special_tokens=True,
#             truncation=True,
#             max_length=1022
#         )
#         inputs_esm2 = {k: v.to(self.gpu_device) for k, v in inputs_esm2.items()}
#         with torch.no_grad():
#             out_esm2 = self.esm2_model(**inputs_esm2)
#         feat_esm2 = out_esm2.last_hidden_state.mean(dim=1).squeeze(0)
#         feat_esm2_cpu = feat_esm2.cpu()
#
#         # ============== ProtT5 特征提取 ==============
#         inputs_prot5 = self.prot5_tokenizer(
#             seq,
#             return_tensors="pt",
#             add_special_tokens=True,
#             truncation=True,
#             max_length=1022
#         )
#         inputs_prot5 = {k: v.to(self.cpu_device) for k, v in inputs_prot5.items()}
#         with torch.no_grad():
#             out_prot5 = self.prot5_model(**inputs_prot5)
#         feat_prot5 = out_prot5.last_hidden_state.mean(dim=1).squeeze(0)
#
#         # ============== 拼接最终 embedding ==============
#         combined = torch.cat([feat_esm2_cpu, feat_prot5], dim=-1)
#         return combined.numpy()
#
# class SubstrateEmbeddingExtractor:
#     """
#     使用 ChemBERTa 模型对底物 SMILES 进行嵌入。
#     """
#     def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", device: str = "cpu"):
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.model.eval()
#         self.model.to(self.device)
#
#     def extract(self, smiles: str) -> np.ndarray:
#         inputs = self.tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
#         return embedding.cpu().numpy()
#
# class StructureEmbeddingExtractor:
#     """
#     从 PDB 文件中提取 Cα 原子坐标，用于空间图构建。
#     文件命名假设以 "<idx>_" 开头，其中 idx 对应 DataFrame 的行号。
#     返回：numpy array, shape [N_ca, 3]，若无 PDB 则返回空 array。
#     """
#     def __init__(self, pdb_dir: str):
#         self.pdb_dir = pdb_dir
#         self.parser = PDBParser(QUIET=True)
#
#     def extract(self, idx: int) -> np.ndarray:
#         # 在目录中寻找以 f"{idx}_" 开头的 PDB 文件
#         for fname in os.listdir(self.pdb_dir):
#             if fname.startswith(f"{idx}_") and fname.endswith(".pdb"):
#                 path = os.path.join(self.pdb_dir, fname)
#                 struct = self.parser.get_structure(str(idx), path)
#                 # 收集所有 Cα 原子坐标
#                 coords = np.array([
#                     atom.get_coord()
#                     for atom in struct.get_atoms()
#                     if atom.get_id() == "CA"
#                 ], dtype=float)
#                 return coords  # shape [N_ca, 3]
#         # 若未找到对应 PDB，则返回空坐标
#         return np.zeros((0, 3), dtype=float)
#
# def load_data(file_path: str) -> pd.DataFrame:
#     """
#     尝试用逗号分隔读取，并去除列名空格。
#     """
#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()
#     return df
#
# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     清洗数据：删除关键字段为空的行；过滤 kcat 与温度的合理范围。
#     """
#     essential_cols = ['seq', 'smiles', 'kcat', 'Temp']
#     missing = [col for col in essential_cols if col not in df.columns]
#     if missing:
#         raise KeyError(f"数据中缺少必要的列: {missing}")
#     df = df.dropna(subset=essential_cols)
#     df = df[(df['kcat'] > 1e-3) & (df['kcat'] < 1e5)]
#     df = df[(df['Temp'] >= 0) & (df['Temp'] <= 100)]
#     return df
#
# def extract_features(df: pd.DataFrame,
#                      enzyme_extractor=None,
#                      substrate_extractor=None,
#                      structure_extractor=None,
#                      device: str = "cpu",
#                      precomputed_enzyme_features=None,
#                      use_temperature: bool = True):
#     """
#     针对每一行记录提取：
#       - enzyme_feat: [N, enzyme_dim]
#       - substrate_feat: [N, substrate_dim]
#       - temp_embed: [N, 1]（温度归一化后）
#       - kcat_labels: [N, 1]（log10(kcat)）
#       - reaction_flags: [N]
#       - EC_labels: 列表
#       - info_list: 每个样本的额外信息
#       - coords_list: 列表，每个元素是 shape [N_ca, 3] 的 numpy array
#     """
#     enzyme_features, substrate_features = [], []
#     temp_embeddings, kcat_labels = [], []
#     reaction_flags, EC_labels, info_list, coords_list = [], [], [], []
#
#     for i, (_, row) in enumerate(df.iterrows()):
#         # --- 序列特征 ---
#         if precomputed_enzyme_features is not None:
#             e_feat = precomputed_enzyme_features[i]
#         elif enzyme_extractor is not None:
#             e_feat = enzyme_extractor.extract(row['seq'])
#         else:
#             np.random.seed(abs(hash(row['seq'])) % (10**8))
#             e_feat = np.random.rand(config.ENZYME_DIM)
#         enzyme_features.append(e_feat)
#
#         # --- 结构坐标 ---
#         if structure_extractor is not None:
#             coords = structure_extractor.extract(i)
#         else:
#             coords = np.zeros((0, 3), dtype=float)
#         coords_list.append(coords)
#
#         # --- 底物特征 ---
#         if substrate_extractor is not None:
#             sub_feat = substrate_extractor.extract(row['smiles'])
#         else:
#             np.random.seed(abs(hash(row['smiles'])) % (10**8))
#             sub_feat = np.random.rand(config.SUBSTRATE_DIM)
#         substrate_features.append(sub_feat)
#
#         # --- 温度 ---
#         if use_temperature:
#             temp_norm = (row['Temp'] - 0) / 100.0
#             temp_embeddings.append(np.array([temp_norm], dtype=float))
#         else:
#             temp_embeddings.append(np.empty((0,), dtype=float))
#
#         # --- 标签及其他 ---
#         kcat_labels.append(np.log10(row['kcat']))
#         reaction_flags.append(row.get('reaction_flag', 0))
#         EC_labels.append(row.get('EC', None))
#         info_list.append({'identity': row.get('identity', None)})
#
#     # 转为张量
#     enzyme_feat = torch.tensor(
#         np.stack(enzyme_features), dtype=torch.float32, device=device)
#     substrate_feat = torch.tensor(
#         np.stack(substrate_features), dtype=torch.float32, device=device)
#     temp_embed = torch.tensor(
#         np.stack(temp_embeddings), dtype=torch.float32, device=device)
#     kcat = torch.tensor(
#         np.array(kcat_labels), dtype=torch.float32, device=device).unsqueeze(-1)
#     flags = torch.tensor(
#         np.array(reaction_flags), dtype=torch.long, device=device)
#
#     # 返回最后一项 coords_list 供后续图构建使用
#     return enzyme_feat, substrate_feat, temp_embed, kcat, flags, EC_labels, info_list, coords_list
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import config

def save_enzyme_embeddings(df, enzyme_extractor, output_path, device="cpu"):
    """
    批量处理酶序列，保存为 .npy 文件。
    """
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing enzyme sequences"):
        seq = row['seq']
        embedding = enzyme_extractor.extract(seq)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Enzyme embeddings saved to {output_path}")
    return embeddings

def load_enzyme_embeddings(file_path):
    """
    从 .npy 文件中加载酶嵌入。
    """
    embeddings = np.load(file_path)
    print(f"Enzyme embeddings loaded from {file_path}")
    return embeddings

class EnzymeMultiModelEmbeddingExtractor:
    """
    使用 ESM2（GPU）和 ProtT5（CPU）两种模型提取酶序列特征，并拼接。
    """
    def __init__(self,
                 esm2_model_name: str = "facebook/esm2_t36_3B_UR50D",
                 prot5_model_name: str = "Rostlab/prot_t5_xl_uniref50",
                 device: str = "cuda"):
        # ESM2 部分 (GPU)
        self.esm2_tokenizer = AutoTokenizer.from_pretrained(
            esm2_model_name,
            do_lower_case=False,
            resume_download=True
        )
        self.esm2_model = AutoModel.from_pretrained(
            esm2_model_name,
            ignore_mismatched_sizes=True,
            resume_download=True
        )
        self.esm2_model.eval()
        if hasattr(self.esm2_model, "pooler"):
            self.esm2_model.pooler = None
        self.gpu_device = device
        self.esm2_model.to(self.gpu_device)

        # ProtT5 部分 (CPU)
        self.prot5_tokenizer = AutoTokenizer.from_pretrained(
            prot5_model_name,
            do_lower_case=False,
            use_fast=False,
            resume_download=True
        )
        self.prot5_model = T5EncoderModel.from_pretrained(
            prot5_model_name,
            resume_download=True
        )
        self.prot5_model.eval()
        if hasattr(self.prot5_model, "pooler"):
            self.prot5_model.pooler = None
        self.cpu_device = "cpu"
        self.prot5_model.to(self.cpu_device)

    def extract(self, seq: str) -> np.ndarray:
        # ============== ESM2 特征提取 ==============
        inputs_esm2 = self.esm2_tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=1022
        )
        inputs_esm2 = {k: v.to(self.gpu_device) for k, v in inputs_esm2.items()}
        with torch.no_grad():
            out_esm2 = self.esm2_model(**inputs_esm2)
        feat_esm2 = out_esm2.last_hidden_state.mean(dim=1).squeeze(0).cpu()

        # ============== ProtT5 特征提取 ==============
        inputs_prot5 = self.prot5_tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=1022
        )
        inputs_prot5 = {k: v.to(self.cpu_device) for k, v in inputs_prot5.items()}
        with torch.no_grad():
            out_prot5 = self.prot5_model(**inputs_prot5)
        feat_prot5 = out_prot5.last_hidden_state.mean(dim=1).squeeze(0)

        # ============== 拼接最终 embedding ==============
        combined = torch.cat([feat_esm2, feat_prot5], dim=-1)
        return combined.numpy()

class SubstrateEmbeddingExtractor:
    """
    使用 ChemBERTa 模型对底物 SMILES 进行嵌入。
    """
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def extract(self, smiles: str) -> np.ndarray:
        inputs = self.tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.cpu().numpy()

class StructureEmbeddingExtractor:
    """
    从 PDB 文件中提取 Cα 原子坐标，用于空间图构建。
    文件命名假设以 "<idx>_" 开头，其中 idx 对应 DataFrame 的行号。
    返回：numpy array, shape [N_ca, 3]，若无 PDB 则返回空 array。
    """
    def __init__(self, pdb_dir: str):
        self.pdb_dir = pdb_dir
        self.parser = PDBParser(QUIET=True)

    def extract(self, idx: int) -> np.ndarray:
        for fname in os.listdir(self.pdb_dir):
            if fname.startswith(f"{idx}_") and fname.endswith(".pdb"):
                path = os.path.join(self.pdb_dir, fname)
                struct = self.parser.get_structure(str(idx), path)
                coords = np.array([
                    atom.get_coord()
                    for atom in struct.get_atoms()
                    if atom.get_id() == "CA"
                ], dtype=float)
                return coords
        return np.zeros((0, 3), dtype=float)

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    essential_cols = ['seq', 'smiles', 'kcat', 'Temp']
    missing = [col for col in essential_cols if col not in df.columns]
    if missing:
        raise KeyError(f"数据中缺少必要的列: {missing}")
    df = df.dropna(subset=essential_cols)
    df = df[(df['kcat'] > 1e-3) & (df['kcat'] < 1e5)]
    df = df[(df['Temp'] >= 0) & (df['Temp'] <= 100)]
    return df

def extract_features(df: pd.DataFrame,
                     enzyme_extractor=None,
                     substrate_extractor=None,
                     structure_extractor=None,
                     device: str = "cpu",
                     precomputed_enzyme_features=None,
                     use_temperature: bool = True):
    """
    针对每一行记录提取各类特征，并添加两条进度条：
      - “Extracting enzyme features”
      - “Extracting substrate features”
    返回同原接口，末尾多 coords_list。
    """
    N = len(df)
    enzyme_features, substrate_features = [], []
    temp_embeddings, kcat_labels = [], []
    reaction_flags, EC_labels, info_list, coords_list = [], [], [], []

    # 初始化进度条
    bar_e = tqdm(total=N, desc="Extracting enzyme features")
    bar_s = tqdm(total=N, desc="Extracting substrate features")

    for i, (_, row) in enumerate(df.iterrows()):
        # 酶序列特征
        if precomputed_enzyme_features is not None:
            e_feat = precomputed_enzyme_features[i]
        elif enzyme_extractor is not None:
            e_feat = enzyme_extractor.extract(row['seq'])
        else:
            np.random.seed(abs(hash(row['seq'])) % (10**8))
            e_feat = np.random.rand(config.ENZYME_DIM)
        enzyme_features.append(e_feat)
        bar_e.update(1)

        # 结构坐标
        if structure_extractor is not None:
            coords = structure_extractor.extract(i)
        else:
            coords = np.zeros((0, 3), dtype=float)
        coords_list.append(coords)

        # 底物 SMILES 特征
        if substrate_extractor is not None:
            s_feat = substrate_extractor.extract(row['smiles'])
        else:
            np.random.seed(abs(hash(row['smiles'])) % (10**8))
            s_feat = np.random.rand(config.SUBSTRATE_DIM)
        substrate_features.append(s_feat)
        bar_s.update(1)

        # 温度 embedding
        if use_temperature:
            temp_norm = (row['Temp'] - 0) / 100.0
            temp_embeddings.append(np.array([temp_norm], dtype=float))
        else:
            temp_embeddings.append(np.empty((0,), dtype=float))

        # 标签及其他
        kcat_labels.append(np.log10(row['kcat']))
        reaction_flags.append(row.get('reaction_flag', 0))
        EC_labels.append(row.get('EC', None))
        info_list.append({'identity': row.get('identity', None)})

    bar_e.close()
    bar_s.close()

    # 转为张量
    enzyme_feat = torch.tensor(np.stack(enzyme_features), dtype=torch.float32, device=device)
    substrate_feat = torch.tensor(np.stack(substrate_features), dtype=torch.float32, device=device)
    temp_embed = torch.tensor(np.stack(temp_embeddings), dtype=torch.float32, device=device)
    kcat = torch.tensor(np.array(kcat_labels), dtype=torch.float32, device=device).unsqueeze(-1)
    flags = torch.tensor(np.array(reaction_flags), dtype=torch.long, device=device)

    return enzyme_feat, substrate_feat, temp_embed, kcat, flags, EC_labels, info_list, coords_list
