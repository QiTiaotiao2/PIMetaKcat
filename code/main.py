import colorama
colorama.deinit()

import os
import torch
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
from collections import Counter
from sklearn.metrics import mean_squared_error, r2_score

import config
from data_preprocessing import (
    load_data,
    clean_data,
    extract_features,
    EnzymeMultiModelEmbeddingExtractor,
    SubstrateEmbeddingExtractor,
    StructureEmbeddingExtractor,
    save_enzyme_embeddings,
    load_enzyme_embeddings
)
from utils import EnzymeDataset, get_dataloader
from models import FewShotRegressionModel  # EnhancedFewShotRegressionModel
from training import Trainer

def evaluate(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            enzyme_feat, substrate_feat, temp_embed, kcat_labels, reaction_flags, cluster_ids, coords_list = batch
            enzyme_feat = enzyme_feat.to(device)
            substrate_feat = substrate_feat.to(device)
            temp_embed = temp_embed.to(device)
            kcat_labels = kcat_labels.to(device)
            # forward: pass coords_list to model
            kcat_preds = model(enzyme_feat, substrate_feat, temp_embed, coords_list)
            # squeeze to [B,]
            kcat_preds = kcat_preds.squeeze(-1)
            true_vals = kcat_labels.squeeze(-1)
            preds.append(kcat_preds.cpu())
            trues.append(true_vals.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2   = r2_score(trues, preds)
    mse  = mean_squared_error(trues, preds)
    pcc  = np.corrcoef(trues, preds)[0,1] if len(trues)>1 else float('nan')
    return {"RMSE": rmse, "R2": r2, "MSE": mse, "PCC": pcc}

def main():
    device = config.DEVICE

    # 1. Load & clean data
    df_train = load_data(config.TRAIN_DATA_PATH)
    df_train = clean_data(df_train)
    df_train["cluster_original"] = df_train["cluster"]

    df_test  = load_data(config.TEST_DATA_PATH)
    df_test  = clean_data(df_test)
    df_test["cluster_original"] = df_test["cluster"]

    # 2. Feature extractors
    if config.USE_PRECOMPUTED_ENZYME:
        enzyme_extractor = None
    else:
        enzyme_extractor = EnzymeMultiModelEmbeddingExtractor(
            esm2_model_name="facebook/esm2_t36_3B_UR50D",
            prot5_model_name="Rostlab/prot_t5_xl_uniref50",
            device=device
        )
    substrate_extractor = SubstrateEmbeddingExtractor(device=device)
    structure_extractor_train = StructureEmbeddingExtractor(config.TRAIN_PDB_DIR)
    structure_extractor_test  = StructureEmbeddingExtractor(config.TEST_PDB_DIR)

    # 3. Precompute or load enzyme embeddings
    if config.USE_PRECOMPUTED_ENZYME:
        if os.path.exists(config.ENZYME_EMBEDDING_PATH_TRAIN):
            emb_train = load_enzyme_embeddings(config.ENZYME_EMBEDDING_PATH_TRAIN)
        else:
            emb_train = save_enzyme_embeddings(
                df_train, enzyme_extractor,
                config.ENZYME_EMBEDDING_PATH_TRAIN, device=device
            )
        if os.path.exists(config.ENZYME_EMBEDDING_PATH_TEST):
            emb_test = load_enzyme_embeddings(config.ENZYME_EMBEDDING_PATH_TEST)
        else:
            emb_test = save_enzyme_embeddings(
                df_test, enzyme_extractor,
                config.ENZYME_EMBEDDING_PATH_TEST, device=device
            )
    else:
        emb_train = emb_test = None

    # 4. Extract features (including coords_list)
    (
        enzyme_feat_train,
        substrate_feat_train,
        temp_embed_train,
        kcat_train,
        flag_train,
        EC_train,
        info_train,
        coords_list_train
    ) = extract_features(
        df_train,
        enzyme_extractor,
        substrate_extractor,
        structure_extractor_train,
        device=device,
        precomputed_enzyme_features=emb_train,
        use_temperature=config.USE_TEMPERATURE
    )
    (
        enzyme_feat_test,
        substrate_feat_test,
        temp_embed_test,
        kcat_test,
        flag_test,
        EC_test,
        info_test,
        coords_list_test
    ) = extract_features(
        df_test,
        enzyme_extractor,
        substrate_extractor,
        structure_extractor_test,
        device=device,
        precomputed_enzyme_features=emb_test,
        use_temperature=config.USE_TEMPERATURE
    )

    print("Reaction flag distribution:", Counter(flag_train.cpu().numpy()))
    print("Enzyme feature dim:", enzyme_feat_train.size(1))
    print("Test set Cα avg nodes:", np.mean([c.shape[0] for c in coords_list_test]))

    # 5. Encode clusters
    df_train['cluster'] = df_train['cluster'].astype('category').cat.codes
    df_test ['cluster'] = df_test ['cluster'].astype('category').cat.codes
    cluster_ids_train = torch.tensor(df_train['cluster'].values, dtype=torch.long)
    cluster_ids_test  = torch.tensor(df_test ['cluster'].values, dtype=torch.long)

    # 6. Build datasets & dataloaders
    train_dataset = EnzymeDataset(
        enzyme_feat_train,
        substrate_feat_train,
        temp_embed_train,
        kcat_train,
        flag_train,
        cluster_ids_train,
        coords_list_train
    )
    test_dataset = EnzymeDataset(
        enzyme_feat_test,
        substrate_feat_test,
        temp_embed_test,
        kcat_test,
        flag_test,
        cluster_ids_test,
        coords_list_test
    )
    train_loader = get_dataloader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader  = get_dataloader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False)

    # 7. Model & optimizer
    model = FewShotRegressionModel(
        enzyme_dim=enzyme_feat_train.size(1),
        substrate_dim=substrate_feat_train.size(1),
        feature_dim=config.FEATURE_DIM
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_GLOBAL, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 8. Trainer
    trainer = Trainer(model, optimizer, None, device, global_logA=float(kcat_train.mean()), use_classification=False, dataset=train_dataset)

    # 9. Training loop
    best_val_loss = float('inf')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"===== Epoch {epoch}/{config.NUM_EPOCHS} =====")
        # Meta or stage logic omitted here; Trainer will handle internal flags
        train_loss = trainer.train_epoch(train_loader, current_epoch=epoch, total_epochs=config.NUM_EPOCHS,
                                         lambda_ewc=config.LAMBDA_EWC, use_ewc=config.USE_EWC)
        val_loss   = trainer.validate(test_loader, base_physics_weight=config.WEIGHT_PHYSICS)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = f"best_model_{timestamp}.pth"
            torch.save(model.state_dict(), best_model)

    print("Training complete. Best val loss:", best_val_loss)
    model.load_state_dict(torch.load(best_model))

    # 10. Final evaluation
    metrics = evaluate(model, test_loader, device)
    print("Test Metrics:", metrics)

    # 11. Save predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            enzyme_feat, substrate_feat, temp_embed, *_ , coords_list = batch
            enzyme_feat = enzyme_feat.to(device)
            substrate_feat = substrate_feat.to(device)
            temp_embed = temp_embed.to(device)
            pred = model(enzyme_feat, substrate_feat, temp_embed, coords_list)
            preds.extend(pred.cpu().squeeze(-1).tolist())
    df_test['predicted_kcat'] = preds
    df_test['actual_kcat']    = np.log10(df_test['kcat'])
    out_df = df_test[["seq", "smiles", "Temp", "actual_kcat", "predicted_kcat", "EC", "cluster_original"]]
    out_file = f"test_results_{timestamp}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Saved test results to {out_file}")

if __name__ == "__main__":
    main()


