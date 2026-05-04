# Contrastive Cluster v2

This folder is for inspecting what the cross-ego contrastive stage changed in
SigLIP2 representations.

## Experiments

1. **Team observation alignment**
   - Compare off-the-shelf SigLIP2 to saved contrastive epochs.
   - Measure same-observation cosine similarity, negative cosine similarity,
     retrieval hit rate for another alive teammate from the same observation,
     and the positive-minus-negative margin.
   - Use this to show whether the training objective actually pulls teammate
     perspectives together.

2. **Embedding geometry and cluster emergence**
   - Run t-SNE on either the pre-projector vision space or the trained
     contrastive projector space.
   - Color the same points by team side, alive teammate count, start time, and
     inferred k-means clusters.
   - Save cluster summaries so unusual clusters can be inspected by match,
     round, side, time, and alive count.

3. **Language-anchor preservation and drift**
   - Keep the SigLIP2 text encoder fixed and compare video embeddings against a
     CS-oriented concept vocabulary.
   - Track which concept groups rise or fall across checkpoints even though the
     contrastive loss never directly sees text.
   - Plot a joint t-SNE of video points and fixed text anchors to show how the
     visual space moves relative to language anchors.
   - This probe uses the SigLIP2 vision/pre-projector embedding, not the
     contrastive projector, because the projector is trained for team alignment
     rather than original SigLIP2 text compatibility.

## Example

```powershell
python src\scripts\contra_cluster_v2\analyze_contrastive_space.py `
  --experiment main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo `
  --epochs 1 4 10 19 27 39 `
  --max-observations 256 `
  --batch-size 32 `
  --spaces vision projected `
  --run-language
```

If checkpoints were downloaded into a nonstandard folder, point the script at
the experiment and checkpoint directories directly:

```powershell
python src\scripts\contra_cluster_v2\analyze_contrastive_space.py `
  --experiment-dir output\main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo `
  --checkpoint-dir output\main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo\checkpoint `
  --all-epochs `
  --tsne-epochs 1 4 10 19 27 39 `
  --max-observations 256 `
  --batch-size 32 `
  --spaces vision projected `
  --run-language
```

Wiki concept vocabulary:

```powershell
python src\scripts\contra_cluster_v2\analyze_contrastive_space.py `
  --experiment main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo `
  --epochs 1 4 10 19 27 39 `
  --spaces vision projected `
  --skip-embedding `
  --run-language `
  --language-vocab data\wiki\concepts.json `
  --language-text-key concept `
  --language-count-key count `
  --language-prompt-mode ensemble `
  --language-top-k 40
```

Wiki bigram vocabulary with tactical grouping:

```powershell
python src\scripts\contra_cluster_v2\analyze_contrastive_space.py `
  --experiment main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo `
  --epochs 1 4 10 19 27 39 `
  --skip-embedding `
  --language-only `
  --run-language `
  --language-vocab data\wiki\bigrams.json `
  --language-text-key bigram `
  --language-count-key count `
  --language-prompt-mode ensemble `
  --language-group-mode tactical `
  --language-group-threshold 0.22 `
  --language-group-margin 0.03 `
  --language-top-k 80
```

Outputs are written to:

```text
artifacts/contra_cluster_v2/<experiment>/
```

The most useful first files are:

- `alignment_metrics.csv`
- `alignment_trajectory.png`
- `embedding_change_metrics.csv`
- `embedding_change_trajectory.png`
- `language_group_metrics.csv`
- `language_concept_drift.csv`
- `tsne_vision_team_side.png`
- `tsne_projected_cluster.png`
- `language_anchor_tsne.png`

For wiki concepts, language outputs are written to:

```text
artifacts/contra_cluster_v2/<experiment>/language_concepts/
```

The most useful files there are:

- `language_concept_drift.csv`
- `language_delta_distribution.png`
- `language_baseline_vs_final_scatter.png`
- `language_group_centered_delta_boxplot.png`
- `language_all_concepts_centered_heatmap.png`
- `language_top_centered_concept_deltas.png`
- `language_top_centered_drift_heatmap.png`
- `language_top_concept_deltas.png`
- `language_top_drift_heatmap.png`
- `language_rank_stability.png`

For wiki bigrams, language outputs are written to:

```text
artifacts/contra_cluster_v2/<experiment>/language_bigrams/
```

The grouping audit is:

- `language_vocab_assignments.csv`
- `language_vocab_manifest.json`
