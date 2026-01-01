#!/usr/bin/env python3
"""
Analyze Type Structure in Walsh Models (Archetype Framework)
-----------------------------------------------------------
Analysis using archetype-based type vocabulary.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import argparse
import tiktoken

# -----------------------------------------------------------------------------
# ARCHETYPE VOCABULARY - Wikipedia-level words
# -----------------------------------------------------------------------------

SEMANTIC_CATEGORIES = {
    # Entity Types
    "Person": [
        "king", "queen", "soldier", "doctor", "teacher", "artist", "scientist",
        "philosopher", "emperor", "prophet", "hero", "villain", "mother", "father",
        "child", "warrior", "priest", "merchant", "farmer", "craftsman"
    ],
    
    "Place": [
        "mountain", "river", "ocean", "forest", "desert", "island", "valley",
        "city", "village", "temple", "castle", "palace", "garden", "battlefield",
        "market", "harbor", "crossroads", "frontier", "sanctuary", "wilderness"
    ],
    
    "Object": [
        "sword", "crown", "book", "mirror", "key", "ring", "cup", "stone",
        "fire", "water", "gold", "silver", "bread", "wine", "cloth", "iron",
        "wheel", "bridge", "door", "window"
    ],
    
    "Abstract": [
        "truth", "beauty", "justice", "wisdom", "power", "love", "death", "time",
        "fate", "freedom", "honor", "duty", "faith", "hope", "fear", "anger",
        "joy", "sorrow", "peace", "war"
    ],
    
    # Process Types
    "Creation": [
        "birth", "growth", "building", "crafting", "writing", "painting", "singing",
        "planting", "forging", "weaving", "cooking", "teaching", "healing", "blessing"
    ],
    
    "Destruction": [
        "death", "decay", "breaking", "burning", "falling", "fading", "killing",
        "cursing", "corrupting", "betraying", "abandoning", "forgetting", "losing"
    ],
    
    "Transformation": [
        "change", "becoming", "awakening", "journey", "crossing", "ascending",
        "descending", "returning", "remembering", "discovering", "choosing", "sacrifice"
    ],
    
    # Relation Types
    "Connection": [
        "marriage", "friendship", "alliance", "covenant", "promise", "gift",
        "inheritance", "teaching", "guidance", "protection", "service", "loyalty"
    ],
    
    "Conflict": [
        "battle", "rivalry", "betrayal", "revenge", "judgment", "punishment",
        "exile", "conquest", "rebellion", "resistance", "challenge", "trial"
    ],
    
    # Temporal Types
    "Beginning": [
        "dawn", "spring", "birth", "creation", "origin", "seed", "foundation",
        "awakening", "emergence", "genesis", "first", "new", "young", "fresh"
    ],
    
    "Ending": [
        "dusk", "winter", "death", "destruction", "end", "harvest", "completion",
        "closure", "finale", "last", "old", "ancient", "final", "ultimate"
    ],
    
    # Quality Types
    "Light": [
        "sun", "star", "fire", "gold", "white", "pure", "holy", "divine",
        "bright", "clear", "truth", "wisdom", "hope", "glory", "radiant"
    ],
    
    "Dark": [
        "shadow", "night", "void", "black", "hidden", "secret", "forbidden",
        "cursed", "corrupt", "false", "ignorance", "despair", "shame", "mystery"
    ],
}

# Colors for plotting
CATEGORY_COLORS = {
    "Person": "#e41a1c",
    "Place": "#377eb8",
    "Object": "#4daf4a",
    "Abstract": "#984ea3",
    "Creation": "#ff7f00",
    "Destruction": "#a65628",
    "Transformation": "#f781bf",
    "Connection": "#999999",
    "Conflict": "#66c2a5",
    "Beginning": "#ffd92f",
    "Ending": "#8da0cb",
    "Light": "#ffffb3",
    "Dark": "#1b9e77",
}


def get_concept_to_category():
    """Create mapping from concept word to its category."""
    mapping = {}
    for category, words in SEMANTIC_CATEGORIES.items():
        for word in words:
            mapping[word] = category
    return mapping


def get_valid_concepts(enc, categories=None):
    """Get concepts that are single tokens in the tokenizer."""
    if categories is None:
        categories = SEMANTIC_CATEGORIES
    
    valid = {}
    for category, words in categories.items():
        valid_words = []
        for word in words:
            # Try with and without space prefix
            tokens1 = enc.encode(word)
            tokens2 = enc.encode(" " + word)
            
            if len(tokens1) == 1 or len(tokens2) == 1:
                valid_words.append(word)
        valid[category] = valid_words
    
    return valid


def load_model(ckpt_path, device='cuda'):
    """Load Walsh model from checkpoint."""
    from src.model import Walsh, WalshConfig, mHCWalsh
    
    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Get state dict
    state_dict = ckpt.get('model', ckpt)
    
    # Strip _orig_mod prefix if present (from torch.compile)
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            clean_state[k[10:]] = v
        else:
            clean_state[k] = v
    
    # Get config - prefer model_args for architecture, config for training settings
    model_args = ckpt.get('model_args', {})
    config_dict = ckpt.get('config', {})
    
    # Use model_args for architecture config (it has correct vocab_size)
    valid_fields = ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size', 
                    'bias', 'dropout', 'head_mixing', 'algebra', 'hash_embeddings']
    arch_config = {k: v for k, v in model_args.items() if k in valid_fields}
    
    # Set vocab size if not present
    if 'vocab_size' not in arch_config:
        arch_config['vocab_size'] = 50304  # Default for padded vocab
    
    config = WalshConfig(**arch_config)
    
    # Check if this is an mHC model
    use_mhc = config_dict.get('use_mhc', False)
    n_streams = config_dict.get('n_streams', 4)
    
    if use_mhc:
        print(f"Loading mHCWalsh with {n_streams} streams...")
        model = mHCWalsh(config, n_streams=n_streams)
    else:
        model = Walsh(config)
    
    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()
    
    print(f"Embedding shape: {model.tok_embeddings.weight.shape}")
    return model, config


def get_hidden_states(model, enc, word, layer=-1, device='cuda'):
    """Get hidden states for a word after processing through model."""
    # Encode word (try with space prefix for better tokenization)
    tokens = enc.encode(" " + word)
    if len(tokens) > 1:
        tokens = enc.encode(word)
    
    if len(tokens) != 1:
        return None
    
    token_tensor = torch.tensor([[tokens[0]]], device=device)
    
    with torch.no_grad():
        # Use the model's full forward pass with return_hidden=True
        # This works for both Walsh and mHCWalsh
        try:
            logits, loss, hidden = model(token_tensor, return_hidden=True)
            # hidden is [B, T, n_embd] for both model types after merging
            return hidden[0, 0].cpu().numpy()
        except TypeError:
            # Fallback for models without return_hidden support
            # Get embeddings
            h = model.tok_embeddings(token_tensor)
            
            # Process through layers
            freqs_cis = model.freqs_cis[:1]
            
            if layer == -1:
                # All layers
                for block in model.layers:
                    h = block(h, freqs_cis)
                h = model.norm(h)
            else:
                # Specific layer
                for i, block in enumerate(model.layers):
                    h = block(h, freqs_cis)
                    if i == layer:
                        break
    
            return h[0, 0].cpu().numpy()


def analyze_channel_patterns(concept_states, concept_to_category, hadamard_dim=32):
    """Analyze Hadamard channel activation patterns by category."""
    categories = list(set(concept_to_category.values()))
    n_blocks = list(concept_states.values())[0].shape[0] // hadamard_dim
    
    # Compute channel patterns per category
    category_patterns = {}
    for category in categories:
        cat_concepts = [c for c, cat in concept_to_category.items() if cat == category and c in concept_states]
        if len(cat_concepts) == 0:
            continue
            
        states = np.array([concept_states[c] for c in cat_concepts])
        # Reshape to [n_concepts, n_blocks, hadamard_dim]
        states_reshaped = states.reshape(len(cat_concepts), n_blocks, hadamard_dim)
        # Mean activation per channel
        channel_pattern = np.abs(states_reshaped).mean(axis=(0, 1))
        category_patterns[category] = channel_pattern
    
    return category_patterns


def plot_channel_heatmap(category_patterns, output_path='channel_analysis.png'):
    """Plot heatmap of channel activations by category."""
    categories = list(category_patterns.keys())
    hadamard_dim = len(list(category_patterns.values())[0])
    
    # Create activation matrix
    activation_matrix = np.array([category_patterns[cat] for cat in categories])
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Heatmap
    im = axes[0].imshow(activation_matrix, aspect='auto', cmap='viridis')
    axes[0].set_yticks(range(len(categories)))
    axes[0].set_yticklabels(categories, fontsize=8)
    axes[0].set_xlabel(f'Hadamard Channel (0-{hadamard_dim-1})')
    axes[0].set_ylabel('Category')
    axes[0].set_title('Mean Activation per Hadamard Channel by Semantic Category')
    plt.colorbar(im, ax=axes[0], label='Mean |activation|')
    
    # Line plot
    for i, category in enumerate(categories):
        color = CATEGORY_COLORS.get(category, f'C{i}')
        axes[1].plot(category_patterns[category], label=category, alpha=0.7, color=color)
    
    axes[1].set_xlabel(f'Hadamard Channel (0-{hadamard_dim-1})')
    axes[1].set_ylabel('Mean |activation|')
    axes[1].set_title('Channel Activation Profiles by Category')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_hadamard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved channel heatmap to {output_path.replace('.png', '_hadamard.png')}")


def plot_embedding_space(concept_2d, concept_to_category, output_path='channel_analysis.png'):
    """Plot 2D PCA of concept embeddings colored by category."""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot by category
    for category in SEMANTIC_CATEGORIES.keys():
        cat_concepts = [c for c, cat in concept_to_category.items() if cat == category and c in concept_2d]
        if len(cat_concepts) == 0:
            continue
            
        xs = [concept_2d[c][0] for c in cat_concepts]
        ys = [concept_2d[c][1] for c in cat_concepts]
        color = CATEGORY_COLORS.get(category, 'gray')
        
        ax.scatter(xs, ys, c=color, label=category, alpha=0.7, s=50)
        
        # Add labels
        for c, x, y in zip(cat_concepts, xs, ys):
            ax.annotate(c, (x, y), fontsize=6, alpha=0.7,
                       xytext=(2, 2), textcoords='offset points')
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Concept Embeddings in 2D Space\n(Colored by Semantic Category)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved embedding plot to {output_path}")


def compute_metrics(concept_states, concept_to_category):
    """Compute clustering quality metrics."""
    concepts = list(concept_states.keys())
    states = np.array([concept_states[c] for c in concepts])
    labels = [concept_to_category.get(c, 'unknown') for c in concepts]
    
    # Encode labels as integers
    unique_labels = list(set(labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    label_ints = [label_to_int[l] for l in labels]
    
    # Silhouette score
    if len(unique_labels) > 1:
        sil_score = silhouette_score(states, label_ints)
    else:
        sil_score = 0.0
    
    # Within vs between category distances
    within_dists = []
    between_dists = []
    
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i >= j:
                continue
            dist = np.linalg.norm(states[i] - states[j])
            if labels[i] == labels[j]:
                within_dists.append(dist)
            else:
                between_dists.append(dist)
    
    within_mean = np.mean(within_dists) if within_dists else 0
    between_mean = np.mean(between_dists) if between_dists else 0
    ratio = between_mean / (within_mean + 1e-8)
    
    return {
        'silhouette': sil_score,
        'within_dist': within_mean,
        'between_dist': between_mean,
        'distance_ratio': ratio,
        'n_concepts': len(concepts),
        'n_categories': len(unique_labels)
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze channel specialization in Walsh models')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='channel_analysis.png', help='Output path')
    parser.add_argument('--layer', type=int, default=-1, help='Layer to analyze (-1 = final)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--hadamard-dim', type=int, default=32, help='Hadamard dimension')
    args = parser.parse_args()
    
    # Load model
    model, config = load_model(args.ckpt, args.device)
    
    # Get tokenizer
    enc = tiktoken.get_encoding('gpt2')
    
    # Get valid concepts
    valid_categories = get_valid_concepts(enc)
    concept_to_category = {}
    for cat, words in valid_categories.items():
        for word in words:
            concept_to_category[word] = cat
    
    all_concepts = list(concept_to_category.keys())
    print(f"\nAnalyzing {len(all_concepts)} concepts across {len(valid_categories)} categories...")
    
    # Get hidden states
    concept_states = {}
    for concept in all_concepts:
        state = get_hidden_states(model, enc, concept, layer=args.layer, device=args.device)
        if state is not None:
            concept_states[concept] = state
    
    print(f"Got states for {len(concept_states)} concepts")
    
    # Analyze channel patterns
    category_patterns = analyze_channel_patterns(
        concept_states, concept_to_category, args.hadamard_dim
    )
    
    # Compute category similarity
    print("\n=== Category Channel Similarity (Cosine) ===")
    categories = list(category_patterns.keys())
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i >= j:
                continue
            v1, v2 = category_patterns[cat1], category_patterns[cat2]
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            if sim < 0.99:  # Only print if not essentially identical
                print(f"  {cat1[:12]:12s} - {cat2[:12]:12s}: {sim:.3f}")
    
    # PCA projection
    states_matrix = np.array([concept_states[c] for c in concept_states.keys()])
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states_matrix)
    concept_2d = {c: states_2d[i] for i, c in enumerate(concept_states.keys())}
    
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_[0]*100:.1f}%, {pca.explained_variance_ratio_[1]*100:.1f}%")
    
    # Compute metrics
    metrics = compute_metrics(concept_states, concept_to_category)
    print(f"\n=== Clustering Metrics ===")
    print(f"Silhouette Score: {metrics['silhouette']:.3f}")
    print(f"Within-category distance: {metrics['within_dist']:.3f}")
    print(f"Between-category distance: {metrics['between_dist']:.3f}")
    print(f"Distance ratio (higher = better): {metrics['distance_ratio']:.3f}")
    
    # Generate plots
    plot_channel_heatmap(category_patterns, args.output)
    plot_embedding_space(concept_2d, concept_to_category, args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
