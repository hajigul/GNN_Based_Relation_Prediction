import os
import argparse
import torch
import torch.optim as optim
import numpy as np
import random
from config import Config
from utils import load_dataset, build_graph, create_dataloader
from models import KGModel
from train import train_epoch, evaluate

# Set memory optimization for CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_gpu_memory():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def train_single_dataset(dataset_name, gnn_type, args):
    """Train model on a single dataset"""
    print("\n" + "="*80)
    print(f"Training on {dataset_name} with {gnn_type.upper()}")
    print("="*80)
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Adjust hyperparameters based on dataset size
    if dataset_name == 'YAGO3-10':
        hidden_dim = args.yago3_10_hidden_dim
        batch_size = args.yago3_10_batch_size
        negative_sample_size = args.yago3_10_negative_sample_size
        print(f"Using optimized settings for YAGO3-10: hidden_dim={hidden_dim}, batch_size={batch_size}")
    else:
        hidden_dim = args.hidden_dim
        batch_size = args.batch_size
        negative_sample_size = args.negative_sample_size
    
    # Setup device - Force multi-GPU usage
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        device = torch.device('cuda')
        print(f"Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    set_seed()

    # Load data
    print(f"\nLoading {dataset_name}...")
    num_entities, num_relations, train_triples, valid_triples, test_triples, _, _ = load_dataset(dataset_name, args.data_dir)
    print(f"Entities: {num_entities}, Relations: {num_relations}")
    print(f"Train: {len(train_triples)}, Valid: {len(valid_triples)}, Test: {len(test_triples)}")

    # Build graph
    edge_index, edge_type = build_graph(train_triples, num_entities)
    edge_index = edge_index.to(device)
    print(f"Graph edges: {edge_index.size(1)}")

    # Create dataloaders
    train_loader = create_dataloader(train_triples, num_entities, num_relations, batch_size, shuffle=True)
    valid_loader = create_dataloader(valid_triples, num_entities, num_relations, batch_size, shuffle=False)
    test_loader = create_dataloader(test_triples, num_entities, num_relations, batch_size, shuffle=False)

    # Initialize model
    model = KGModel(num_entities, num_relations, hidden_dim, args.num_layers, 
                   args.dropout, gnn_type, device)
    
    # Multi-GPU - Move model to GPU before wrapping
    model = model.to(device)
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        # Use DataParallel with specified devices
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        print("✓ Model wrapped with DataParallel for multi-GPU training")
        print(f"  - GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU 1: {torch.cuda.get_device_name(1)}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Training loop
    best_valid_mrr = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        # Clear cache periodically
        if epoch % 5 == 0:
            clear_gpu_memory()
        
        # Get entity embeddings
        if isinstance(model, torch.nn.DataParallel):
            entity_embs = model.module.get_entity_embeddings(edge_index)
        else:
            entity_embs = model.get_entity_embeddings(edge_index)
        
        # Train
        loss = train_epoch(model, train_loader, optimizer, num_relations, 
                          negative_sample_size, device, entity_embs.detach(), 
                          epoch, args.warmup_epochs)
        
        # Evaluate every epoch
        if isinstance(model, torch.nn.DataParallel):
            entity_embs = model.module.get_entity_embeddings(edge_index)
            mr, mrr, hits1, hits3, hits10 = evaluate(model.module, valid_loader, 
                                                     entity_embs, num_relations, device)
        else:
            entity_embs = model.get_entity_embeddings(edge_index)
            mr, mrr, hits1, hits3, hits10 = evaluate(model, valid_loader, 
                                                     entity_embs, num_relations, device)
        
        # Display with decimal format
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"Loss: {loss:.4f}")
        print(f"Valid - MR: {mr:.2f}, MRR: {mrr:.4f}, H@1: {hits1:.4f}, H@3: {hits3:.4f}, H@10: {hits10:.4f}")
        
        # Show GPU memory usage
        if epoch % 10 == 0:
            print(f"GPU Memory - GPU0: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB, GPU1: {torch.cuda.memory_allocated(1)/1024**3:.2f}GB")
        
        # Learning rate scheduling
        scheduler.step(mrr)
        
        # Save best model
        if mrr > best_valid_mrr:
            best_valid_mrr = mrr
            best_epoch = epoch
            patience_counter = 0
            save_path = f'best_model_{gnn_type}_{dataset_name}.pt'
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"✓ New best model saved (MRR: {mrr:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Test best model
    print(f"\nLoading best model from epoch {best_epoch}...")
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(torch.load(f'best_model_{gnn_type}_{dataset_name}.pt'))
    else:
        model.load_state_dict(torch.load(f'best_model_{gnn_type}_{dataset_name}.pt'))
    
    # Final test
    if isinstance(model, torch.nn.DataParallel):
        entity_embs = model.module.get_entity_embeddings(edge_index)
        test_mr, test_mrr, test_hits1, test_hits3, test_hits10 = evaluate(
            model.module, test_loader, entity_embs, num_relations, device)
    else:
        entity_embs = model.get_entity_embeddings(edge_index)
        test_mr, test_mrr, test_hits1, test_hits3, test_hits10 = evaluate(
            model, test_loader, entity_embs, num_relations, device)
    
    # Return results
    results = {
        'dataset': dataset_name,
        'model': gnn_type,
        'MR': test_mr,
        'MRR': test_mrr,
        'Hits@1': test_hits1,
        'Hits@3': test_hits3,
        'Hits@10': test_hits10
    }
    
    # Clear memory after training
    clear_gpu_memory()
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default=Config.datasets, help='Datasets to train on')
    parser.add_argument('--gnn_type', type=str, default=Config.gnn_type, choices=['gcn', 'gat', 'sage'])
    parser.add_argument('--data_dir', type=str, default=Config.data_dir)
    parser.add_argument('--hidden_dim', type=int, default=Config.hidden_dim)
    parser.add_argument('--num_layers', type=int, default=Config.num_layers)
    parser.add_argument('--dropout', type=float, default=Config.dropout)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--num_epochs', type=int, default=Config.num_epochs)
    parser.add_argument('--learning_rate', type=float, default=Config.learning_rate)
    parser.add_argument('--negative_sample_size', type=int, default=Config.negative_sample_size)
    parser.add_argument('--patience', type=int, default=Config.patience)
    parser.add_argument('--warmup_epochs', type=int, default=Config.warmup_epochs)
    parser.add_argument('--use_multi_gpu', action='store_true', default=Config.use_multi_gpu)
    
    # YAGO3-10 specific parameters
    parser.add_argument('--yago3_10_hidden_dim', type=int, default=Config.yago3_10_hidden_dim)
    parser.add_argument('--yago3_10_batch_size', type=int, default=Config.yago3_10_batch_size)
    parser.add_argument('--yago3_10_negative_sample_size', type=int, default=Config.yago3_10_negative_sample_size)
    
    args = parser.parse_args()
    
    # Verify GPU availability
    if args.use_multi_gpu:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Store all results
    all_results = []
    
    # Train on each dataset sequentially
    for dataset_name in args.datasets:
        try:
            results = train_single_dataset(dataset_name, args.gnn_type, args)
            all_results.append(results)
            
            # Print results with decimal format
            print("\n" + "="*60)
            print(f"RESULTS FOR {dataset_name.upper()} WITH {args.gnn_type.upper()}:")
            print("="*60)
            print(f"MR (Mean Rank): {results['MR']:.2f}")
            print(f"MRR (Mean Reciprocal Rank): {results['MRR']:.4f}")
            print(f"Hits@1: {results['Hits@1']:.4f}")
            print(f"Hits@3: {results['Hits@3']:.4f}")
            print(f"Hits@10: {results['Hits@10']:.4f}")
            print("="*60)
            
        except Exception as e:
            print(f"\n Error training on {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Clear memory on error
            clear_gpu_memory()
            continue
    
    # Print summary of all results with decimal format
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY OF ALL RESULTS")
        print("="*80)
        print(f"{'Dataset':<15} {'MR':<10} {'MRR':<12} {'Hits@1':<10} {'Hits@3':<10} {'Hits@10':<10}")
        print("-"*80)
        for res in all_results:
            print(f"{res['dataset']:<15} {res['MR']:<10.2f} {res['MRR']:<12.4f} "
                  f"{res['Hits@1']:<10.4f} {res['Hits@3']:<10.4f} {res['Hits@10']:<10.4f}")
        print("="*80)
        
        # Save results to file
        with open(f'results_{args.gnn_type}.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"RESULTS FOR {args.gnn_type.upper()} MODEL\n")
            f.write("="*80 + "\n\n")
            for res in all_results:
                f.write(f"{res['dataset']}:\n")
                f.write(f"  MR: {res['MR']:.2f}\n")
                f.write(f"  MRR: {res['MRR']:.4f}\n")
                f.write(f"  Hits@1: {res['Hits@1']:.4f}\n")
                f.write(f"  Hits@3: {res['Hits@3']:.4f}\n")
                f.write(f"  Hits@10: {res['Hits@10']:.4f}\n\n")
        print(f"\n✓ Results saved to results_{args.gnn_type}.txt")

if __name__ == "__main__":
    main()