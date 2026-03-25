class Config:
    # Model hyperparameters
    hidden_dim = 256  # Reduced from 512 for better memory
    num_layers = 2
    dropout = 0.3
    learning_rate = 0.001
    batch_size = 512  # Reduced from 1024
    num_epochs = 200
    negative_sample_size = 128  # Reduced from 256
    eval_batch_size = 512
    patience = 15

    # GNN specific
    gnn_type = 'gcn'  # 'gcn', 'gat', 'sage'
    
    # Dataset list for automatic sequential execution
    datasets = ['FB15k-237', 'WN18', 'WN18RR', 'YAGO3-10', 'FB15K']
    
    # Data directory
    data_dir = '/home/user/23h1710_KGC/GNN_Based_relation_pre/data'
    
    # Multi-GPU settings
    use_multi_gpu = True
    gpu_ids = [0, 1]
    
    # Training improvements
    label_smoothing = 0.1
    gradient_clip = 1.0
    warmup_epochs = 5
    
    # Memory optimization for large datasets
    yago3_10_hidden_dim = 128  # Even smaller for YAGO3-10
    yago3_10_batch_size = 128  # Much smaller for YAGO3-10
    yago3_10_negative_sample_size = 32  # Smaller negative samples