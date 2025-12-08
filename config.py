class Config:
    # MCTS parameters
    NUM_SIMULATIONS = 50            # number of MCTS rollouts per move
    C_PUCT = 1.4                    # exploration vs exploitation constant (AlphaZero uses ~1.5)

    # Dirichlet noise for root exploration (during self-play)
    DIRICHLET_ALPHA = 0.3           # AlphaZero uses 0.3 for chess
    EPSILON = 0.25                  # Mix 75% NN policy + 25% noise

    # Temperature for move selection during self-play
    TEMP_THRESHOLD = 15             # Use temp=1 for first N moves, then temp→0

    # Inference batching
    INFER_BATCH_SIZE = 16           # batch size for model inference in MCTS

    # Neural network training
    LEARNING_RATE = 2e-4            # Lower learning rate for stability
    WEIGHT_DECAY = 1e-4             # L2 regularization
    BATCH_SIZE = 64                 # Larger batch size
    TRAIN_EPOCHS = 3                # Number of epochs per training iteration

    # Network architecture
    INPUT_CHANNELS = 17             # Improved state representation
    RESIDUAL_BLOCKS = 5            # More blocks for better performance
    CHANNELS = 128                  # Wider network

    # Self-play
    NUM_SELFPLAY_GAMES = 20         # Games per iteration (increase for better learning)
    MAX_GAME_LENGTH = 200           # Maximum moves per game

    # Training iterations
    NUM_ITERATIONS = 4             # Total training iterations

    # Replay buffer
    REPLAY_BUFFER_SIZE = 100000     # Keep last N game positions
    MIN_BUFFER_SIZE = 10000         # Start training after collecting this many examples

    # Validation (early stopping)
    NUM_VALIDATION_GAMES = 20       # More games for better validation
    VALIDATION_INTERVAL = 5         # Validate every N iterations
    EARLY_STOPPING_PATIENCE = 20    # Stop if no improvement for N validations

    # Data augmentation (board symmetries)
    USE_AUGMENT_SYMMETRIES = False

    # Logging / saving
    LOG_DIR = "./logs"
    MODEL_DIR = "./models"
    GAMES_DIR = "./games"
    SAVE_INTERVAL = 2               # Save model every N iterations
    LOG_INTERVAL = 1                # Log to tensorboard every N iterations

    # GPU optimizations
    USE_TORCHSCRIPT = False         # Disabled for easier debugging
    NUM_WORKERS = 4                 # DataLoader workers

    # Move encoding
    # 64*64 = 4096 normal moves (from_square * 64 + to_square)
    # 64*9 = 576 underpromotions (knight, bishop, rook for each starting square)
    # Total: 4672 possible moves
    POLICY_OUTPUT_SIZE = 4672

    # EXPERIMENT_NAME = "ches"

    INIT_MODEL_PATH="./models/best_model.pt"

    ARENA_GAMES = 20
    ARENA_THRESHOLD = 0.55
