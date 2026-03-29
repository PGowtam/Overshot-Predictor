import os

# Project Root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Subdirectories
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
EXECUTION_DIR = os.path.join(ROOT_DIR, 'execution')
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# Create Logs dir if not exists
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Models Root (User specified Trader/models)
MODELS_ROOT = os.path.join(ROOT_DIR, 'models') 

# Model Specific Paths
M_PPO_PATH = os.path.join(MODELS_ROOT, 'models_ppo', 'final_model')
M_DQN_PATH = os.path.join(MODELS_ROOT, 'models_dqn', 'final_model_dqn')
M_QRDQN_PATH = os.path.join(MODELS_ROOT, 'models_qrdqn', 'final_model_qrdqn')
M_RECURRENT_PATH = os.path.join(MODELS_ROOT, 'models_recurrent', 'final_model_recurrent')
M_TRANSFORMER_PATH = os.path.join(MODELS_ROOT, 'models_transformer', 'final_model_transformer')

MODEL_DIR = MODELS_ROOT # Alias for predictors looking for root

# Outcome Models & Scalers
# We copied them to Trader/models
M_CATBOOST_BINARY = os.path.join(MODEL_DIR, 'binary_model.pkl')
M_CATBOOST_MULTI = os.path.join(MODEL_DIR, 'stacked_model.pkl')

M_BINARY_SCALER = os.path.join(MODEL_DIR, 'binary_scaler.pkl')
M_MULTI_SCALER = os.path.join(MODEL_DIR, 'scaler.pkl')
