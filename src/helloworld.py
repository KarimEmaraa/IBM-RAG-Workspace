import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for containers
import matplotlib.pyplot as plt
import lightning as L
import langchain
import optuna
import mlflow
import jupyter

print("=== Library Check ===")
print(f"NumPy Version:        {np.__version__}")
print(f"Matplotlib Version:   {matplotlib.__version__}")
print(f"PyTorch Version:      {torch.__version__}")
print(f"CUDA Available:       {torch.cuda.is_available()} (Expected: False on CPU)")
print(f"Lightning Version:    {L.__version__}")
print(f"LangChain Version:    {langchain.__version__}")
print(f"Optuna Version:       {optuna.__version__}")
print(f"MLflow Version:       {mlflow.__version__}")

# Validate array operations and plotting
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Headless Test Plot")
plt.savefig("test_plot.png")
print("\nPlot saved successfully to test_plot.png!")