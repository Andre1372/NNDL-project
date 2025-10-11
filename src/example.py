"""
Esempio di base per verificare l'installazione di PyTorch.
"""

def check_pytorch():
    """
    Verifica che PyTorch sia installato correttamente.
    """
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Creare un tensore semplice
        x = torch.rand(3, 3)
        print(f"\nTensore di esempio:\n{x}")
        
        return True
    except ImportError:
        print("PyTorch non è installato. Eseguire: pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    check_pytorch()
