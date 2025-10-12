"""
Esempio avanzato per verificare l'installazione di PyTorch e le capacità hardware (GPU o CPU o MPS).
Compatibile con:
 - GPU NVIDIA (CUDA)
 - GPU Apple Silicon (MPS)
 - CPU generica
"""

import time

try:
    import torch
    import platform
    import psutil

    print(f"PyTorch version: {torch.__version__}\n")

    # Selezione automatica del device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple GPU (MPS)"
    else:
        device = torch.device("cpu")
        device_name = platform.processor() or "CPU"

    if device.type == "cuda":
        # Informazioni sulle GPU
        num_gpus = torch.cuda.device_count()
        print(f"Numero di GPU disponibili: {num_gpus}\n")

        for i in range(num_gpus):
            prop = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {prop.name}")
            print(f"  Memoria totale: {prop.total_memory / (1024 ** 3):.2f} GB")
            print(f"  Multiprocessori: {prop.multi_processor_count}")
            print(f"  Capabilities: {prop.major}.{prop.minor}\n")

    elif device.type == "mps":
        print("== Informazioni GPU (MPS / Apple Silicon) ==")
        print("MPS è disponibile e attivo tramite Metal Performance Shaders.\n")

    else:
        # Informazioni sulla CPU
        print("== Informazioni CPU ==")
        print(f"Nome CPU: {platform.processor() or 'N/D'}")
        print(f"Numero di core fisici: {psutil.cpu_count(logical=False)}")
        print(f"Numero di thread logici: {psutil.cpu_count(logical=True)}")
        print(f"Frequenza CPU: {psutil.cpu_freq().current / 1000:.2f} GHz\n")

    # Test di calcolo su device selezionato
    print("== Test di moltiplicazione matriciale ==")
    size = 5000  # più piccolo per evitare crash su CPU
    multiplications = 10
    x = torch.rand(size, size, device=device)
    y = torch.rand(size, size, device=device)

    total_start = time.time()
    flops_per_op = 2 * size**3 - size**2  # FLOPs per moltiplicazione
    flops_list = []

    for i in range(0, multiplications):
        start = time.time()
        z = x @ y
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        elapsed = end - start
        flops = flops_per_op / elapsed
        flops_list.append(flops)

    total_end = time.time()
    avg_flops = sum(flops_list) / len(flops_list)

    print(f"Tempo totale per {multiplications} moltiplicazioni tra matrici di {size}x{size} elementi: {total_end - total_start:.4f} secondi")
    print(f"FLOPS medi: {avg_flops/1e12:.2f} TFLOPS")

except ImportError:
    print("PyTorch non è installato")
