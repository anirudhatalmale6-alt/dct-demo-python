"""
Discrete Cosine Transform (DCT) - How It Works
================================================
Demonstrates DCT operation on signals and images:
1. 1D DCT on a simple signal
2. 2D DCT on an image (JPEG-style compression demo)
3. Step-by-step visualization of how DCT compresses data
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# PART 1: 1D DCT - Transform a signal
# ============================================================

def dct_1d(signal):
    """Compute 1D DCT-II (the standard DCT used in JPEG/MP3)."""
    N = len(signal)
    result = np.zeros(N)
    for k in range(N):
        total = 0.0
        for n in range(N):
            total += signal[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        result[k] = alpha * total
    return result


def idct_1d(coefficients):
    """Compute inverse 1D DCT-II to reconstruct the signal."""
    N = len(coefficients)
    result = np.zeros(N)
    for n in range(N):
        total = 0.0
        for k in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            total += alpha * coefficients[k] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        result[n] = total
    return result


def demo_1d_dct():
    """Show how 1D DCT transforms a signal into frequency components."""
    print("=" * 60)
    print("PART 1: 1D Discrete Cosine Transform")
    print("=" * 60)

    # Create a signal: mix of two cosine waves + noise
    N = 64
    t = np.arange(N)
    signal = 3 * np.cos(2 * np.pi * 2 * t / N) + 1.5 * np.cos(2 * np.pi * 7 * t / N)
    noisy_signal = signal + 0.5 * np.random.randn(N)

    # Apply DCT
    dct_coeffs = dct_1d(noisy_signal)

    # Compress: keep only top K coefficients
    K = 10
    compressed = dct_coeffs.copy()
    sorted_indices = np.argsort(np.abs(compressed))
    compressed[sorted_indices[:-K]] = 0  # zero out smallest coefficients

    # Reconstruct
    reconstructed = idct_1d(compressed)

    print(f"Original signal:    {N} samples")
    print(f"DCT coefficients:   {N} total")
    print(f"Kept coefficients:  {K} (compression ratio {N/K:.1f}x)")
    print(f"Reconstruction MSE: {np.mean((noisy_signal - reconstructed)**2):.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("1D DCT Operation", fontsize=14, fontweight='bold')

    axes[0, 0].plot(t, noisy_signal, 'b-', alpha=0.7)
    axes[0, 0].set_title("Original Signal (with noise)")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].stem(np.arange(N), dct_coeffs, markerfmt='ro', linefmt='r-', basefmt='k-')
    axes[0, 1].set_title("DCT Coefficients (Frequency Domain)")
    axes[0, 1].set_xlabel("Frequency Index")
    axes[0, 1].set_ylabel("Magnitude")

    axes[1, 0].stem(np.arange(N), compressed, markerfmt='go', linefmt='g-', basefmt='k-')
    axes[1, 0].set_title(f"After Compression (keep top {K})")
    axes[1, 0].set_xlabel("Frequency Index")
    axes[1, 0].set_ylabel("Magnitude")

    axes[1, 1].plot(t, noisy_signal, 'b-', alpha=0.4, label='Original')
    axes[1, 1].plot(t, reconstructed, 'r-', linewidth=2, label='Reconstructed')
    axes[1, 1].set_title("Reconstruction from Compressed DCT")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("dct_1d_demo.png", dpi=150)
    plt.close()
    print("Plot saved: dct_1d_demo.png\n")


# ============================================================
# PART 2: 2D DCT - Image compression (JPEG-style)
# ============================================================

def dct_2d(block):
    """Compute 2D DCT on an NxN block by applying 1D DCT to rows then columns."""
    N = block.shape[0]
    temp = np.zeros_like(block, dtype=float)
    result = np.zeros_like(block, dtype=float)

    # DCT on each row
    for i in range(N):
        temp[i, :] = dct_1d(block[i, :])

    # DCT on each column
    for j in range(N):
        result[:, j] = dct_1d(temp[:, j])

    return result


def idct_2d(block):
    """Compute inverse 2D DCT."""
    N = block.shape[0]
    temp = np.zeros_like(block, dtype=float)
    result = np.zeros_like(block, dtype=float)

    # IDCT on each column
    for j in range(N):
        temp[:, j] = idct_1d(block[:, j])

    # IDCT on each row
    for i in range(N):
        result[i, :] = idct_1d(temp[i, :])

    return result


def demo_2d_dct():
    """Show how 2D DCT works on image blocks (JPEG compression principle)."""
    print("=" * 60)
    print("PART 2: 2D DCT - Image Compression (JPEG-style)")
    print("=" * 60)

    # Create a synthetic 64x64 grayscale image
    size = 64
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    image = (
        128
        + 50 * np.sin(2 * np.pi * x / 16)
        + 30 * np.cos(2 * np.pi * y / 8)
        + 20 * np.sin(2 * np.pi * (x + y) / 32)
    ).astype(float)

    # Process in 8x8 blocks (like JPEG)
    block_size = 8
    quality_levels = [64, 16, 6, 2]  # number of coefficients to keep per block

    results = {}
    for keep in quality_levels:
        compressed_image = np.zeros_like(image)
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                dct_block = dct_2d(block)

                # Zigzag-style: keep top-left coefficients (low frequencies)
                mask = np.zeros((block_size, block_size))
                count = 0
                for diag in range(2 * block_size - 1):
                    for r in range(max(0, diag - block_size + 1), min(diag + 1, block_size)):
                        c = diag - r
                        if c < block_size:
                            mask[r, c] = 1
                            count += 1
                            if count >= keep:
                                break
                    if count >= keep:
                        break

                compressed_block = dct_block * mask
                compressed_image[i:i+block_size, j:j+block_size] = idct_2d(compressed_block)

        mse = np.mean((image - compressed_image) ** 2)
        ratio = (block_size * block_size) / keep
        results[keep] = (compressed_image, mse, ratio)
        print(f"Keep {keep:2d}/{block_size**2} coefficients | "
              f"Compression: {ratio:.1f}x | MSE: {mse:.2f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("2D DCT Image Compression (JPEG-style, 8x8 blocks)", fontsize=14, fontweight='bold')

    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Show DCT of one block
    sample_block = image[0:8, 0:8]
    dct_sample = dct_2d(sample_block)
    axes[0, 1].imshow(np.log1p(np.abs(dct_sample)), cmap='hot')
    axes[0, 1].set_title("DCT of 8x8 Block (log scale)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(sample_block, cmap='gray')
    axes[0, 2].set_title("Original 8x8 Block")
    axes[0, 2].axis('off')

    for idx, keep in enumerate([16, 6, 2]):
        img, mse, ratio = results[keep]
        axes[1, idx].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[1, idx].set_title(f"Keep {keep}/64 ({ratio:.0f}x compression)\nMSE: {mse:.1f}")
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig("dct_2d_demo.png", dpi=150)
    plt.close()
    print("Plot saved: dct_2d_demo.png\n")


# ============================================================
# PART 3: Step-by-step DCT explanation
# ============================================================

def demo_dct_basis():
    """Visualize the DCT basis functions to understand HOW it works."""
    print("=" * 60)
    print("PART 3: DCT Basis Functions - The Building Blocks")
    print("=" * 60)
    print()
    print("How DCT works:")
    print("-" * 40)
    print("1. DCT decomposes a signal into a sum of cosine waves")
    print("   at different frequencies.")
    print("2. Low-frequency components capture the overall shape.")
    print("3. High-frequency components capture fine details.")
    print("4. For compression: discard small high-frequency")
    print("   coefficients (human eye/ear can't notice them).")
    print("5. This is why JPEG, MP3, and AAC all use DCT!")
    print()

    # Show 1D basis functions
    N = 32
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    fig.suptitle("DCT Basis Functions (each frequency component)", fontsize=14, fontweight='bold')

    for k in range(16):
        row, col = k // 4, k % 4
        n = np.arange(N)
        basis = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        axes[row, col].plot(n, basis, 'b-', linewidth=1.5)
        axes[row, col].fill_between(n, basis, alpha=0.3)
        axes[row, col].set_title(f"k={k} (freq={k})", fontsize=10)
        axes[row, col].set_ylim(-1.3, 1.3)
        axes[row, col].set_xticks([])
        if col == 0:
            axes[row, col].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig("dct_basis_functions.png", dpi=150)
    plt.close()
    print("Plot saved: dct_basis_functions.png")

    # Show 2D basis patterns (8x8, like JPEG)
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    fig.suptitle("2D DCT Basis Patterns (8x8, used in JPEG)", fontsize=13, fontweight='bold')

    for u in range(8):
        for v in range(8):
            basis_2d = np.zeros((8, 8))
            for x in range(8):
                for y in range(8):
                    basis_2d[x, y] = (
                        np.cos(np.pi * u * (2 * x + 1) / 16)
                        * np.cos(np.pi * v * (2 * y + 1) / 16)
                    )
            axes[u, v].imshow(basis_2d, cmap='RdBu', vmin=-1, vmax=1)
            axes[u, v].axis('off')

    plt.tight_layout()
    plt.savefig("dct_2d_basis.png", dpi=150)
    plt.close()
    print("Plot saved: dct_2d_basis.png\n")


# ============================================================
# RUN ALL DEMOS
# ============================================================

if __name__ == "__main__":
    print()
    print("DCT (Discrete Cosine Transform) - Complete Demo")
    print("=" * 60)
    print()

    np.random.seed(42)

    demo_1d_dct()
    demo_2d_dct()
    demo_dct_basis()

    print("=" * 60)
    print("Summary:")
    print("-" * 60)
    print("- DCT converts spatial/time data into frequency components")
    print("- Low frequencies = overall shape, High frequencies = details")
    print("- Compression works by discarding small high-freq coefficients")
    print("- Used in JPEG (images), MP3/AAC (audio), MPEG (video)")
    print("- DCT-II is the most common variant (what we demonstrated)")
    print("=" * 60)
