"""NewsRepLab: an educational feasibility test, not a classification experiment.

Research question: do pretrained text representations improve news classification
over TF-IDF when only a small amount of labeled training data is available?
Eventually, we will compare TF-IDF + Logistic Regression with frozen MiniLM
embeddings + Logistic Regression. This script checks only the prerequisites:
the Python/GPU environment, access to AG News, and encoding five news articles.

A dataset pairs input text X with a target category y. MiniLM converts each text
into a numeric feature vector with 384 coordinates. Those coordinates are NOT
category predictions. A separate classifier would need to learn how features
relate to labels; no such classifier is trained or evaluated here.

Run this file in Spyder using the existing icsi436-p environment. The first run
can download dataset/model files. The dataset cache uses a short absolute path;
project paths are derived from this file, so Spyder's working directory does not
need to be the project root. The complete dataset
is loaded through Hugging Face Datasets, but only five training texts are encoded.
No full-dataset preprocessing, fine-tuning, reports, or evaluation is performed.
"""

from pathlib import Path
import os
import random
import sys
import time

# These are the four direct third-party dependencies of this feasibility test.
# A failed import often means Spyder is using a different Python environment.
# Preserve the original exception so the exact missing/broken dependency is visible.
try:
    import numpy as np
    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
except Exception:
    print("Dependency import failed. Python executable:", sys.executable)
    print(r"Expected environment: D:\Anaconda\envs\icsi436-p\python.exe")
    print("Check Spyder's interpreter and restart its kernel before changing packages.")
    raise


# __file__ identifies this script even when Spyder starts in another directory.
# parents[0] is src; parents[1] is the project root. Model downloads stay in an
# ignored project folder rather than being mistaken for source code or results.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Datasets builds long cache and lock-file names. A deeply nested project path
# can make those names exceed Windows limits (WinError 206). Use a short external
# directory by default; HF_DATASETS_CACHE can select another short absolute path.
# Read the environment variable ourselves and pass cache_dir explicitly below:
# this also honors a setting made in Spyder after datasets was already imported.
DATASET_CACHE = Path(os.environ.get("HF_DATASETS_CACHE") or r"D:\hf_cache\datasets").expanduser()
if not DATASET_CACHE.is_absolute():
    raise ValueError("HF_DATASETS_CACHE must be a short absolute path, such as D:/hf_cache/datasets.")

MODEL_CACHE = PROJECT_ROOT / ".cache" / "sentence_transformers"
DATASET_NAME = "fancyzhx/ag_news"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_COUNT = 5
EXPECTED_DIMENSION = 384
RANDOM_SEED = 436


def verify_environment():
    """Print environment details and select a usable computation device."""
    print("\nPart A - Environment verification")
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if not cuda_available:
        print("GPU model / GPU memory: unavailable to this PyTorch process.")
        print("WARNING: CUDA is unavailable; continuing on CPU.")
        return "cpu"

    # Availability alone does not prove a CUDA kernel can execute. A tiny matrix
    # multiplication checks real GPU work without allocating a large tensor.
    # Two 2-by-2 matrices of ones multiply to a 2-by-2 matrix of twos.
    try:
        device = "cuda:0"
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        gib = 1024 ** 3
        print("GPU model:", properties.name)
        print(f"GPU physical memory: {properties.total_memory / gib:.2f} GiB")
        print(f"GPU free / total memory: {free_bytes / gib:.2f} / {total_bytes / gib:.2f} GiB")
        print(f"PyTorch allocated memory: {torch.cuda.memory_allocated(device) / gib:.4f} GiB")
        matrix = torch.ones((2, 2), device=device)
        product = matrix @ matrix
        # CUDA normally queues work asynchronously. Synchronization makes sure
        # the calculation finishes before we announce that it passed.
        torch.cuda.synchronize(device)
        if not torch.allclose(product, torch.full_like(product, 2.0)):
            raise RuntimeError("The CUDA tensor operation returned unexpected values.")
        print("CUDA tensor operation passed:", product.cpu().tolist())
        return device
    except (RuntimeError, AssertionError) as error:
        # A failed GPU probe does not prevent a CPU feasibility check. Report the
        # exact failure, and make the fallback explicit instead of hiding it.
        print(f"WARNING: CUDA probe failed: {type(error).__name__}: {error}")
        print("Check the active interpreter, GPU driver, and available GPU memory.")
        print("Continuing on CPU; this run does not validate GPU feasibility.")
        return "cpu"


def inspect_dataset():
    """Load AG News and return five unmodified training texts and their labels."""
    print("\nPart B - AG News dataset")
    print("Dataset source:", DATASET_NAME)
    print("Dataset cache:", DATASET_CACHE)
    # Create missing parent folders as well. An existing cache is reused safely.
    # The printed path, directory creation, and loader all use the same value.
    DATASET_CACHE.mkdir(parents=True, exist_ok=True)

    # load_dataset returns a DatasetDict: split names map to table-like datasets.
    # Loading downloads/prepares the dataset on first use; it is not ML training.
    # We do not call map(), tokenize the full dataset, or extract all its texts.
    dataset = load_dataset(DATASET_NAME, cache_dir=str(DATASET_CACHE))
    print("Dataset split names:", list(dataset.keys()))
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Expected both train and test splits in AG News.")
    train_data = dataset["train"]
    test_data = dataset["test"]
    print("Number of training examples:", len(train_data))
    print("Number of test examples:", len(test_data))

    # ClassLabel metadata translates integer targets into readable category names.
    # The numbers identify categories; a larger label does not mean a better class.
    class_names = train_data.features["label"].names
    print("Class labels:", dict(enumerate(class_names)))
    if len(train_data) < SAMPLE_COUNT:
        raise ValueError("The training split contains fewer than five examples.")

    # A local seeded random generator selects repeatable row indices without
    # shuffling or modifying the full dataset. This tiny sample is illustrative,
    # not a balanced training subset or a statistically meaningful evaluation.
    indices = random.Random(RANDOM_SEED).sample(range(len(train_data)), SAMPLE_COUNT)
    texts = []
    labels = []
    for index in indices:
        example = train_data[index]
        article = example["text"]
        label = example["label"]
        if not isinstance(article, str) or not article.strip():
            raise ValueError(f"Training row {index} does not contain a nonempty text.")
        texts.append(article)
        labels.append(label)

        # Whitespace cleanup and truncation are ONLY for console display. The
        # original article, not this preview, is passed to the pretrained model.
        preview = " ".join(article.split())
        if len(preview) > 180:
            preview = preview[:177] + "..."
        print(f"\nTraining row {index} | label {label}: {class_names[label]}")
        print(preview)

    print("\nX = input articles; later X will be a numeric feature matrix.")
    print("y = known news categories, used to train/evaluate a future classifier.")
    print("These labels are displayed for understanding; MiniLM receives only text.")
    return texts, labels


def extract_embeddings(texts, device):
    """Load a frozen pretrained encoder and transform only the selected texts."""
    print("\nPart C - Pretrained MiniLM feature extraction")
    print("Selected device:", device)
    print("Number of input articles:", len(texts))
    print("Pretrained model:", MODEL_NAME)
    print("Model cache:", MODEL_CACHE)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    # Model loading includes download time on the first run and device transfer.
    # perf_counter measures elapsed wall-clock seconds. Synchronizing CUDA at
    # timing boundaries prevents queued GPU work from escaping the measurement.
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    loading_start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device=device, cache_folder=str(MODEL_CACHE))
    model.eval()
    # eval() disables training-time behavior such as dropout; it does not itself
    # freeze weights. requires_grad_(False) makes our frozen-feature intent clear.
    # There is no optimizer, backward pass, or weight update anywhere in this test.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    loading_seconds = time.perf_counter() - loading_start
    print("Model loaded successfully on:", model.device)
    print(f"Model loading time: {loading_seconds:.3f} seconds")
    print("Maximum input length (tokens):", model.max_seq_length)
    print("Long texts may be truncated by the model's tokenizer.")

    # Each row of the output describes one article in the same input order.
    # batch_size=5 processes this tiny sample together. convert_to_numpy returns
    # a CPU NumPy array, even when the neural network ran on the GPU.
    # normalize_embeddings=True scales each vector to unit length; it does not
    # turn its coordinates into probabilities. Tokenization happens on the CPU.
    # inference_mode avoids gradient tracking, reducing inference memory use.
    encoding_start = time.perf_counter()
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=SAMPLE_COUNT,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    encoding_seconds = time.perf_counter() - encoding_start
    print(f"Embedding extraction time: {encoding_seconds:.3f} seconds")
    print("Timing includes tokenization, inference, and conversion to a CPU array.")
    print("This first tiny batch is a feasibility check, not a speed benchmark.")
    print("These vectors are pretrained representations, not classification predictions.")
    return embeddings


def validate_embeddings(texts, labels, embeddings):
    """Check the feature matrix before declaring the feasibility test successful."""
    print("\nPart D - Basic validation")
    # The expected matrix has n rows (articles) and 384 columns (features).
    # Explicit checks remain active even when Python is run with optimization.
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Expected a two-dimensional NumPy embedding array.")
    if embeddings.shape[0] != len(texts) or len(texts) != len(labels):
        raise ValueError("Input text, label, and embedding counts do not match.")
    if embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(f"Expected 384 features per article; got {embeddings.shape[1]}.")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise ValueError("Expected floating-point numeric embedding values.")
    # NaN and infinity are numeric values but cannot be used reliably as ML
    # features. isfinite checks every element, and all requires every check to pass.
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")
    print("Embedding array shape:", embeddings.shape)
    print("Embedding vector dimension:", embeddings.shape[1])
    print("Embedding data type:", embeddings.dtype)
    print("First embedding, first 8 values:", np.array2string(embeddings[0, :8], precision=4))
    print("Input/output counts match; every embedding value is finite.")


def main():
    """Run the four checks in order and retain helpful failure diagnostics."""
    stage = "environment verification"
    device = "not selected"
    try:
        device = verify_environment()
        stage = "AG News loading and inspection"
        texts, labels = inspect_dataset()
        stage = "MiniLM loading and embedding extraction"
        embeddings = extract_embeddings(texts, device)
        stage = "embedding validation"
        validate_embeddings(texts, labels, embeddings)
    except Exception as error:
        # Libraries raise several types of network/download/CUDA errors. Catch
        # them at this boundary to add context, then re-raise the original error
        # so Spyder shows its full traceback and a failed run cannot look passed.
        print(f"\nFEASIBILITY TEST FAILED during {stage}.")
        print(f"{type(error).__name__}: {error}")
        print("For download errors, check internet access, Hugging Face access, and proxy settings.")
        print("For file errors, check free disk space and cache directory write permissions.")
        if device.startswith("cuda"):
            print("For CUDA errors, close GPU-heavy applications and restart the Spyder kernel.")
            print("Check the interpreter and driver before considering package changes.")
        print("Do not reinstall PyTorch or change CUDA just to troubleshoot a download error.")
        raise

    print(f"\nFEASIBILITY TEST PASSED on {device}: five articles produced valid 384-D embeddings.")
    if device == "cpu":
        print("CPU feasibility passed; GPU feasibility was not established by this run.")
    print("No classifier training, fine-tuning, accuracy, or Macro-F1 evaluation was performed.")


# Importing this module for inspection does not download data or run the model.
# Running the file directly, including Spyder's Run File action, executes main().
if __name__ == "__main__":
    main()
