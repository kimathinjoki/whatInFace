# Ollama Model Benchmarking Tool

This tool allows you to benchmark different Ollama language models for performance testing. It's designed to help you determine which models run most efficiently on your hardware for specific tasks, such as loan approval decision-making.

## Prerequisites

Before using this tool, you need to install:

1. **Python** (3.8 or higher)
2. **Ollama** for running language models locally
3. Required Python packages

## Installation Steps

### 1. Install Python

If you don't have Python installed:

- **Windows**: 
  - Download the installer from [python.org](https://www.python.org/downloads/)
  - During installation, check "Add Python to PATH"
  - Verify installation by opening Command Prompt or PowerShell and typing:
    ```
    python --version
    ```

- **macOS**:
  - Install using Homebrew: `brew install python`
  - Or download from [python.org](https://www.python.org/downloads/)

- **Linux**:
  - Most distributions come with Python pre-installed
  - If not, use your package manager:
    ```
    # Ubuntu/Debian
    sudo apt update
    sudo apt install python3 python3-pip
    ```

### 2. Install Ollama

Ollama allows you to run large language models locally on your machine.

- **Windows**:
  - Download the installer from [ollama.com](https://ollama.com)
  - Run the installer and follow the prompts
  - After installation, Ollama will appear in your system tray

- **macOS**:
  - Download from [ollama.com](https://ollama.com)
  - Open the downloaded file and follow installation instructions

- **Linux**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

### 3. Install Required Python Packages

```bash
pip install pandas requests tqdm
```

## Setting Up Models

Before running the benchmark, you need to download the language models you want to test:

```bash
# Pull the models (this downloads them to your computer)
ollama pull llama3:8b
ollama pull codellama:7b
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull phi3:mini
```

## Using the Benchmark Tool

### 1. Prepare Your Data

The tool expects a CSV file with loan application data. Ensure your CSV file contains the required columns as shown in the sample data schema.

### 2. Configure the Script

Open `ollamaModelTest.py` and update the input file path to point to your CSV file:

```python
# Path to the input file
input_file_path = './your_data_directory/your_file.csv'
```

### 3. Run the Benchmark

```bash
python ollamaModelTest.py
```

When you run the script, you'll be guided through an interactive menu:

1. **Check Model Availability**: The script will first check if the required models are installed
2. **Choose Test Type**:
   - Quick benchmark of all models (faster, fewer samples)
   - Full test of a specific model (more comprehensive, more samples)
   - Run both tests
3. **Customize Sample Size**: Specify how many samples to use for testing

### 4. Analyze Results

The script generates several output files:

- **Benchmark Summary**: A JSON file with performance stats for all tested models
- **Model-specific CSVs**: Detailed results for each model with processing times and decisions
- **Full Test Results**: Complete dataset with model decisions and performance metrics

Look for these files in:
- `./ollama_benchmark_results/` (for benchmark tests)
- `./ollama_full_test_results/` (for full tests)

## Performance Optimization Tips

To get the best performance from Ollama:

1. **Environment Variables**: Set optimization variables before running Ollama:
   ```bash
   # Windows PowerShell
   $env:OLLAMA_FLASH_ATTENTION=1
   $env:OLLAMA_KV_CACHE_TYPE=q8_0
   
   # Linux/macOS
   export OLLAMA_FLASH_ATTENTION=1
   export OLLAMA_KV_CACHE_TYPE=q8_0
   ```

2. **Hardware Considerations**:
   - GPUs significantly outperform CPUs for these tasks
   - VRAM is often the limiting factor for larger models
   - For NVIDIA GPUs, ensure you have the latest CUDA drivers

## Troubleshooting

- **"Python not found"**: Ensure Python is added to your PATH
- **Ollama connection errors**: Verify Ollama is running (check system tray or run `ollama list`)
- **Out of memory errors**: Try smaller models or reduce context length
- **Slow performance**: Run on GPU if available, or try more quantized models (e.g., models with Q4_K_M in their name)

## Model Selection Guidelines

- **8-7B parameter models** (like Llama3:8b) work well on consumer GPUs with 8GB+ VRAM
- **Smaller models** (like Gemma3:1b, Phi3:mini) work on less powerful hardware
- **CPU-only setups** should use the smallest models for reasonable performance

## License

This tool is provided under the MIT License.