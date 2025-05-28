# Code Optimization Tool for C and Python

A web-based code optimization tool that analyzes and optimizes C and Python code using advanced static analysis and LLVM-based transformations. The tool provides detailed feedback on optimizations applied and code metrics, supporting both Python and C languages.

---

## Features

- **Python Code Optimization**
  - Dead code removal (unreachable code, unused variables)
  - Constant folding (evaluates constant expressions at compile time)
  - Strength reduction (replaces expensive operations with cheaper ones)
  - Loop optimizations (such as loop unrolling for small constant ranges)
  - Function inlining (inlines simple functions for performance)
  - Detailed metrics: line reduction, applied optimizations

- **C Code Optimization**
  - Uses LLVM (`clang` and `opt`) for standard O2 optimizations
  - Constant folding and strength reduction using pattern matching
  - Dead code removal (detects and removes obvious dead code, e.g., `if(0)`)
  - Reports on LLVM IR-level optimizations (branch, load/store, function inlining)
  - Provides both original and optimized LLVM IR for analysis

- **Web Interface**
  - Simple web UI for submitting code and viewing optimizations
  - REST API endpoints for integration

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Flask** (for the web server)
- **LLVM tools** (`clang`, `opt`) for C code optimization

### Install dependencies:

#### pip install flask

Install LLVM tools:

- **Ubuntu:**  
  `sudo apt-get install llvm clang`
- **macOS (Homebrew):**  
  `brew install llvm`

### Running the Application

``` python app.py ```


The server will start at `http://localhost:5000`.

---

## Usage

### Web Interface

- Open `http://localhost:5000` in your browser.
- Select language (C or Python), paste your code, and click "Optimize".
- View the optimized code, metrics, and a list of optimizations applied.

### API

- **POST** `/optimize`
  - **Request JSON:**
    ```
    {
      "code": "<your code here>",
      "language": "python" // or "c"
    }
    ```
  - **Response JSON:**
    - `original_code`: Original input code
    - `optimized_code`: Optimized version
    - `optimizations_applied`: List of optimizations performed
    - `metrics`: Lines before/after, reduction percentage
    - Additional fields for C: `original_ir`, `optimized_ir`

- **GET** `/health`
  - Returns server health status.

---

## Example Optimizations

- **Python:**
  - Removes dead branches:  
    ```
    if False:
        print("Never runs")
    ```
    → Removed

  - Constant folding:  
    ```
    x = 2 + 3
    ```
    → `x = 5`

  - Loop unrolling for small ranges

- **C:**
  - Strength reduction:  
    ```
    int y = x * 2;
    ```
    → `int y = x << 1;`

  - Dead code removal:  
    ```
    if (0) { ... }
    ```
    → Removed

---

## Project Structure

app.py # Main Flask application and optimization logic
templates/index.html # Web UI (not shown)
static/ # CSS/JS assets (if any)


---

## Limitations

- C code optimization requires LLVM tools (`clang`, `opt`) to be installed and available in PATH.
- The C optimizer uses a simplified IR-to-C process; some advanced optimizations may not be fully reflected in the C output.
- Python optimizations focus on static AST transformations and may not catch all dynamic or runtime behaviors.

---

## Acknowledgments

- Built using Python's `ast` module for static analysis.
- C optimization powered by LLVM.

---

## Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss changes or enhancements.

---

## Contact

For questions or support, open an issue on the repository.

---

**Happy optimizing!**

