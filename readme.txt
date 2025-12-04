README — How to run the Python program from the terminal

1) Open a terminal in the project directory
    cd path/adaptive-gaussian-filtering

2) Create a virtual environment (only once)
    python3 -m venv venv
    (If your system uses `python` instead of `python3`, use that.)

3) Activate the virtual environment
    - Linux / macOS:
      source venv/bin/activate
    - Windows (cmd):
      venv\Scripts\activate
    - Windows (PowerShell):
      venv\Scripts\Activate.ps1

    After activation your prompt typically shows (venv).

4) Install dependencies
    Ensure there's a requirements.txt in the project root. Then run:
      pip install --upgrade pip
      pip install -r requirements.txt

    To create requirements.txt from an existing environment:
      pip freeze > requirements.txt

5) Run the program
    Typical forms:
      python main.py
    or, if your entry point is a module:
      python -m package.module

    If the program expects arguments:
      python main.py --input input.png --output out.png

6) When finished, deactivate the venv
    deactivate

Notes:
 - Use a matching Python version for the project (check with `python --version`).
 - If you get permission errors, prefer using a virtual environment rather than sudo.
 - For reproducible builds, pin package versions in requirements.txt.