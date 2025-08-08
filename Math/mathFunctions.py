import numpy as np
def log_base(value,base):
    return np.log(value) / np.log(base)

import re

def sanitize_user_input(raw_func: str) -> str:
    # 1. Lowercase everything for consistency
    func = raw_func.lower()

    # 2. Remove "y =" or "f(x) =" if user includes it
    func = re.sub(r"^\s*(y\s*=|f\s*\(\s*x\s*\)\s*=)\s*", "", func)

    # 3. Replace common math notations:
    func = func.replace("^", "**")           # x^2 → x**2
    func = func.replace("ln", "log")  # ln(x) → log(x)
    func = func.replace("π","pi")
    func = re.sub(r"√\s*\(\s*([^)]+?)\s*\)", r"sqrt(\1)", func)
    func = re.sub(r"√\s*([a-zA-Z0-9_]+)", r"sqrt(\1)", func)
    
    typo_map = {
        "sine": "sin",
        "cosine": "cos",
        "tangent": "tan",
        "cotangent": "cot",  # Not supported yet but optional
        "secant": "sec",
        "cosecant": "csc",
        "sinn": "sin",   # Just in case
        "cosn": "cos",
    }
    for wrong, right in typo_map.items():
        func = func.replace(wrong, right)

    # 4. Insert * between number/variable and variable (e.g., 2x → 2*x, xsin(x) → x*sin(x))
    func = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", func)       # 2x → 2*x
    func = re.sub(r"([a-zA-Z])(\()", r"\1*\2", func)       # xsin(x) → x*sin(x)
    func = re.sub(r"(\))([a-zA-Z])", r"\1*\2", func)       # )x → )*x
    func = re.sub(r"(\d)(\()", r"\1*\2", func)             # 2(x) → 2*(x)

    # 5. Clean up any duplicate operators like ***, or */ or spaces
    func = re.sub(r"\*{3,}", "**", func)
    func = re.sub(r"\s+", "", func)

    return func