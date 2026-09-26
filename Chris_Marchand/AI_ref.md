### Google AI Overview (Sep 26 9:56pm)

Prompt: "in python how to recruively get paths of all sub dir of a path"

```
Method 1: Using pathlib (Recommended for Python 3.4+)The pathlib module provides an object-oriented approach. You can use .rglob("*") to search recursively and filter for directories using .is_dir().pythonfrom pathlib import Path

# Define your starting directory
base_path = Path("/path/to/your/directory")

# Get paths of all subdirectories recursively
subdirectories = [str(p) for p in base_path.rglob("*") if p.is_dir()]

print(subdirectories)
```