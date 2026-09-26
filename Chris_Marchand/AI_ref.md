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


### ChatGPT 5.6 (Sep 26 10:51pm)

Prompt: "Give python regex for matching the patient id from this string.

"ADNI_T1_3T/ADNI_016_S_6381_MR_Accelerated_Sagittal_MPRAGE__br_raw_20180522132015417_85_S688087_I1000691.nii"

The patient id is 016_S_6381"

```
You can use this regex:

r"ADNI_(\d{3}_S_\d{4})"
```