# Python

## Can do
Run any Python code. Batch processing, custom logic, file manipulation, image processing, data transforms.

## Cannot do
Cannot call Pinpoint API tools directly. Cannot access GPU models (InsightFace). For those use the dedicated tools.

## Tools
- **run_python(code, timeout?)** → Execute Python code. Returns stdout + list of created files. Timeout default 30s, max 120s. Working dir: /tmp/pinpoint_python/.

## Pre-loaded Libraries
PIL (Image, ImageDraw), pandas, numpy, matplotlib, os, json, pathlib, shutil, csv, math, re, hashlib

## Notes
- Use print() to return results
- Save output files to WORK_DIR variable
- matplotlib: use plt.savefig(), not plt.show()
- For images: Image.open(path), process, .save(output_path)
