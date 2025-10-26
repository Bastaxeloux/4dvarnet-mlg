from pathlib import Path
from tqdm import tqdm

def rename_files(data_dir):
    """Rename all _13vars files to _x1"""
    data_dir = Path(data_dir)
    items = sorted(data_dir.glob('*_13vars.*'))
    print(f"Found {len(items)} items to rename")
    for item in tqdm(items, desc="Renaming", unit="file"):
        new_name = item.name.replace('_13vars', '_x1')
        new_path = item.parent / new_name
        item.rename(new_path)
    print(f"\nRenamed {len(items)} items")


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python rename_to_x1.py /path/to/data/dir")
        sys.exit(1)

    data_dir = sys.argv[1]
    rename_files(data_dir)
