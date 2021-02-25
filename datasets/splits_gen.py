from pathlib import Path

def gen_ADE20K():
    '''
    /workspace/stereo-from-mono/workdata/ADE20K$ tree -L 1
    .
    ├── testing
    ├── training
    └── validation
    '''

    base_path = '/workspace/stereo-from-mono/workdata/ADE20K'
    base_path = Path(base_path)
    train_path = base_path / 'training'
    val_path = base_path / 'validation'

    with open('train_files_all.txt', 'w') as file:
        file.writelines(sorted([str(f.relative_to(base_path)) + '\n' 
            for f in train_path.glob('*') if f.suffix in ['.png', '.jpg']]))
    with open('val_files_all.txt', 'w') as file:
        file.writelines(sorted([str(f.relative_to(base_path)) + '\n' 
            for f in val_path.glob('*') if f.suffix in ['.png', '.jpg']]))
