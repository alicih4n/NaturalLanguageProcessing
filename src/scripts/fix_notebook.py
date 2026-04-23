import json

def fix_notebook():
    file_path = 'notebooks/PathoIntern_Intent_Engine.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                new_source = []
                for line in source:
                    if "stop_words='english'" in line:
                        line = line.replace("stop_words='english'", "stop_words=None")
                    new_source.append(line)
                cell['source'] = new_source
            else:
                if "stop_words='english'" in source:
                    cell['source'] = source.replace("stop_words='english'", "stop_words=None")
                    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    fix_notebook()
