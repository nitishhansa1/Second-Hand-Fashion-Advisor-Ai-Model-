import json

# Read notebook
with open('FashionAdvisor.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

num_modified = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'torch.save(model.state_dict(),' in line:
                # Add line to attach classes if relevant
                if "fashion_model.pth" in line:
                    cell['source'].insert(i, "model.classes = dataset.classes  # Attach classes to model object\n")
                    # After insertion, the index shifts by 1
                    cell['source'][i+1] = line.replace('torch.save(model.state_dict(),', 'torch.save(model,')
                    num_modified += 1
                elif "model.pth" in line:
                    # Just replace for the Google Drive export
                    cell['source'][i] = line.replace('torch.save(model.state_dict(),', 'torch.save(model,')
                    num_modified += 1

with open('FashionAdvisor.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Modified {num_modified} save statements to export entire model.')
