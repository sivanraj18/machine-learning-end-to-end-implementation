import os
dirs = [
    'src','data//raw','data//processed','notebooks','saved_models','data_given'
]

for dirs_ in dirs:
    os.makedirs(dirs_,exist_ok=True)
    with open(os.path.join(dirs_,'.gitkeep'),'w') as f:
        pass


files = ['dvc.yaml','params.yaml','.gitignore',os.path.join('src','__init__.py')]

for files_ in files:
    with open(files_,'w') as f:
        pass
