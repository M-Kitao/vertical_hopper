import zipfile, io, torch

MODEL_PATH = r"C:\Users\masan\Documents\vertical_hopper_2\RL\results\TH_full_4\models\default\seed_0\final_model.zip"

with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
    print("=== zip内ファイル ===")
    print(zf.namelist())
    
    print("\n=== policy.pth のキー ===")
    with zf.open('policy.pth') as f:
        params = torch.load(io.BytesIO(f.read()), map_location='cpu')
    for k in params.keys():
        print(k)
    
    print("\n=== policy.optimizer.pth の内容 ===")
    with zf.open('policy.optimizer.pth') as f:
        opt = torch.load(io.BytesIO(f.read()), map_location='cpu')
    print(type(opt))
    if isinstance(opt, dict):
        print("キー:", list(opt.keys()))