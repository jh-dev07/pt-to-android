"""Train synthetic shape classifier and export .pt -> ONNX. No downloads or datasets."""
from pathlib import Path
import json, random, math
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
class ShapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(256,48),nn.ReLU(),nn.Linear(48,3))
    def forward(self,x): return self.net(x)
def sample(label,rng):
    a=np.zeros((16,16),np.float32)
    cx=rng.uniform(7,9); cy=rng.uniform(7,9); r=rng.uniform(4,6)
    for y in range(16):
        for x in range(16):
            dx=x+.5-cx; dy=y+.5-cy
            if label==0: inside=dx*dx+dy*dy<=r*r
            elif label==1: inside=abs(dx)<=r and abs(dy)<=r
            else: inside=-r<=dy<=r and abs(dx)<=(dy+r)/2
            a[y,x]=inside
    return a.flatten()
def dataset(n,seed):
    rng=random.Random(seed); y=np.arange(n)%3
    return np.stack([sample(int(t),rng) for t in y]),y
if __name__=='__main__':
    torch.manual_seed(42)
    x,y=dataset(6000,42); tx,ty=dataset(900,99)
    model=ShapeNet(); opt=torch.optim.Adam(model.parameters(),lr=.003)
    for epoch in range(90):
        order=torch.randperm(len(y))
        for idx in order.split(256):
            out=model(torch.from_numpy(x)[idx]); loss=nn.functional.cross_entropy(out,torch.from_numpy(y)[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    assets=ROOT/'android/app/src/main/assets'; assets.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),ROOT/'pc/shapes.pt')
    # Demonstrate restoration from the exact artifact being shipped.
    restored=ShapeNet(); restored.load_state_dict(torch.load(ROOT/'pc/shapes.pt',map_location='cpu',weights_only=True)); restored.eval()
    torch.onnx.export(restored,torch.zeros(1,256),str(assets/'shapes.onnx'),input_names=['pixels'],output_names=['logits'],opset_version=17,dynamo=False)
    fixtures=[]
    for label in range(3):
        row=sample(label,random.Random(123))
        logits=restored(torch.from_numpy(row).reshape(1,256)).detach().numpy()[0]
        fixtures.append({'label':label,'pixels':row.tolist(),'logits':logits.tolist()})
    (assets/'fixtures.json').write_text(json.dumps(fixtures))
    sd=restored.state_dict()
    weights={k:v.tolist() for k,v in sd.items()}
    (ROOT/'pc/weights.json').write_text(json.dumps(weights))
    np.savez(ROOT/'pc/validation.npz',x=tx,y=ty)
    accuracy=float((restored(torch.from_numpy(tx)).argmax(1).numpy()==ty).mean())
    (ROOT/'pc/training_metrics.json').write_text(json.dumps({'synthetic_holdout_accuracy':accuracy,'samples':900,'seed':42,'epochs':90},indent=2))
    print('Synthetic holdout accuracy:',accuracy)
