"""Validate the shipped .pt against ONNX. No network calls."""
from pathlib import Path
import json, time, socket
import numpy as np
import torch
import onnx
import onnxruntime as ort
from train_export import ShapeNet
# Fail if any Python code attempts an outgoing connection during validation.
socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled during validation'))
root=Path(__file__).resolve().parents[1]
model=ShapeNet();model.load_state_dict(torch.load(root/'pc/shapes.pt',weights_only=True,map_location='cpu'));model.eval()
path=root/'android/app/src/main/assets/shapes.onnx';onnx.checker.check_model(str(path))
options=ort.SessionOptions();options.intra_op_num_threads=2
s=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
data=np.load(root/'pc/validation.npz');x,y=data['x'],data['y']
with torch.no_grad(): expected=model(torch.from_numpy(x)).numpy()
actual=np.concatenate([s.run(['logits'],{'pixels':row.reshape(1,256)})[0] for row in x])
error=float(np.max(np.abs(expected-actual)))
np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
assert (actual.argmax(1)==y).mean()>=.98
fixtures=json.loads((root/'android/app/src/main/assets/fixtures.json').read_text())
for f in fixtures:
    got=s.run(None,{'pixels':np.array([f['pixels']],np.float32)})[0][0]
    assert int(got.argmax())==f['label'];np.testing.assert_allclose(got,f['logits'],atol=.001)
for _ in range(20):s.run(None,{'pixels':x[:1]})
times=[]
for _ in range(100):
    start=time.perf_counter();s.run(None,{'pixels':x[:1]});times.append((time.perf_counter()-start)*1000)
result={'status':'PASS','onnx_checker':'PASS','holdout_samples':len(y),'synthetic_holdout_accuracy':float((actual.argmax(1)==y).mean()),'max_logit_error':error,'fixtures_passed':len(fixtures),'desktop_cpu_median_inference_ms':float(np.median(times)),'desktop_cpu_p95_inference_ms':float(np.percentile(times,95)),'versions':{'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__},'limitations':['Synthetic data only; no real handwriting evaluation.','Desktop timings are not Android performance measurements.','Python outgoing connection helper blocked; no model downloads.']}
(root/'TEST_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
