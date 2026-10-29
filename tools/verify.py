"""변환 결과 검증 스크립트

1) 원본 ONNX 모델을 numpy로 그대로 실행한 결과(ref.py)와
   변환된 단순 그래프를 numpy로 실행한 결과(simp.py)를 비교하고,
2) 브라우저용 엔진(engine.js)을 가짜 tf(mocktf.js)로 Node에서 돌린 결과와도 비교합니다.

사용법 (tools 폴더에서):
    python3 verify.py u2netp ../원본/u2netp.onnx
"""
import sys, subprocess, numpy as np
import ref, simp

name, onnx_path = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)

# 1) 320×320 입력으로 원본 vs 변환 그래프
x = rng.standard_normal((1, 3, 320, 320)).astype(np.float32)
r, _ = ref.run_onnx(onnx_path, x, '1959')
js, data = simp.load(name)
s = simp.run(js, data, x)
print(f'[원본 ONNX vs 변환 그래프] 최대 오차 {np.abs(r - s).max():.2e}')

# 2) 64×64 입력으로 변환 그래프(numpy) vs engine.js(Node)
x64 = rng.standard_normal((64, 64, 3)).astype(np.float32)   # NHWC
x64.tofile('x64.bin')
subprocess.run(['node', 'test.js', name], check=True)
y = np.fromfile(f'y64_{name}.bin', np.float32).reshape(64, 64)
s64 = simp.run(js, data, x64.transpose(2, 0, 1)[None])[0, 0]
print(f'[변환 그래프 vs engine.js] 최대 오차 {np.abs(s64 - y).max():.2e}')
