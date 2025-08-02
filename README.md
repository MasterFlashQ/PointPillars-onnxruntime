# PointPillars-onnxruntime
![QQ202582-14374_1](https://github.com/user-attachments/assets/cb96779f-48ec-4cc4-9ac3-a6a15c4626c3)
The code for onnx file generation and point cloud preprocessing and postprocessing comes from [[PointPillars demployment](https://github.com/zhulf0804/PointPillars/tree/feature/deployment)].

## Dependecies:
- OpenCV 4.x
- CUDA 11.1
- onnxruntime 1.10.0
- GPU GTX 1660Ti

## Encountered problems:
1. The conversion of onnx files to rtr files failed：Unsolved
    ```
    [08/02/2025-14:54:36] [I] === Device Information ===
    [08/02/2025-14:54:36] [I] Selected Device: NVIDIA GeForce GTX 1660 Ti with Max-Q Design
    [08/02/2025-14:54:36] [I] Compute Capability: 7.5
    [08/02/2025-14:54:36] [I] SMs: 24
    [08/02/2025-14:54:36] [I] Compute Clock Rate: 1.335 GHz
    [08/02/2025-14:54:36] [I] Device Global Memory: 6143 MiB
    [08/02/2025-14:54:36] [I] Shared Memory per SM: 64 KiB
    [08/02/2025-14:54:36] [I] Memory Bus Width: 192 bits (ECC disabled)  
    [08/02/2025-14:54:36] [I] Memory Clock Rate: 6.001 GHz
    [08/02/2025-14:54:36] [I]
    [08/02/2025-14:54:37] [I] [TRT] [MemUsageChange] Init CUDA: CPU +480, GPU +0, now: CPU 14834, GPU 1153 (MiB)
    [08/02/2025-14:54:38] [I] [TRT] [MemUsageSnapshot] Begin constructing builder kernel library: CPU 14865 MiB, GPU 1153 MiB
    [08/02/2025-14:54:38] [I] [TRT] [MemUsageSnapshot] End constructing builder kernel library: CPU 15035 MiB, GPU 1185 MiB
    PS F:\Neusoft\TensorRT-7.2.3.4\bin>
    ```
2. The type of input node input_coors_batch is int64, but all int in C++ is int32. To avoid errors when copying memory, the type of coors_batch in the pytorch2onnx.py file was modified
   ```
   coors_batch = coors_batch.to(torch.int32)
   ```
   and it must be changed to int64 in the pointpillars.py later, otherwise an error will be reported
   ```
   for i in range(bs):
    cur_coors = coors_batch.to(torch.int64)
    cur_features = pooling_features
    canvas = torch.zeros((self.x_l * self.y_l, self.out_channel), dtype=torch.float32, device=device)
    cur_coors_flat = cur_coors[:, 2] * self.x_l + cur_coors[:, 1]  ## why select here.
    canvas[cur_coors_flat] = cur_features
    canvas = canvas.view(self.y_l, self.x_l, self.out_channel)
    canvas = canvas.permute(2, 0, 1).contiguous()
    batched_canvas.append(canvas)
   ```
