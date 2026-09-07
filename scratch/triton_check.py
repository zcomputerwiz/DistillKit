import torch, triton, triton.language as tl
print("triton", triton.__version__)

@triton.jit
def add_k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)

x = torch.randn(4096, device="cuda"); y = torch.randn(4096, device="cuda"); o = torch.empty_like(x)
add_k[(4,)](x, y, o, 4096, BLOCK=1024)
torch.cuda.synchronize()
print("triton kernel compiles and runs:", torch.allclose(o, x + y))
