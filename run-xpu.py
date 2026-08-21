import sys
import torch

# Map CUDA calls to Intel Arc XPU
if hasattr(torch, "xpu") and torch.xpu.is_available():
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_name = lambda dev=None: torch.xpu.get_device_name(0)
    
    # Override device mapping
    _orig_tensor_to = torch.Tensor.to
    def _to(self, *args, **kwargs):
        args = tuple('xpu' if a == 'cuda' or a == 'cuda:0' else a for a in args)
        if 'device' in kwargs and kwargs['device'] in ['cuda', 'cuda:0']:
            kwargs['device'] = 'xpu'
        return _orig_tensor_to(self, *args, **kwargs)
    torch.Tensor.to = _to

# Execute target script
if __name__ == "__main__":
    import runpy
    sys.argv.pop(0) # Remove run_xpu.py from arguments
    runpy.run_path(sys.argv[0], run_name="__main__")