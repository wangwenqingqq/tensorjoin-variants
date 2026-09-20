"""Independent CPU evaluation of the frozen SM120 FP64 terminal operation order."""
import numpy as np

def terminal_order_distances(x):
    """Emulate the measured SM120 terminal's explicit FP64 PTX reduction graph.

    Two 256-element chunks, adjacent pairs per thread, XOR 16/8/4/2/1
    across 32 lanes, then XOR 2/1 across four warps, then add the chunks.
    Re-derive this graph if the compiled terminal, layout or toolkit changes.
    """
    delta=x.astype(np.float64)[:,None,:]-x.astype(np.float64)[None,:,:]
    sq=delta*delta
    local=sq.reshape(len(x),len(x),2,4,32,2)
    lanes=local[...,0]+local[...,1]
    for shift in (16,8,4,2,1):
        lanes=lanes+lanes[...,np.arange(32)^shift]
    warps=lanes[...,0]
    for shift in (2,1):
        warps=warps+warps[...,np.arange(4)^shift]
    chunks=warps[...,0]
    return (np.float64(0)+chunks[...,0])+chunks[...,1]


