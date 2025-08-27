import torch

def poincare_distance(u, v, eps=1e-5):
    v = v.to(u.device).contiguous()
    u = u.contiguous()

    sq_u = (u ** 2).sum(dim=1, keepdim=True).clamp(max=1 - eps)
    sq_v = (v ** 2).sum(dim=0, keepdim=True).clamp(max=1 - eps)
    dot = u @ v

    num = (1 - 2 * dot + sq_v) * sq_u + (1 - sq_u) ** 2
    den = (1 - sq_u) * (1 - sq_v) + eps

    z = 1 + 2 * num / den
    z = torch.clamp(z, min=1 + eps)

    dist = torch.arccosh(z)
    return dist


def manhattan_distance(u, v):
    v = v.to(u.device).contiguous()
    u = u.contiguous()
    dist = torch.sum(torch.abs(u.unsqueeze(1) - v.T.unsqueeze(0)), dim=2)
    return dist


def euclidean_distance(u, v):
    v = v.to(u.device).contiguous()
    u = u.contiguous()
    sq_u = (u ** 2).sum(dim=1, keepdim=True)
    sq_v = (v ** 2).sum(dim=0, keepdim=True)
    dot = u @ v
    dist = sq_u - 2 * dot + sq_v
    dist = torch.sqrt(torch.clamp(dist, min=1e-5))
    return dist


def chebyshev_distance(u, v, chunk=50):
    """
    u: [B, D]
    v: [C, D] or [D, C]  ← 会自动判断
    chunk: 一次最多处理多少个类，防止显存炸
    """
    if v.device != u.device:
        v = v.to(u.device)
    if v.shape[0] == u.shape[1]:  # [D, C] → [C, D]
        v = v.T

    B, D = u.shape
    C = v.shape[0]

    results = []
    for i in range(0, C, chunk):
        v_chunk = v[i:i+chunk]  # [chunk, D]
        diff = u.unsqueeze(1) - v_chunk.unsqueeze(0)  # [B, chunk, D]
        dist = torch.max(torch.abs(diff), dim=2).values  # [B, chunk]
        results.append(dist)

    return torch.cat(results, dim=1)  # [B, C]



def distance(u, v, method="poincare"):
    if method is None or method == 'cosine':
        return u @ v
    elif method == "poincare":
        return poincare_distance(u, v)
    elif method == "euclidean":
        return euclidean_distance(u, v)
    elif method == "chebyshev":
        return chebyshev_distance(u, v)
    else:
        raise ValueError(f"Unsupported distance method: {method}")
