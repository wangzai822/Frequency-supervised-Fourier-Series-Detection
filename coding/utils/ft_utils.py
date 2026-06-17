import numpy as np
import torch
import torch.nn.functional as F
def scale_coords_ft(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, 0] -= pad[0]
    coords[:, 1] -= pad[1]
    coords /= gain
    return coords
def fft_1(coeffs,npts):
    n = (len(coeffs) - 2) // 4
    t = np.linspace(0, 1, npts, endpoint=False)
    x = np.ones_like(t) * coeffs[0]
    y = np.ones_like(t) * coeffs[1]
    offset = 2
    for i in range(n):
        assert offset + 4 <= len(coeffs)
        a, b, c, d = coeffs[offset : offset + 4]
        x += a * np.cos(2*np.pi*(i+1)*t) + b * np.sin(2*np.pi*(i+1)*t)
        y += c * np.cos(2*np.pi*(i+1)*t) + d * np.sin(2*np.pi*(i+1)*t)
        offset+=4
    return x,y
def fft_pts(coeffs, npts):
    n = (len(coeffs) - 2) // 4
    t = np.linspace(0, 1, npts, endpoint=False)
    x = np.ones_like(t) * coeffs[0]
    y = np.ones_like(t) * coeffs[1]
    offset = 2
    for i in range(n):
        assert offset + 4 <= len(coeffs)
        a, b, c, d = coeffs[offset : offset + 4]
        x += a * np.cos(2*np.pi*(i+1)*t) + b * np.sin(2*np.pi*(i+1)*t)
        y += c * np.cos(2*np.pi*(i+1)*t) + d * np.sin(2*np.pi*(i+1)*t)
        offset += 4
    xy = np.column_stack((x, y))
    return xy
def ft2dir(coefs,cen=0):
    abcd = coefs[:, 2:6]
    a1, b1, c1, d1 = torch.split(abcd, 1, dim=-1)
    if cen==0:
        a1,b1,c1,d1 = a1.float(),b1.float(),c1.float(),d1.float()
        cos_sin2t = torch.stack([a1**2 + c1**2 - b1**2 - d1**2, 2*(a1*b1 + c1*d1)],dim=1).squeeze(2)
        cos_sin2t = F.normalize(cos_sin2t,p=2,dim=-1)
        cos_t = torch.sqrt((1+cos_sin2t[:,0])/2)
        sin_t = torch.sqrt((1-cos_sin2t[:,0])/2)
        sin_t[cos_sin2t[:,1]<0] *= -1
        cos_sin = torch.stack([cos_t,sin_t],dim=1)
    elif cen==1:
        tan2t = (2*(a1*b1 + c1*d1))/(a1**2 + c1**2 - b1**2 - d1**2)
        cos2t = torch.sqrt(1 / (1 + tan2t**2)).view(-1,1)
        cos_sin = cos2t @ torch.tensor([[0.5, -0.5]], dtype=abcd.dtype, device=coefs.device) + 0.5
        cos_sin = torch.sqrt(cos_sin)
        index = torch.where(tan2t < 0)[0]
        cos_sin[index, 1] *= -1
    else:
        x = a1**2 + c1**2 - b1**2 - d1**2
        y = 2*(a1*b1 + c1*d1)
        phi2 = torch.atan2(y + 0.0, x + 0.0)
        phi  = 0.5 * phi2
        cos_t = torch.cos(phi)
        sin_t = torch.sin(phi)
        cos_sin = torch.cat([cos_t,sin_t],dim=1)
    return cos_sin
def ft2pts(coefs,cen=0):
    cos_sin = ft2dir(coefs,cen)
    abcd = coefs[:, 2:6]
    xc, yc = torch.split(coefs[:, :2].clone(), 1, dim=-1)
    an1, bn1, cn1, dn1 = torch.split(abcd, 1, dim=-1)
    m_sin = cos_sin[:, 1:2]
    m_cos = cos_sin[:, 0:1]
    scale=1.0
    a1 = scale*(an1 * m_cos + bn1 * m_sin)
    b1 = scale*(-an1 * m_sin + bn1 * m_cos)
    c1 = scale*(cn1 * m_cos + dn1 * m_sin)
    d1 = scale*(-cn1 * m_sin + dn1 * m_cos)
    P0=torch.cat([xc-a1-b1,yc-c1-d1], dim=-1).view(-1, 2)
    P1=torch.cat([xc-a1+b1,yc-c1+d1], dim=-1).view(-1, 2)
    P2=torch.cat([xc+a1+b1,yc+c1+d1], dim=-1).view(-1, 2)
    P3=torch.cat([xc+a1-b1,yc+c1-d1], dim=-1).view(-1, 2)
    points = torch.cat([P0, P1, P2, P3], dim=1)
    return points
def ft2xy(an,bn,cn,dn,theta_fine,term):
    term = an.shape[0] if term<=0 else term
    x_approx = sum([an[i]*np.cos(i*theta_fine) + bn[i]*np.sin(i*theta_fine) for i in range(term)])
    y_approx = sum([cn[i]*np.cos(i*theta_fine) + dn[i]*np.sin(i*theta_fine) for i in range(term)])
    return x_approx,y_approx
def fft_area(ft_label):
    an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)
    an, bn, cn, dn = an.squeeze(-1), bn.squeeze(-1), cn.squeeze(-1), dn.squeeze(-1)
    assert an.shape[0]==len(an)==bn.shape[0]==cn.shape[0]==dn.shape[0]
    return np.pi * ((an*dn-bn*cn)*np.arange(1,1+an.shape[0])).sum()
def fft_areas(ft_labels):
    n_obj, dim = ft_labels.shape
    n = (dim - 2) // 4
    coeffs = ft_labels[:, 2:].reshape(n_obj, n, 4)
    an, bn, cn, dn = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2], coeffs[..., 3]
    k = np.arange(1, n+1, dtype=ft_labels.dtype)
    areas = np.pi * ((an * dn - bn * cn) * k).sum(axis=1)
    return areas
def reverse_ffts(ft_label):
    ft_label[:,3::2] *= -1