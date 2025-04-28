import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
def construct_affine(params):
    device = params.device
    n = params.shape[0]
    sx, sy, angle = params[:, 0], params[:, 1], params[:, 2]
    m, tx, ty = params[:, 3], params[:, 4], params[:, 5]
    zeros = torch.zeros(n, 2, 2, device=device)
    rot = torch.stack(
        (
            torch.stack((angle.cos(), -angle.sin()), dim=1),
            torch.stack((angle.sin(), angle.cos()), dim=1),
        ),
        dim=1,
    )
    shear = zeros.clone()
    shear[:, 0, 0] = 1
    shear[:, 1, 1] = 1
    shear[:, 0, 1] = m
    scale = zeros.clone()
    scale[:, 0, 0] = sx
    scale[:, 1, 1] = sy
    A = torch.matmul(torch.matmul(rot, shear), scale)
    b = torch.stack((tx, ty), dim=1)
    theta = torch.cat((A, b[:, :, None]), dim=2)
    return theta.reshape(n, 6)


def expm(theta):
    n_theta = theta.shape[0]
    zero_row = torch.zeros(n_theta, 1, 3, dtype=theta.dtype, device=theta.device)
    theta = torch.cat([theta, zero_row], dim=1)
    theta = torch_expm(theta)
    theta = theta[:, :2, :]
    return theta


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ST_Affine(nn.Module):
    def __init__(self, input_shape):
        super(ST_Affine, self).__init__()
        self.input_shape = input_shape

    def forward(self, x, theta, inverse=False):
        if inverse:
            A = theta[:, :4]
            b = theta[:, 4:]
            A = torch.inverse(A.view(-1, 2, 2)).reshape(-1, 4)
            b = -b
            theta = torch.cat((A, b), dim=1)

        theta = theta.view(-1, 2, 3)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size, align_corners=True)  # type: ignore
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def trans_theta(self, theta):
        return theta

    def dim(self):
        return 6


class ST_AffineDecomp(nn.Module):
    def __init__(self, input_shape):
        super(ST_AffineDecomp, self).__init__()
        self.input_shape = input_shape

    def forward(self, x, theta, inverse=False):
        # theta = [sx, sy, angle, shear, tx, ty]
        if inverse:
            theta[:, :2] = 1 / theta[:, :2]
            theta[:, 2:] = -theta[:, 2:]

        theta = construct_affine(theta)
        theta = theta.view(-1, 2, 3)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size, align_corners=True)  # type: ignore
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def trans_theta(self, theta):
        return theta

    def dim(self):
        return 6


class ST_AffineDiff(nn.Module):
    def __init__(self, input_shape):
        super(ST_AffineDiff, self).__init__()
        self.input_shape = input_shape

    def forward(self, x, theta, inverse=False):
        if inverse:
            theta = -theta
        theta = theta.view(-1, 2, 3)
        theta = expm(theta)
        output_size = torch.Size([x.shape[0], *self.input_shape])
        grid = F.affine_grid(theta, output_size, align_corners=True)  # type: ignore
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def trans_theta(self, theta):
        return expm(theta)

    def dim(self):
        return 6


def get_transformer(name):
    transformers = {
        "affine": ST_Affine,
        "affinediff": ST_AffineDiff,
        "affinedecomp": ST_AffineDecomp,
    }
    assert name in transformers, "Transformer not found, choose between: " + ", ".join(
        [k for k in transformers.keys()]
    )
    return transformers[name]


# %%
def _complex_case3x3(a, b, c, d, e, f, x, y):
    nom = 1 / ((-(a**2) - 2 * a * e - e**2 - x**2) * x)
    sinx = (0.5 * x).sin()
    cosx = (0.5 * x).cos()
    expea = (0.5 * (a + e)).exp()

    Ea = -(4 * ((a - e) * sinx + cosx * x)) * (a * e - b * d) * expea * nom
    Eb = -8 * b * (a * e - b * d) * sinx * expea * nom
    Ec = (
        -4
        * (
            (
                (-c * e**2 + (a * c + b * f) * e + b * (a * f - 2 * c * d)) * sinx
                - cosx * x * (b * f - c * e)
            )
            * expea
            + (b * f - c * e) * x
        )
        * nom
    )
    Ed = -8 * d * (a * e - b * d) * sinx * expea * nom
    Ee = 4 * ((a - e) * sinx - cosx * x) * (a * e - b * d) * expea * nom
    Ef = (
        4
        * (
            (
                (a**2 * f + (-c * d - e * f) * a + d * (2 * b * f - c * e)) * sinx
                - x * cosx * (a * f - c * d)
            )
            * expea
            + x * (a * f - c * d)
        )
        * nom
    )

    E = torch.stack(
        [torch.stack([Ea, Eb, Ec], dim=1), torch.stack([Ed, Ee, Ef], dim=1)], dim=1
    )
    return E


def _real_case3x3(a, b, c, d, e, f, x, y):
    eap = a + e + x
    eam = a + e - x
    nom = 1 / (x * eam * eap)
    expeap = (0.5 * eap).exp()
    expeam = (0.5 * eam).exp()

    Ea = -2 * (a * e - b * d) * ((a - e - x) * expeam - (a - e + x) * expeap) * nom
    Eb = -4 * b * (expeam - expeap) * (a * e - b * d) * nom
    Ec = (
        ((4 * c * d - 2 * f * eap) * b - 2 * c * e * (a - e - x)) * expeam
        + ((-4 * c * d + 2 * f * eam) * b + 2 * c * e * (a - e + x)) * expeap
        + 4 * (b * f - c * e) * x
    ) * nom
    Ed = -4 * d * (expeam - expeap) * (a * e - b * d) * nom
    Ee = 2 * (a * e - b * d) * ((a - e + x) * expeam - (a - e - x) * expeap) * nom
    Ef = (
        (
            2 * a**2 * f
            + (-2 * c * d - 2 * f * (e - x)) * a
            + 4 * d * (b * f - (1 / 2) * c * (e + x))
        )
        * expeam
        + (
            -2 * a**2 * f
            + (2 * c * d + 2 * f * (e + x)) * a
            - 4 * (b * f - (1 / 2) * c * (e - x)) * d
        )
        * expeap
        - (4 * (a * f - c * d)) * x
    ) * nom

    E = torch.stack(
        [torch.stack([Ea, Eb, Ec], dim=1), torch.stack([Ed, Ee, Ef], dim=1)], dim=1
    )
    return E


def _limit_case3x3(a, b, c, d, e, f, x, y):
    ea2 = (a + e) ** 2
    expea = (0.5 * (a + e)).exp()
    Ea = 2 * (a - e + 2) * (a * e - b * d) * expea / ea2
    Eb = 4 * b * (a * e - b * d) * expea / ea2
    Ec = (
        (
            -2 * c * e**2
            + (2 * b * f + 2 * c * (a + 2)) * e
            + 2 * b * (-2 * c * d + f * (a - 2))
        )
        * expea
        + 4 * b * f
        - 4 * c * e
    ) / ea2
    Ed = 4 * d * (a * e - b * d) * expea / ea2
    Ee = -(2 * (a - e - 2)) * (a * e - b * d) * expea / ea2
    Ef = (
        (
            -2 * a**2 * f
            + (2 * c * d + 2 * f * (e + 2)) * a
            - 4 * d * (b * f - 0.5 * c * (e - 2))
        )
        * expea
        - 4 * a * f
        + 4 * c * d
    ) / ea2

    E = torch.stack(
        [torch.stack([Ea, Eb, Ec], dim=1), torch.stack([Ed, Ee, Ef], dim=1)], dim=1
    )
    return E


def torch_expm3x3(A):
    # Initilial computations
    a, b, c = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    d, e, f = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
    y = a**2 - 2 * a * e + 4 * b * d + e**2
    x = y.abs().sqrt()

    # Calculate all cases and then choose according to the input
    real_res = _real_case3x3(a, b, c, d, e, f, x, y)
    complex_res = _complex_case3x3(a, b, c, d, e, f, x, y)

    expmA = torch.where(y[:, None, None] > 0, real_res, complex_res)
    return expmA


def torch_expm(A):
    """ """
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1, 2), keepdim=True))

    # Scaling step
    maxnorm = torch.Tensor([5.371920351148152]).type(A.dtype).to(A.device)
    zero = torch.Tensor([0.0]).type(A.dtype).to(A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    Ascaled = A / 2.0**n_squarings
    n_squarings = n_squarings.flatten().type(torch.int64)

    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V
    R, _ = torch.linalg.solve(P, Q)  # solve P = Q*R

    # Unsquaring step
    n = n_squarings.max()
    res = [R]
    for i in range(n):
        res.append(res[-1].matmul(res[-1]))
    R = torch.stack(res)
    expmA = R[n_squarings, torch.arange(n_A)]
    return expmA


def torch_log2(x):
    return torch.log(x) / torch.log(torch.Tensor([2.0])).type(x.dtype).to(x.device)


def torch_pade13(A):
    b = (
        torch.Tensor(
            [
                64764752532480000.0,
                32382376266240000.0,
                7771770303897600.0,
                1187353796428800.0,
                129060195264000.0,
                10559470521600.0,
                670442572800.0,
                33522128640.0,
                1323241920.0,
                40840800.0,
                960960.0,
                16380.0,
                182.0,
                1.0,
            ]
        )
        .type(A.dtype)
        .to(A.device)
    )

    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(
        A,
        torch.matmul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2)
        + b[7] * A6
        + b[5] * A4
        + b[3] * A2
        + b[1] * ident,
    )
    V = (
        torch.matmul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2)
        + b[6] * A6
        + b[4] * A4
        + b[2] * A2
        + b[0] * ident
    )
    return U, V
