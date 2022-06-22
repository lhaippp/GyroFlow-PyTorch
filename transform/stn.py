import torch
import numpy as np


def construct_point_pair_v2(S_1p, T_1p):
    """
    construct the matrix 2x8,2x1
    :param S_1p: one point, shape=[bs,2]
    :param T_1p: one point, shape=[bs,2]

    m_2x6 = [[0,-x],
             [x,0]]

    M1 = [[v'],
          [-u']]

    m_2x3 = M1*x

    x=[u,v,1]
    [m_2x6, m_2x3] = m_2x9---> m_2x8
    :return:
    """
    bs, num2 = S_1p.shape
    data_type = S_1p.dtype

    # u,v,1
    ones_1x1 = torch.ones([bs, 1, 1], dtype=data_type)
    if torch.cuda.is_available():
        ones_1x1 = ones_1x1.cuda()
    xyz = torch.cat((S_1p.reshape([bs, 1, 2]), ones_1x1), dim=2)

    # construct the m_2x6
    zeors_1x3 = torch.zeros([bs, 1, 3], dtype=data_type)
    if torch.cuda.is_available():
        zeors_1x3 = zeors_1x3.cuda()
    m1_1x6 = torch.cat((zeors_1x3, -xyz), dim=2)
    m2_1x6 = torch.cat((xyz, zeors_1x3), dim=2)
    m_2x6 = torch.cat((m1_1x6, m2_1x6), dim=1)

    # construct the m_2x2
    temp_m = torch.tensor([[0, -1], [1, 0]], dtype=data_type)
    if torch.cuda.is_available():
        temp_m = temp_m.cuda()
    M1 = torch.matmul(T_1p.reshape([bs, 1, 2]), temp_m).reshape([bs, 2, 1])

    m_2x3 = torch.matmul(M1, xyz)

    m_2x9 = torch.cat((m_2x6, m_2x3), dim=2)
    m_2x8 = m_2x9[:, :, :8]

    # construct b
    m_2x1 = -M1

    return m_2x8, m_2x1


def solve_DLT(delta):
    """
    :param S_4p: the locations of 4 points of source image; shape=[batch_size, 8]
    :param T_4p: the locations of 4 points of target image; shape=[batch_size, 8]
    :return:
    """
    S_4p = torch.tensor([[550, 250], [550, 350], [650, 250], [650, 350]], dtype=torch.float32).view(-1, 8)
    if torch.cuda.is_available():
        S_4p = S_4p.cuda()
    T_4p = S_4p + delta
    S_4p = T_4p - delta
    # A, b
    bs, num8 = S_4p.shape

    # split the four points
    # four points matrix
    M_p1, b1 = construct_point_pair_v2(S_4p[:, :2], T_4p[:, :2])
    M_p2, b2 = construct_point_pair_v2(S_4p[:, 2:4], T_4p[:, 2:4])
    M_p3, b3 = construct_point_pair_v2(S_4p[:, 4:6], T_4p[:, 4:6])
    M_p4, b4 = construct_point_pair_v2(S_4p[:, 6:8], T_4p[:, 6:8])

    A = torch.cat((M_p1, M_p2, M_p3, M_p4), dim=1)
    b = torch.cat((b1, b2, b3, b4), dim=1)

    # Ax = b
    A_inv = torch.inverse(A)
    H_8el = torch.matmul(A_inv, b).reshape(bs, 8)

    # add the last element 1
    ones = torch.ones([bs, 1], dtype=H_8el.dtype)
    if torch.cuda.is_available():
        ones = ones.cuda()
    H = torch.cat((H_8el, ones), dim=1).reshape([bs, 3, 3])

    return H


def get_grid(batch_size, H, W, start):
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()
    # if torch.cuda.is_available():
    #     grid = grid.cuda()
    grid[:, :2, :, :] = grid[:, :2, :, :] + start  # 加上patch在原图内的偏移量
    return grid


def get_flow(H_mat_mul, patch_indices, image_size_h=600, image_size_w=800):
    # (N, 6, 3, 3)
    batch_size = H_mat_mul.shape[0]
    divide = H_mat_mul.shape[1]
    H_mat_mul = torch.from_numpy(H_mat_mul.reshape(batch_size, divide, 3, 3))

    small_patch_sz = [image_size_h // divide, image_size_w]
    small = 1e-7

    H_mat_pool = torch.zeros(batch_size, image_size_h, image_size_w, 3, 3)

    # 计算光流场
    for i in range(divide):
        H_mat = H_mat_mul[:, i, :, :]

        if i == divide - 1:
            H_mat = H_mat.unsqueeze(1).unsqueeze(1).expand(batch_size, image_size_h - i * small_patch_sz[0],
                                                           image_size_w, 3, 3)
            H_mat_pool[:, i * small_patch_sz[0]:, ...] = H_mat
            continue

        H_mat = H_mat.unsqueeze(1).unsqueeze(1).expand(batch_size, small_patch_sz[0], image_size_w, 3, 3)
        H_mat_pool[:, i * small_patch_sz[0]:(i + 1) * small_patch_sz[0], ...] = H_mat

    pred_I2_index_warp = patch_indices.permute(0, 2, 3, 1).unsqueeze(4)
    pred_I2_index_warp = torch.matmul(H_mat_pool, pred_I2_index_warp).squeeze(-1).permute(0, 3, 1, 2)
    T_t = pred_I2_index_warp[:, 2:3, ...]
    smallers = 1e-6 * (1.0 - torch.ge(torch.abs(T_t), small).float())
    T_t = T_t + smallers  # 避免除0
    v1 = pred_I2_index_warp[:, 0:1, ...]
    v2 = pred_I2_index_warp[:, 1:2, ...]
    v1 = v1 / T_t
    v2 = v2 / T_t
    warp_index = torch.cat((v1, v2), 1)
    vgrid = patch_indices[:, :2, ...]

    flow = warp_index - vgrid

    return flow, vgrid


def transformer(I, vgrid, train=False):
    # I: Img, shape: batch_size, 1, full_h, full_w
    # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
    # outsize: (patch_h, patch_w)

    def _repeat(x, n_repeats):

        rep = torch.ones([n_repeats, ]).unsqueeze(0)
        rep = rep.int()
        x = x.int()

        x = torch.matmul(x.reshape([-1, 1]), rep)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size, scale_h):
        # x: x_grid_flat
        # y: y_grid_flat
        # out_size: same as im.size
        # scale_h: True if normalized
        # constants
        num_batch, num_channels, height, width = im.size()

        out_height, out_width = out_size[0], out_size[1]
        # zero = torch.zeros_like([],dtype='int32')
        zero = 0
        max_y = height - 1
        max_x = width - 1
        if scale_h:
            # scale indices from [-1, 1] to [0, width or height]
            # print('--Inter- scale_h:', scale_h)
            x = (x + 1.0) * (height) / 2.0
            y = (y + 1.0) * (width) / 2.0

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        dim1 = torch.from_numpy(np.array(width * height))
        dim2 = torch.from_numpy(np.array(width))

        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)  # 其实就是单纯标出batch中每个图的下标位置
        # base = torch.arange(0,num_batch) * dim1
        # base = base.reshape(-1, 1).repeat(1, out_height * out_width).reshape(-1).int()
        # 区别？expand不对数据进行拷贝 .reshape(-1,1).expand(-1,out_height * out_width).reshape(-1)
        if torch.cuda.is_available():
            dim2 = dim2.cuda()
            dim1 = dim1.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()
            x0 = x0.cuda()
            x1 = x1.cuda()
            base = base.cuda()
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _transform(I, vgrid, scale_h):

        C_img = I.shape[1]
        B, C, H, W = vgrid.size()

        x_s_flat = vgrid[:, 0, ...].reshape([-1])
        y_s_flat = vgrid[:, 1, ...].reshape([-1])
        out_size = vgrid.shape[2:]
        input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

        output = input_transformed.reshape([B, H, W, C_img])
        return output

    # scale_h = True
    output = _transform(I, vgrid, scale_h=False)
    output = output.permute(0, 3, 1, 2)
    return output


def dlt_spatial_transform(flow, data_img):
    img_indices = get_grid(batch_size=data_img.shape[0],
                           H=data_img.shape[2],
                           W=data_img.shape[3],
                           start=0)
    vgrid = img_indices[:, :2, ...].cuda()
    grid_warp = vgrid + flow
    warp_imgs = transformer(data_img, grid_warp)
    return warp_imgs
