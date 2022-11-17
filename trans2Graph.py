import json
import configparser
import torch
import dgl
import numpy as np
from zxMidData import ZxMidDataCenter
from genGraphFea import GenFea

config = configparser.ConfigParser()
config.read("./data_trans.cfg", encoding='utf-8')

genFea = GenFea()


def mapParseLine(x):

    josOb = json.loads(x)
    data = ZxMidDataCenter(josOb, config)

    reportDate = data.getReportDate()
    prcid = data.getPrcid()

    identity = data.proMidDataIdentity()

    reside = data.proMidDataReside()
    occupation = data.proMidDataOccupation()
    hfrList = data.proMidDataHfrList()

    x = (identity, reside, occupation, hfrList, prcid)
    profileFea = genFea.genProfileFea(x)   # 560

    query_x = data.proMidDataQuery()
    loan_x = data.proMidDataLoan()
    credit_x = data.proMidDataCredit()

    # repay_node_num, lcq_node_num, tg_node_num, org_node_num, \
    # l_list, c_list, q_list, \
    # lc2m_src, lc2m_dst, lc2m_edge, \
    # tg_src, tg_dst, \
    # org_src, org_dst = genFea.genGraphFea_v2(query_x, loan_x, credit_x, reportDate)

    repay_node_num, lcq_node_num, tg_node_num, org_node_num, \
    l_list, c_list, q_list, \
    lc2m_src, lc2m_dst, lc2m_edge, \
    tg_src, tg_dst, \
    org_src, org_dst = genFea.genGraphFea_v3(query_x, loan_x, credit_x, reportDate)

    summry = data.proMidDataSummary()
    summry_fea = genFea.genSummryFea(summry, reportDate)   # 42

    # ---   解析profile   ---
    profile_identity = profileFea[:55]
    res_profile_identity = profile_identity
    # res_profile_identity = profile_identity[2:] + profile_identity[:2]  # 55

    profile_reside = profileFea[55: 175]
    resi_updt = profile_reside[:5]
    resi_stat = profile_reside[5:10]
    resi_addr_ids = profile_reside[10:120]

    reside_list = \
        np.concatenate([np.array(resi_updt).reshape([5, 1]),
                        np.array(resi_stat).reshape([5, 1]),
                        np.array(resi_addr_ids).reshape([5, 22])], axis=1)
    res_profile_reside = reside_list.tolist()  # 120 = 24 *5
    # res_profile_reside = reside_list.reshape([120]).tolist()  # 120 = 24 *5

    profile_occup = profileFea[175: 420]
    occ_dt = np.array(profile_occup[:5]).reshape([5, 1])
    occ_companyType = np.array(profile_occup[5:10]).reshape([5, 1])
    occ_pation = np.array(profile_occup[10:15]).reshape([5, 1])
    occ_position = np.array(profile_occup[15:20]).reshape([5, 1])
    occ_postTitle = np.array(profile_occup[20:25]).reshape([5, 1])

    occ_ca = np.array(profile_occup[25:135]).reshape([5, 22])
    occ_stcaL1 = occ_ca[:, 0].reshape([5, 1])
    occ_stcaL2 = occ_ca[:, 1].reshape([5, 1])
    occ_stcaL3 = occ_ca[:, 2].reshape([5, 1])
    occ_caid = occ_ca[:, 3:]  # [5, 19]

    occ_cn = np.array(profile_occup[135:245]).reshape([5, 22])
    occ_stcnL1 = occ_cn[:, 0].reshape([5, 1])
    occ_stcnL2 = occ_cn[:, 1].reshape([5, 1])
    occ_stcnL3 = occ_cn[:, 2].reshape([5, 1])
    occ_cnid = occ_cn[:, 3:]  # [5, 19]

    occ_list = np.concatenate([occ_dt, occ_companyType, occ_pation, occ_position, occ_postTitle,
                               occ_stcaL1, occ_stcaL2, occ_stcaL3, occ_stcnL1, occ_stcnL2, occ_stcnL3,
                               occ_caid, occ_cnid], axis=1)
    res_profile_occ = occ_list.tolist()  # 245 = 49 * 5
    # res_profile_occ = occ_list.reshape([245]).tolist()  # 245 = 49 * 5

    profile_hrf = profileFea[420: 560]
    hrf_dt = np.array(profile_hrf[:5]).reshape([5, 1])
    hrf_amt = np.array(profile_hrf[5:10]).reshape([5, 1])
    hrf_pr = np.array(profile_hrf[10:15]).reshape([5, 1])
    hrf_ur = np.array(profile_hrf[15:20]).reshape([5, 1])
    hrf_gl = np.array(profile_hrf[20:25]).reshape([5, 1])
    hrf_sta = np.array(profile_hrf[25:30]).reshape([5, 1])
    hrf_unt = np.array(profile_hrf[30:140]).reshape([5, 22])

    hrf_list = np.concatenate([hrf_dt, hrf_amt, hrf_pr, hrf_ur, hrf_gl, hrf_sta,
                               hrf_unt], axis=1)
    res_profile_hrf = hrf_list.tolist()  # 140 = 28 *5

    result_str = (prcid, reportDate, \
        res_profile_identity, res_profile_reside, res_profile_occ, res_profile_hrf, \
        summry_fea, \
        repay_node_num, lcq_node_num, tg_node_num, org_node_num, \
        l_list, c_list, q_list, \
        lc2m_src, lc2m_dst, lc2m_edge, \
        tg_src, tg_dst, \
        org_src, org_dst)
    # 7 + 14 = 21

    return result_str


def transave_v3(items):
    dt, prcid, \
    identity, reside, occ, hrf, summary, \
    repay_node_num, lcq_node_num, tg_node_num, org_node_num, \
    l_list, c_list, q_list, \
    lc2m_src, lc2m_dst, lc2m_edge, \
    tg_src, tg_dst, \
    org_src, org_dst = items

    # --- generate graph ---
    l_node_num = len(l_list)
    c_node_num = len(c_list)
    q_node_num = len(q_list)
    num_nodes_dict = {'lcq': lcq_node_num,
                      'mIdx': repay_node_num,
                      'tGroup': tg_node_num, 'org': org_node_num,
                      'self': 1
                      }
    # --- edge and edge fea ---
    e_repay1, e_repay2 = torch.Tensor(lc2m_src).long(), torch.Tensor(lc2m_dst).long()
    eFea_repay = torch.Tensor(lc2m_edge).float()

    e_tg1, e_tg2 = torch.Tensor(tg_src).long(), torch.Tensor(tg_dst).long()
    e_org1, e_org2 = torch.Tensor(org_src).long(), torch.Tensor(org_dst).long()

    # --- node fea ---
    l_list, c_list, q_list = torch.Tensor(l_list).float(), \
                             torch.Tensor(c_list).float(), torch.Tensor(q_list).float()
    # lcq node feature padding
    lcq_dim = 27
    if len(c_list) > 0:
        c_list = torch.cat([c_list,
                            torch.zeros([len(c_list), lcq_dim-c_list.shape[-1]]).float()], dim=-1)
    if len(q_list) > 0:
        q_list = torch.cat([q_list,
                            torch.zeros([len(q_list), lcq_dim-q_list.shape[-1]]).float()], dim=-1)
    lcq_list = torch.cat([l_list, c_list, q_list], dim=0)  # [lcq_num, 26]

    identity = torch.Tensor(identity).float().unsqueeze(0)
    reside = torch.Tensor(reside).float().unsqueeze(0)
    occ = torch.Tensor(occ).float().unsqueeze(0)
    hrf = torch.Tensor(hrf).float().unsqueeze(0)

    summary = torch.Tensor(summary).float().unsqueeze(0)
    # print(list(range(tg_node_num - 1)) + list(range(1, tg_node_num)))
    data_dict = {
        ('self', 'basic', 'self'): (torch.Tensor([0]).long(), torch.Tensor([0]).long()),
        # ('mIdx', 'loop', 'mIdx'): (torch.Tensor([0]).long(), torch.Tensor([0]).long()),

        ('lcq', 'repay', 'mIdx'): (e_repay1, e_repay2),
        ('mIdx', 'inv_repay', 'lcq'): (e_repay2, e_repay1),

        ('lcq', 'tAgg', 'tGroup'): (e_tg1, e_tg2),
        ('tGroup', 'inv_tAgg', 'lcq'): (e_tg2, e_tg1),
        ('tGroup', 'tLine', 'tGroup'): (torch.Tensor(
            list(range(tg_node_num - 1)) + list(range(1, tg_node_num))).long(),
                                        torch.Tensor(
                                            list(range(1, tg_node_num)) + list(range(tg_node_num - 1))
                                        ).long()),

        ('lcq', 'orgAgg', 'org'): (e_org1, e_org2),
        ('org', 'inv_orgAgg', 'lcq'): (e_org2, e_org1)
    }

    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict, idtype=None, device=None)

    g.nodes['lcq'].data['x_0'] = lcq_list
    g.nodes['lcq'].data['type'] = torch.Tensor([0]*l_node_num + [1]*c_node_num + [2]*q_node_num)

    g.edges['repay'].data['e_0'] = eFea_repay

    g.nodes['self'].data['identity'] = identity  # [1, d_i]
    g.nodes['self'].data['reside'] = reside  # [1, 5, d_r]
    g.nodes['self'].data['occ'] = occ  # [1, 5, d_o]
    g.nodes['self'].data['hrf'] = hrf
    g.nodes['self'].data['summary'] = summary

    g.nodes['self'].data['loan_len'] = torch.Tensor([[l_node_num]]).long()
    g.nodes['self'].data['credit_len'] = torch.Tensor([[c_node_num]]).long()
    g.nodes['self'].data['query_len'] = torch.Tensor([[q_node_num]]).long()

    return g


def getFea(x):
    result_items = mapParseLine(x)
    return transave_v3(result_items)
