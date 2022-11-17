from dgl.nn import edge_softmax
from dgl.nn import GraphConv
import dgl

import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.utils import expand_as_pair

config = configparser.ConfigParser()
config.read("./data_trans.cfg", encoding='utf-8')


type2num = {"zxAgeEdge": 9,
            'Zx_BasicEducation': 8,
            'Zx_BasicDegree': 7,
            'Zx_BasicGender': 4,
            'Zx_BasicMaritalStatus': 10,
            'Zx_BasicEmploymentType': 16,
            'email_type': 23,
            'Zx_ResideStatus': 11,
            'Zx_Occupation': 10,
            'Zx_Position': 6,
            'Zx_PostTitle': 5,
            'u_Zx_OccupationEmploymentType': 2,
            'Zx_CompanyType': 22,
            'Zx_GjjStatus': 4,
            'Zx_UnitProperty': 7,
            'Zx_QueryReason': 8,
            'Zx_QueryOrgType': 18,
            'Zx_LoanTransactionType': 17,
            'Zx_LoanAccStatus': 9,
            'Zx_LoanOrgType': 18,
            'Zx_LoanRepayMode': 14,
            'Zx_LoanCurrency': 3,
            'Zx_LoanFiveclasscode': 7,
            'Zx_LoanPledge': 10,
            'Zx_LoanSpecTraDataType': 16,
            'Zx_LoanJoinLoan': 4,
            'Zx_LoanPersonAccountType': 4,
            'Zx_LoanRepaymentType': 16,
            'Zx_LoanGrantForm': 3,
            'Zx_CreditAccStatus': 7,
            'Zx_CreditOrgType': 5,
            'Zx_CreditCurrency': 14,
            'Zx_CreditPledge': 10,
            'Zx_CreditRepaymentStatus': 13,
            'Zx_CreditPersonAccountType': 3,
            'Zx_PhoneType': 5}


#  -.-..-.-.-..-.-.-.  parse fea functions -.-.-.-.-.-.
def gen_dtId(x):
    x = x * 1.
    x[x < -3600] = -3600
    x[x > 3600] = 3600
    x += 3600
    x = x // 30
    dt_id = x.long().clone()
    # print("dt_id:", dt_id.shape)
    return dt_id


# -.-.-.-.-.-.-. parse fea function end .-.-.-.-.-.-.-.-


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        # print("Mish activation loaded..")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Dense(nn.Module):
    def __init__(self, fin, fout, act="mish", bn=True, dropout=0.1):
        super(Dense, self).__init__()

        self.linear = nn.Linear(fin, fout)

        self.act = act
        # if act == "tanh":
        #     self.actF = torch.tanh()
        if act == "mish":
            self.actF = Mish()
        if act == "relu":
            self.actF = nn.ReLU()

        self.has_bn = bn
        if bn:
            self.bn = nn.BatchNorm1d(fout)

        self.has_drop = (dropout > 0)
        if self.has_drop:
            self.drop = nn.Dropout(dropout, inplace=True)  # ～～～

    def forward(self, data):
        x = self.linear(data)

        if self.act is not None:
            if self.act == "tanh":
                x = torch.tanh(x)
            else:
                x = self.actF(x)

        if self.has_bn:
            shape = x.size()
            x = self.bn(x.view((-1, shape[-1]))).view(shape)

        if self.has_drop:
            x = self.drop(x)

        return x


class MHA(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dropout=0.1):
        super(MHA, self).__init__()
        self.mha = nn.MultiheadAttention(model_dim, num_heads, dropout)
        self.dense = Dense(model_dim, model_dim)

    def forward(self, q, k, v, mask=None):
        """

        :param q: [L, N, E]
        :param k: [L_k, N, E]
        :param v:
        :param mask: [N, L_k]
        :return: attn_output [N, L, E]
        """
        # pytorch 原生 Multihead Attention， key_padding_mask 全为True 返回全部为 nan
        if mask is not None:
            mask = mask.clone()
            mask[:, 0] = False

        attn_output, attn_output_weights = self.mha(q, k, v, mask)  # attn_output [L, N, E]
        # print("attn_output.isnan()", attn_output.isnan().sum())
        # print("attn_output_weights.isnan()", attn_output_weights.isnan().sum())
        attn_output = self.dense(attn_output)

        return attn_output.transpose(0, 1)  # [N, L, E]


class SimpleAttn(nn.Module):
    def __init__(self, model_dim=256):
        super(SimpleAttn, self).__init__()
        self.qw = nn.Linear(model_dim, model_dim)
        self.kw = nn.Linear(model_dim, model_dim)
        self.vw = nn.Linear(model_dim, model_dim)

    def forward(self, q, k, v, mask):
        """

        :param q: shape [b, x_l q_l, d]
        :param k: [b, x_l, v_l, d]
        :param v: [b, x_l, v_l, d]
        :param mask: [b, x_l, v_l]
        :return:
        """
        # print(mask)
        # print("simple mask->", mask.shape)
        q = self.qw(q)
        k = self.kw(k)
        v = self.vw(v)

        temp_attn = torch.matmul(q, k.transpose(-1, -2))  # [b, x_l, q_l, v_l]
        temp_attn = torch.max(temp_attn, dim=-2).values
        temp_attn = temp_attn / torch.norm(temp_attn, p=2, dim=-1, keepdim=True)  # [b, x_l, v_l]

        noWrods_line = ((~mask).sum(dim=-1) == 0)  # [b, x_l]
        # print("same shape->", temp_attn.shape == mask.shape)

        temp_attn = temp_attn.clone()
        temp_attn[mask] = -10.
        attn = torch.softmax(temp_attn, dim=-1).unsqueeze(-2)  # [b, x_l, 1, v_l]

        attn = attn.clone()
        attn[noWrods_line] = 0.

        out = torch.matmul(attn, v)  # [b, x_l, 1, d]

        return out


class Simple_GAt(nn.Module):
    def __init__(self, model_dim=256):
        super(Simple_GAt, self).__init__()
        self.aw = nn.Linear(2*model_dim, 1)
        self.vw = Dense(model_dim, model_dim//2, bn=False)  # , 128
        self.model_dim = model_dim

    def forward(self, graph, feat):

        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.apply_edges(fn.copy_u('x', 'src'))
            graph.edata['_'] = torch.zeros([graph.num_edges(), self.model_dim]).float().to(feat[0].device)
            graph.apply_edges(fn.v_add_e('x', '_', 'dst'))

            cat = torch.cat([graph.edata['src'], graph.edata['dst']
                             ], dim=-1)  # [, 512]
            a = torch.sigmoid(self.aw(cat))
            graph.edata['a'] = a

            graph.edata["a"] = edge_softmax(graph, graph.edata["a"])

            graph.srcdata['v'] = self.vw(feat_src)

            graph.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'h'))

            rst = graph.dstdata['h']

            return rst  # [, model_dim//2]


# --- v4 ---
class ProfileV4(nn.Module):
    def __init__(self, attr_embDim=32, stdAddrL1_embDim=32, stdAddrL2_embDim=32, stdAddrL3_embDim=64, word_embDim=128):
        super(ProfileV4, self).__init__()
        self.stdAddrL1_size = 68
        self.stdAddrL2_size = 666
        self.stdAddrL3_size = 6424

        self.word_size = 20005

        self.embed_edu = nn.Embedding(type2num["Zx_BasicEducation"], attr_embDim)
        self.embed_degree = nn.Embedding(type2num["Zx_BasicDegree"], attr_embDim)
        self.embed_gender = nn.Embedding(type2num["Zx_BasicGender"], attr_embDim)
        self.embed_mari = nn.Embedding(type2num["Zx_BasicMaritalStatus"], attr_embDim)
        self.embed_employmentType = nn.Embedding(type2num["Zx_BasicEmploymentType"], attr_embDim)
        self.embed_email_type = nn.Embedding(type2num["email_type"], attr_embDim)
        # self.embed_zxAgeEdge = nn.Embedding(type2num["zxAgeEdge"], attr_embDim)

        # self.embed_companyLevel = nn.Embedding(type2num["zxCompanyLevel"], attr_embDim)
        self.embed_resideStatus = nn.Embedding(type2num["Zx_ResideStatus"], attr_embDim)

        self.embed_occupation = nn.Embedding(type2num["Zx_Occupation"], attr_embDim)
        self.embed_companyType = nn.Embedding(type2num["Zx_CompanyType"], attr_embDim)
        self.embed_position = nn.Embedding(type2num["Zx_Position"], attr_embDim)
        self.embed_postTitle = nn.Embedding(type2num["Zx_PostTitle"], attr_embDim)

        self.embed_gjjStatus = nn.Embedding(type2num["Zx_GjjStatus"], attr_embDim)

        self.embed_phoneType = nn.Embedding(type2num["Zx_PhoneType"], attr_embDim)

        self.embed_stdAddrL1 = nn.Embedding(self.stdAddrL1_size + 1, stdAddrL1_embDim)  # 标准行政区域 省-嵌入
        self.embed_stdAddrL2 = nn.Embedding(self.stdAddrL2_size + 1, stdAddrL2_embDim)  # 标准行政区域 市-嵌入
        self.embed_stdAddrL3 = nn.Embedding(self.stdAddrL3_size + 1, stdAddrL3_embDim)  # 标准行政区域 区县-嵌入

        self.embed_words = nn.Embedding(self.word_size + 1, word_embDim)  # 普通文本 词嵌入

        self.embed_time = nn.Embedding(240 + 1, 128)  # 时间嵌入
        self.embed_word_position = nn.Parameter(torch.Tensor(20, 128))
        nn.init.normal_(self.embed_word_position)

        # identity
        self.mha_contactAddr = MHA(model_dim=128)
        self.mha_resideAddr = MHA(model_dim=128)
        self.identity_num_dense = Dense(2, 32)
        self.identity_dense = Dense(864, 128)

        # reside list
        self.reside_word_attn = SimpleAttn(model_dim=128)
        self.reside_updt_dense = Dense(1, 32)
        self.reside_dense = Dense(320, 256)
        # self.lstm_resideList = nn.LSTM(576, 256, bidirectional=True)
        self.dense_reside_vList = Dense(256, 128)

        # occupation list
        self.occupation_nameWord_attn = SimpleAttn(model_dim=128)
        self.occupation_addrWord_attn = SimpleAttn(model_dim=128)
        self.occup_updt_dense = Dense(1, 32)
        # self.lstm_occupationList = nn.LSTM(1184, 256, bidirectional=True)
        self.gru_occupationList = nn.GRU(672, 256, bidirectional=True)
        # self.occupation_dense = Dense(1184, 512)
        self.dense_occup_vList = Dense(512, 128)

        # hrf list
        self.hrf_word_attn = SimpleAttn(model_dim=128)
        self.hrf_num_dense = Dense(4, 32)
        # self.lstm_hrfList = nn.LSTM(704, 256, bidirectional=True)
        self.gru_hrfList = nn.GRU(384, 256, bidirectional=True)
        # self.hrf_dense = Dense(704, 512)
        self.dense_hrf_vList = Dense(512, 128)

        # summary
        self.summary_dense1 = Dense(42, 256)
        self.summary_dense2 = Dense(256, 128)
        self.summary_dense3 = Dense(128, 256)
        self.summary_dense4 = Dense(256, 128)

        # forward
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.mha_occ_hrf = MHA(model_dim=128)
        self.mha_hrf_occ = MHA(model_dim=128)

    def identity_forward(self, x):
        num_x = x[:, 0:2]
        num_x[num_x < 0] = 0.
        num_x[num_x > 99] = 99.
        num_x = torch.log10(num_x + 1.)  # [b, 2]

        id_x = x[:, 2:].long()
        # print("identity_forward:", id_x.shape)
        edu, deg, gen, mari, eplyTp, emlTp = id_x[:, 0], id_x[:, 1], id_x[:, 2], id_x[:, 3], id_x[:, 4], id_x[:, 5]
        prcL1, prcL2, prcL3 = id_x[:, 6], id_x[:, 7], id_x[:, 8]
        contactL1, contactL2, contactL3, contactWs = id_x[:, 9], id_x[:, 10], id_x[:, 11], id_x[:, 12:31]
        resideL1, resideL2, resideL3, resideWs = id_x[:, 31], id_x[:, 32], id_x[:, 33], id_x[:, 34:53]

        edu_embed = self.embed_edu(edu)  # [b, 32]
        deg_embed = self.embed_degree(deg)
        gen_embed = self.embed_gender(gen)
        mar_embed = self.embed_mari(mari)
        epTp_embed = self.embed_employmentType(eplyTp)
        elTp_embed = self.embed_email_type(emlTp)

        # 3 prcid Location
        prcL1_emb = self.embed_stdAddrL1(prcL1)  # [b, 32]
        prcL2_emb = self.embed_stdAddrL2(prcL2)  # [b, 32]
        prcL3_emb = self.embed_stdAddrL3(prcL3)  # [b, 64]

        # 2 Addr
        cStdL1_emb = self.embed_stdAddrL1(contactL1)  # [b, 32]
        cStdL2_emb = self.embed_stdAddrL2(contactL2)  # [b, 32]
        cStdL3_emb = self.embed_stdAddrL3(contactL3)  # [b, 64]
        cWords_emb = self.embed_words(contactWs)  # [b, 19, 128]
        cWords_mask = (contactWs == 0)  # [b, 19]
        cWords_mask = torch.cat(
            [torch.zeros(contactWs.shape[0], 1, dtype=torch.bool).to(cWords_mask.device), cWords_mask],
            dim=-1)  # [b, 20]
        # add std location to words seq
        c_stdAddr = torch.cat([cStdL1_emb, cStdL2_emb, cStdL3_emb], dim=-1).unsqueeze(1)  # [b, 1, 128]
        contact_wholeWords = torch.cat([c_stdAddr, cWords_emb], dim=1)  # [b, 20, 128]
        contact_wholeWords += self.embed_word_position

        cWords_mha = self.mha_contactAddr(contact_wholeWords.transpose(0, 1), contact_wholeWords.transpose(0, 1),
                                          contact_wholeWords.transpose(0, 1), cWords_mask)  # [b, 20, 128]
        cWords_v = torch.mean(cWords_mha, dim=1)  # [b, 128]

        rStdL1_emb = self.embed_stdAddrL1(resideL1)  # [b, 32]
        rStdL2_emb = self.embed_stdAddrL2(resideL2)  # [b, 32]
        rStdL3_emb = self.embed_stdAddrL3(resideL3)  # [b, 64]
        rWords_emb = self.embed_words(resideWs)  # [b, 19, 128]
        rWords_mask = (resideWs == 0)  # [b, 19]
        rWords_mask = torch.cat(
            [torch.zeros(resideWs.shape[0], 1, dtype=torch.bool).to(rWords_mask.device), rWords_mask],
            dim=-1)  # [b, 20]
        # add std location to words seq
        r_stdAddr = torch.cat([rStdL1_emb, rStdL2_emb, rStdL3_emb], dim=-1).unsqueeze(1)  # [b, 1, 128]
        reside_wholeWords = torch.cat([r_stdAddr, rWords_emb], dim=1)  # [b, 20, 128]
        reside_wholeWords += self.embed_word_position

        rWord_mha = self.mha_resideAddr(reside_wholeWords.transpose(0, 1), reside_wholeWords.transpose(0, 1),
                                        reside_wholeWords.transpose(0, 1), rWords_mask)  # [b, 20, 128]

        rWords_v = torch.mean(rWord_mha, dim=1)  # [b, 128]

        # numFea [b, 2]
        num_x = self.identity_num_dense(num_x)  # [b, 32]
        indentiy_v = torch.cat([num_x,
                                edu_embed, deg_embed, gen_embed, mar_embed, epTp_embed, elTp_embed,
                                prcL1_emb, prcL2_emb, prcL3_emb,
                                cStdL1_emb, cStdL2_emb, cStdL3_emb, cWords_v,
                                rStdL1_emb, rStdL2_emb, rStdL3_emb, rWords_v
                                ], dim=-1)  # x: [b, 864 = 32+ 32*6 + (32+32+64)*3 + 128*2 ]

        indentiy_v = self.identity_dense(indentiy_v)
        # print("indentiy_v.isnan", indentiy_v.isnan().sum())
        return indentiy_v  # [b, 128]

    def reside_forward(self, x):
        """

        :param x: shape [b, 5, 24]
        :return:
        """
        dt = x[:, :, 0]
        dt_id = gen_dtId(dt)  # [b, 5]
        dt_emb = self.embed_time(dt_id)  # [b, 5, 128]

        time_fea = x[:, :, 0:1]
        time_fea[time_fea < -9999] = -9999.
        time_fea[time_fea > 9999] = 9999.
        time_fea = torch.log10(time_fea+10000.)

        id_fea = x[:, :, 1:].long()
        # print("reside_forward:", id_fea.shape)
        reside_stat = id_fea[:, :, 0]
        resideL1, resideL2, resideL3 = id_fea[:, :, 1], id_fea[:, :, 2], id_fea[:, :, 3]
        resideWs = id_fea[:, :, 4:]

        stat_emb = self.embed_resideStatus(reside_stat)  # [b, 5, 32]

        stdL1_emb = self.embed_stdAddrL1(resideL1)  # [b, 5, 32]
        stdL2_emb = self.embed_stdAddrL2(resideL2)  # [b, 5, 32]
        stdL3_emb = self.embed_stdAddrL3(resideL3)  # [b, 5, 64]

        addrWs_emb = self.embed_words(resideWs)  # [b, 5, 19, 128]
        addrWs_emb += self.embed_word_position[1:, :]

        # 合并词向量，｜  为不同词 生成不同权重 的合并
        addrWs_mask = (resideWs == 0)
        addrWs_attn = self.reside_word_attn(addrWs_emb, addrWs_emb, addrWs_emb, addrWs_mask)  # [b, 5, 1, 128]
        addrWs_attn = addrWs_attn.squeeze(-2)  # [b, 5, 128]

        time_fea = self.reside_updt_dense(time_fea)
        x = torch.cat([time_fea, stat_emb, stdL1_emb, stdL2_emb, stdL3_emb, addrWs_attn],
                      dim=-1)  # [b, 5, 32+32+32+32+64 +128 =320]
        """
        bi-LSTM"""
        # self.lstm_resideList.flatten_parameters()
        # output, (h_n, c_n) = self.lstm_resideList(x.transpose(0, 1))
        # reside_vList = output.transpose(0, 1)  # [b, 5, hidden_size*2=512]

        reside_vList = self.reside_dense(x)

        reside_vList = self.dense_reside_vList(reside_vList)  # [b, 5, 128]

        return torch.mean(reside_vList + dt_emb, dim=1)  # [b, 128]

    def occupation_forward(self, x):
        """

        :param x: [b, 5, 49]
        :return:
        """
        dt = x[:, :, 0]
        dt_id = gen_dtId(dt)  # [b, 5]
        dt_emb = self.embed_time(dt_id)  # [b, 5, 128]

        time_fea = x[:, :, 0:1]
        time_fea[time_fea < -9999] = -9999.
        time_fea[time_fea > 9999] = 9999.
        time_fea = torch.log10(time_fea + 10000.)

        id_fea = x[:, :, 1:].long()
        # print("occupation_forward:", id_fea.shape)
        comTpList, occupList, positList, posttList, \
        caStdL1, caStdL2, caStdL3, \
        cnStdL1, cnStdL2, cnStdL3, \
        caWords, cnWords = id_fea[:, :, 0], id_fea[:, :, 1], id_fea[:, :, 2], id_fea[:, :, 3], \
                            id_fea[:, :, 4], id_fea[:, :, 5], id_fea[:, :, 6], \
                            id_fea[:, :, 7], id_fea[:, :, 8], id_fea[:, :, 9], \
                            id_fea[:, :, 10:29], id_fea[:, :, 29:48]

        comTpList[comTpList < 0] = 0
        comTpList_emb = self.embed_companyType(comTpList)  # [b, 5, 32]
        occupList_emb = self.embed_occupation(occupList)
        positList_emb = self.embed_position(positList)
        posttList_emb = self.embed_postTitle(posttList)

        caL1_emb = self.embed_stdAddrL1(caStdL1)  # [b, 5, 32]
        caL2_emb = self.embed_stdAddrL2(caStdL2)  # [b, 5, 32]
        caL3_emb = self.embed_stdAddrL3(caStdL3)  # [b, 5, 64]

        cnL1_emb = self.embed_stdAddrL1(cnStdL1)  # [b, 5, 32]
        cnL2_emb = self.embed_stdAddrL2(cnStdL2)  # [b, 5, 32]
        cnL3_emb = self.embed_stdAddrL3(cnStdL3)  # [b, 5, 64]

        caWs_emb = self.embed_words(caWords)  # [b, 5, 19, 128]
        caWs_emb += self.embed_word_position[1:, :]
        caWs_mask = (caWords == 0)
        caWs_attn = self.occupation_nameWord_attn(caWs_emb, caWs_emb, caWs_emb, caWs_mask)  # [b, 5, 1, 128]
        caWs_attn = caWs_attn.squeeze(-2)  # [b, 5, 128]

        cnWs_emb = self.embed_words(cnWords)  # [b, 5, 19 128]
        cnWs_emb += self.embed_word_position[1:, :]
        cnWs_mask = (cnWords == 0)
        cnWs_attn = self.occupation_addrWord_attn(cnWs_emb, cnWs_emb, cnWs_emb, cnWs_mask)  # [b, 5, 1, 128]
        cnWs_attn = cnWs_attn.squeeze(-2)  # [b, 5, 128]

        updtList = self.occup_updt_dense(time_fea)
        x = torch.cat([updtList, comTpList_emb, occupList_emb, positList_emb, posttList_emb,
                       caL1_emb, caL2_emb, caL3_emb, cnL1_emb, cnL2_emb, cnL3_emb,
                       caWs_attn, cnWs_attn], dim=-1)  # [b, 5, 32+32*4+ (32+32+64)*2 + 128*2=672]

        self.gru_occupationList.flatten_parameters()
        output, hn = self.gru_occupationList(x.transpose(0, 1))  # (seq_len, batch, num_directions * hidden_size)
        occup_vList = output.transpose(0, 1)  # (b, 5, 512)
        # occup_vList = self.occupation_dense(x)

        occup_vList = self.dense_occup_vList(occup_vList)  # [b, 5, 128]

        return occup_vList + dt_emb  # [b, 5, 128]

    def hrf_forward(self, x):
        """

        :param x: [b, 5, 28]
        :return:
        """
        dt = x[:, :, 0]
        dt_id = gen_dtId(dt)  # [b, 5]
        dt_emb = self.embed_time(dt_id)  # [b, 5, 128]

        num_x1 = x[:, :, 0:2]
        num_x1[num_x1 < -9999] = -9999.
        num_x1[num_x1 > 9999] = 9999.
        num_x1 = torch.log10(num_x1 + 10000.)

        num_x2 = x[:, :, 2:4]
        num_x2[num_x2 < 0] = 0.
        num_x2[num_x2 > 1] = 1.

        id_fea = x[:, :, 4:].long()
        # print("hrf_forward:", id_fea.shape)
        gsLandList, gjjStaList, \
        uStdAddrL1, StdAddrL2, StdAddrL3, uNameList = id_fea[:, :, 0], id_fea[:, :, 1], \
                                                        id_fea[:, :, 2], id_fea[:, :, 3], id_fea[:, :, 4], \
                                                        id_fea[:, :, 5:]

        gsLandList_emb = self.embed_stdAddrL3(gsLandList)  # [b, 5, 64]
        gjjStaList_emb = self.embed_gjjStatus(gjjStaList)  # [b, 5, 32]

        uStdL1_emb = self.embed_stdAddrL1(uStdAddrL1)  # [b, 5, 32]
        uStdL2_emb = self.embed_stdAddrL2(StdAddrL2)  # [b, 5, 32]
        uStdL3_emb = self.embed_stdAddrL3(StdAddrL3)  # [b, 5, 64]

        uName_emb = self.embed_words(uNameList)  # [b, 5, 19, 128]
        uName_emb += self.embed_word_position[1:, :]
        uName_mask = (uNameList == 0)
        uName_attn = self.hrf_word_attn(uName_emb, uName_emb, uName_emb, uName_mask)  # [b, 5, 1, 128]
        uName_attn = uName_attn.squeeze(-2)  # [b, 5, 128]

        num_x = torch.cat([num_x1, num_x2], dim=-1)
        num_x = self.hrf_num_dense(num_x)  # [b, 5, 32]
        x = torch.cat([num_x,
                       gsLandList_emb, gjjStaList_emb,
                       uStdL1_emb, uStdL2_emb, uStdL3_emb, uName_attn], dim=-1)
        # [b, 5, 32+ 64 +32 + 32+32+64 +128=384]
        # self.lstm_hrfList.flatten_parameters()
        # output, (h_n, c_n) = self.lstm_hrfList(x.transpose(0, 1))
        #
        # hrf_vList = output.transpose(0, 1)  # [b, 5, hidden_size*2=512]

        # hrf_vList = self.hrf_dense(x)
        self.gru_hrfList.flatten_parameters()
        output, hn = self.gru_hrfList(x.transpose(0, 1))  # (seq_len, batch, num_directions * hidden_size)
        hrf_vList = output.transpose(0, 1)  # (b, 5, 512)

        hrf_vList = self.dense_hrf_vList(hrf_vList)  # [b, 5, 128]

        return hrf_vList + dt_emb  # [b, 5, 128]

    def summary_forward(self, x):
        """

        :param x: [b, 42]
        :return:
        """
        time_fea = x[:, 0:6]
        time_fea[time_fea < -9999] = -9999.
        time_fea[time_fea > 9999] = 9999.
        time_fea = torch.log10(time_fea + 10000.)

        num_x = x[:, 6:]
        num_x[num_x < 0.] = 0.
        num_x[num_x > 9999999.] = 9999999.
        num_x = torch.log10(num_x + 1)

        summary_x = torch.cat([time_fea, num_x], dim=-1)  # [b, 42]
        x_before = self.summary_dense1(summary_x)
        x_before = self.summary_dense2(x_before)  # b, 128

        x_after = self.summary_dense3(x_before)
        x_after = self.summary_dense4(x_after)

        return x_before + x_after  # [b, 128]

    def forward(self, identity, reside, occupation, hrf, summary):

        # identity, reside, occupation, hrf, summary = self.parseProfile(x)
        # ---  ---
        identity_v = self.identity_forward(identity)  # [b, 128]
        reside_v = self.reside_forward(reside)  # [b, 128]
        occupation_v = self.occupation_forward(occupation)  # [b, 5, 128]
        hrf_v = self.hrf_forward(hrf)  # [b, 5, 128]
        summary_v = self.summary_forward(summary)  # [b, 128]

        occ_hrf_v = self.mha_occ_hrf(occupation_v.transpose(0, 1), hrf_v.transpose(0, 1),
                                     hrf_v.transpose(0, 1))  # [b, 5, 128]
        hrf_occ_v = self.mha_occ_hrf(hrf_v.transpose(0, 1), occupation_v.transpose(0, 1),
                                     occupation_v.transpose(0, 1))  # [b, 5, 128]

        return identity_v, reside_v, \
                   torch.mean(occupation_v, dim=1), torch.mean(hrf_v, dim=1), \
                   torch.mean(occ_hrf_v, dim=1), torch.mean(hrf_occ_v, dim=1), \
                   summary_v  #


class LCQEncoderV4(nn.Module):
    def __init__(self, attr_embDim=32):
        super(LCQEncoderV4, self).__init__()
        self.embed_LoanTransactionType = nn.Embedding(type2num["Zx_LoanTransactionType"], attr_embDim)
        self.embed_LoanOrgType = nn.Embedding(type2num["Zx_LoanOrgType"], attr_embDim)
        self.embed_LoanRepayMode = nn.Embedding(type2num["Zx_LoanRepayMode"], attr_embDim)
        self.embed_LoanFiveclasscode = nn.Embedding(type2num["Zx_LoanFiveclasscode"], attr_embDim)
        self.embed_LoanPledge = nn.Embedding(type2num["Zx_LoanPledge"], attr_embDim)
        self.embed_LoanCurrency = nn.Embedding(type2num["Zx_LoanCurrency"], attr_embDim)
        self.embed_LoanAccStatus = nn.Embedding(type2num["Zx_LoanAccStatus"], attr_embDim)
        self.embed_LoanPersonAccountType = nn.Embedding(type2num["Zx_LoanPersonAccountType"], attr_embDim)

        self.embed_CreditAccStatus = nn.Embedding(type2num["Zx_CreditAccStatus"], attr_embDim)
        self.embed_CreditOrgType = nn.Embedding(type2num["Zx_CreditOrgType"], attr_embDim)
        self.embed_CreditCurrency = nn.Embedding(type2num["Zx_CreditCurrency"], attr_embDim)
        self.embed_CreditPledge = nn.Embedding(type2num["Zx_CreditPledge"], attr_embDim)
        self.embed_CreditRepaymentStatus = nn.Embedding(type2num["Zx_CreditRepaymentStatus"], attr_embDim)
        self.embed_CreditPersonAccountType = nn.Embedding(type2num["Zx_CreditPersonAccountType"], attr_embDim)

        self.embed_queryReason = nn.Embedding(type2num["Zx_QueryReason"], attr_embDim)
        self.embed_queryOrgType = nn.Embedding(type2num["Zx_QueryOrgType"], attr_embDim)

        self.embed_time = nn.Embedding(240 + 1, 128)  # 时间嵌入

        self.loanTime_dense = Dense(5, 32)
        self.loanNum_dense = Dense(14, 128)
        self.loanLocal_dense = Dense(160, 128)
        self.loan_dense = Dense(384, 128)

        self.creditTime_dense = Dense(5, 32)
        self.creditNum_dense = Dense(13, 128)
        self.creditLocal_dense = Dense(160, 128)
        self.credit_dense = Dense(320, 128)

        self.queryTime_dense = Dense(1, 32)
        self.query_dense = Dense(96, 128)

        self.uniGlobalNum_dense = Dense(160, 256)
        self.uni_dense = Dense(704, 256)

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def loan_forward(self, loan_x):
        """

        :param loan_x: [b, l, 27]
        :return: [b, l, 256]
        """
        # dt = loan_x[:, :, 0]
        dt = loan_x[:, 0]
        dt_id = gen_dtId(dt)  # [b, l]
        dt_emb = self.embed_time(dt_id)  # [b, l, 128]

        # time_fea = loan_x[:, :, :5]  # [b, l, 5]
        time_fea = loan_x[:, :5]  # [b, l, 5]
        time_fea[time_fea < -9999.] = -9999.
        time_fea[time_fea > 9999.] = 9999.
        time_fea = torch.log10(time_fea + 10000)

        # print("---", time_fea.shape)
        x_time = self.loanTime_dense(time_fea)  # [b, l, 32]

        # num_fea = loan_x[:, :, 5:19]  # [b, l, 14]
        num_fea = loan_x[:, 5:19]  # [b, l, 14]
        num_fea[num_fea < 0] = 0.
        num_fea[num_fea > 999999] = 999999.
        num_fea = torch.log10(num_fea + 1)

        x_num = self.loanNum_dense(num_fea)  # [b, l, 128]

        x_local = torch.cat([x_time, x_num], dim=-1)  # [b, l, 160]
        x_local = self.loanLocal_dense(x_local)  # [b, l, 128]

        # id_fea = loan_x[:, :, 19:].long()
        id_fea = loan_x[:, 19:].long()
        # print("loan_forward:", id_fea.shape)
        # print("id_fea.shape:", id_fea.shape)

        accStList, fcdList, curcyList, pldgeList, rpmdList, orgTpList, txnTpList, pAccTypeList = \
            id_fea[:, 0], id_fea[:, 1], id_fea[:, 2], id_fea[:, 3], id_fea[:, 4], id_fea[:, 5], id_fea[:, 6], id_fea[:, 7]
            # id_fea[:, :, 0], id_fea[:, :, 1], id_fea[:, :, 2], id_fea[:, :, 3], id_fea[:, :, 4], id_fea[:, :, 5], id_fea[:, :, 6]

        accStList_embed = self.embed_LoanAccStatus(accStList)   # [b, 64, 32]
        fcdList_embed = self.embed_LoanFiveclasscode(fcdList)

        curcyList_embed = self.embed_LoanCurrency(curcyList)
        pldgeList_embed = self.embed_LoanPledge(pldgeList)
        rpmdList_embed = self.embed_LoanRepayMode(rpmdList)
        orgTpList_embed = self.embed_LoanOrgType(orgTpList)
        txnTpList_embed = self.embed_LoanTransactionType(txnTpList)
        pAccTypeList = self.embed_LoanPersonAccountType(pAccTypeList)

        x = torch.cat([x_local,
                       accStList_embed, fcdList_embed, curcyList_embed, pldgeList_embed, rpmdList_embed, orgTpList_embed, txnTpList_embed, pAccTypeList],
                      dim=-1)
        # [b, 64, 128 + 32*8=384]
        x = self.loan_dense(x)  # [, l, 128]

        return x + dt_emb  # [, 64, 128]

    def credit_forward(self, credit_x):
        """

        :param credit_x: [b, l, 24]
        :return: [b, l, 256]
        """
        # dt = credit_x[:, :, 0]
        dt = credit_x[:, 0]
        dt_id = gen_dtId(dt)  # [b, l]
        dt_emb = self.embed_time(dt_id)  # [b, l, 128]

        # time_fea = credit_x[:, :, :5]  # [b, l, 5]
        time_fea = credit_x[:, :5]  # [b, l, 5]
        time_fea[time_fea < -9999.] = -9999.
        time_fea[time_fea > 9999.] = 9999.
        time_fea = torch.log10(time_fea + 10000)

        x_time = self.creditTime_dense(time_fea)  # [b, l, 32]

        # num_fea = credit_x[:, :, 5:18]  # [b, l, 13]
        num_fea = credit_x[:, 5:18]  # [b, l, 13]
        num_fea[num_fea < 0] = 0.
        num_fea[num_fea > 999999] = 999999.
        num_fea = torch.log10(num_fea + 1)

        x_num = self.creditNum_dense(num_fea)  # [b, l, 128]

        x_local = torch.cat([x_time, x_num], dim=-1)  # [b, l, 160]
        x_local = self.creditLocal_dense(x_local)  # [b, l, 128]

        # id_fea = credit_x[:, :, 18:].long()  # [b, l, 5]
        id_fea = credit_x[:, 18:].long()  # [b, l, 5]
        # print("credit_forward:", id_fea.shape)
        accStList, pldgeList, orgTpList, curcyList, rpStaList, pAccTypeList = \
            id_fea[:, 0], id_fea[:, 1], id_fea[:, 2], id_fea[:, 3], id_fea[:, 4], id_fea[:, 5]
            # id_fea[:, :, 0], id_fea[:, :, 1], id_fea[:, :, 2], id_fea[:, :, 3], id_fea[:, :, 4]

        accStList_embed = self.embed_CreditAccStatus(accStList)  # [b, 30, 32]
        pldgeList_embed = self.embed_CreditPledge(pldgeList)
        orgTpList_embed = self.embed_CreditOrgType(orgTpList)
        curcyList_embed = self.embed_CreditCurrency(curcyList)
        rpStaList_embed = self.embed_CreditRepaymentStatus(rpStaList)
        pAccTypeList_embed = self.embed_CreditPersonAccountType(pAccTypeList)

        x = torch.cat([x_local,
                       accStList_embed, pldgeList_embed, orgTpList_embed, curcyList_embed, rpStaList_embed, pAccTypeList_embed],
                      dim=-1)
        # [b, l, 128 + 32*6=320]

        x = self.credit_dense(x)  # [b, l, 128]

        return x + dt_emb

    def query_forward(self, query_x):
        """

        :param query_x: [b, l, 3]
        :return: [b, l, 256]
        """
        # dt = query_x[:, :, 0]
        dt = query_x[:, 0]
        dt_id = gen_dtId(dt)  # [b, l]
        dt_emb = self.embed_time(dt_id)  # [b, l, 128]

        # time_fea = query_x[:, :, :1]
        time_fea = query_x[:, :1]
        time_fea[time_fea < -9999.] = -9999.
        time_fea[time_fea > 9999.] = 9999.
        time_fea = torch.log10(time_fea + 10000)

        x_time = self.queryTime_dense(time_fea)  # [b, l, 32]

        # qReasList, orgTpList = query_x[:, :, 1].long(), query_x[:, :, 2].long()
        qReasList, orgTpList = query_x[:, 1].long(), query_x[:, 2].long()
        # print("qReasList:", qReasList.shape)
        # print("orgTpList:", orgTpList.shape)

        # print("qReasList", qReasList)
        # print("qReasList max: ", torch.max(qReasList))
        qReasList_embed = self.embed_queryReason(qReasList)  # [b, l, 32]
        orgTpList_embed = self.embed_queryOrgType(orgTpList)

        x = torch.cat([x_time, qReasList_embed, orgTpList_embed], dim=-1)  # [b, l, 96= 32*2+32]
        x = self.query_dense(x)  # [b, l, 128]

        return x + dt_emb

    def forward(self, loan_x, credit_x, query_x):
        """

        :param x: [, l+c+q, 27]
        :return: [, l+c+q, 256]

        """
        loan_x = loan_x[:, :27]
        credit_x = credit_x[:, :24]
        query_x = query_x[:, :3]

        loan_e = self.loan_forward(loan_x)  # [, l, 128]
        # print("loan_e.shape:", loan_e.shape)
        credit_e = self.credit_forward(credit_x)  # [, c, 128]
        # print("credit_e.shape:", credit_e.shape)
        query_e = self.query_forward(query_x)  # [, q, 128]
        # print("query_e.shape:", query_e.shape)

        return loan_e, credit_e, query_e


class RepayEdgeEncoderV4(nn.Module):
    def __init__(self, repaySta_num=8, repaySta_dim=32):
        super(RepayEdgeEncoderV4, self).__init__()

        self.embed_repaySta = nn.Embedding(repaySta_num, repaySta_dim)
        self.num_dense = Dense(2, 32)

        self.repay_dense = Dense(64, 128)

    def forward(self, x):
        x[:, 0][x[:, 0] > 7] = 7.
        x_repaySta = x[:, 0].long()
        # print("RepayEdgeEncoder:", x_repaySta.shape)
        repaySta_emb = self.embed_repaySta(x_repaySta)  # [b, l, 32]

        # num_fea = x[:, :, 1:]  # [b, l, 2]
        num_fea = x[:, 1:]  # [b, l, 2]
        num_fea[num_fea < -9999999] = -9999999.
        num_fea[num_fea > 9999999] = 9999999.
        num_fea[num_fea < 0] = -torch.log10(1-num_fea[num_fea < 0])
        num_fea[num_fea >= 0] = torch.log10(1+num_fea[num_fea >= 0])

        x_num = self.num_dense(num_fea)  # [b, l, 32]

        x = torch.cat([repaySta_emb, x_num], dim=-1)  # [b, l, 64]
        x = self.repay_dense(x)  # [b, l, 128]

        return x


class GraphV4(nn.Module):
    def __init__(self, prof_path=None, fea_dim=128):
        super(GraphV4, self).__init__()

        self.profile_encoder = ProfileV4()
        self.lcq_encoder = LCQEncoderV4()
        self.repay_encoder = RepayEdgeEncoderV4()

        if prof_path is not None:
            self.profile_encoder.load_state_dict(torch.load(prof_path))
            print("loaded profile encoder parameters from: ", prof_path)

        self.mId_psi = nn.Parameter(torch.Tensor(61, 128))
        nn.init.normal_(self.mId_psi)  # nn.Parameter() 不进行初始化， 会出现 nan 值

        # --- init human define node (mId, tg, org)
        self.repay_att_uW = nn.Linear(fea_dim, fea_dim)
        self.repay_att_eW = nn.Linear(fea_dim, fea_dim)
        self.repay_dense = Dense(2*fea_dim, fea_dim, bn=False)

        self.conv_tg = GraphConv(fea_dim, fea_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.conv_org = GraphConv(fea_dim, fea_dim, norm='both', weight=True, bias=True, allow_zero_in_degree=True)

        # --- collecting messages of others nodes, for lcq update

        self.lcq_tg_att_0 = Simple_GAt(model_dim=fea_dim)  # return [, 64]
        self.lcq_org_att_0 = Simple_GAt(model_dim=fea_dim)
        self.lcq_mId_att_0 = Simple_GAt(model_dim=fea_dim)

        lcq_slots_dim = 128
        self.lcq_other_dense = Dense((fea_dim//2)*3, lcq_slots_dim, bn=False)
        self.lcq_update_dense = Dense(fea_dim+lcq_slots_dim, fea_dim, bn=False)

        # updating human define node (mId, tg, org)
        self.tLine_att_0 = Simple_GAt(model_dim=fea_dim)
        self.tg_lcq_att_0 = Simple_GAt(model_dim=fea_dim)
        self.tg_other_dense = Dense((fea_dim//2) * 2, 128, bn=False)
        self.tg_update_dense = Dense(fea_dim+128, fea_dim, bn=False)

        self.org_lcq_att_0 = Simple_GAt(model_dim=fea_dim)
        self.org_update_dense = Dense(fea_dim+(fea_dim//2), fea_dim, bn=False)

        self.mId_lcq_att_0 = Simple_GAt(model_dim=fea_dim)
        self.mId_update_dense = Dense(fea_dim+(fea_dim//2), fea_dim, bn=False)

        # --- classifier ---
        # self.mId_gru = nn.GRU(256, 128, bidirectional=True)
        # self.tG_gru = nn.GRU(256, 128, bidirectional=True)

        self.graph_dense = Dense(fea_dim*5, fea_dim*2, bn=False)  # 2 gru output 256, sum-mean org node, sum-mean tG node,
        self.pro_dense = Dense(fea_dim * 6, fea_dim*2, bn=False)
        self.cls = nn.Linear(fea_dim*5, 1)

    def forward(self, g):
        lcq_len = \
            torch.cat([g.nodes['self'].data['loan_len'],
                       g.nodes['self'].data['credit_len'],
                       g.nodes['self'].data['query_len']],
                      dim=-1).cpu().numpy().tolist()  # [b, 3]

        loans = []
        credits = []
        querys = []
        lcq_xs = g.nodes['lcq'].data['x_0']

        s = 0
        for ll, cl, ql in lcq_len:
            loans.append(lcq_xs[s:s+ll, :])
            s += ll
            credits.append(lcq_xs[s:s+cl, :])
            s += cl
            querys.append(lcq_xs[s:s+ql, :])
            s += ql

        loans_x = torch.cat(loans, dim=0)
        credits_x = torch.cat(credits, dim=0)
        querys_x = torch.cat(querys, dim=0)

        loans_x, credits_x, querys_x = self.lcq_encoder(loans_x, credits_x, querys_x)

        ls = 0
        cs = 0
        qs = 0
        lcq_e = []
        for ll, cl, ql in lcq_len:
            lcq_e.append(loans_x[ls:ls+ll, :])
            ls += ll
            lcq_e.append(credits_x[cs:cs+cl, :])
            cs += cl
            lcq_e.append(querys_x[qs:qs+ql, :])
            qs += ql

        lcq_e = torch.cat(lcq_e, dim=0)
        g.nodes['lcq'].data['x'] = lcq_e

        # g.nodes['self'].data['x'] = None
        identity_v, reside_v, occupation_v, hrf_v, occ_hrf_v, hrf_occ_v, summary_v = \
            self.profile_encoder(
                g.nodes['self'].data['identity'],
                g.nodes['self'].data['reside'], g.nodes['self'].data['occ'], g.nodes['self'].data['hrf'],
                g.nodes['self'].data['summary']
            )

        # print("-- debug -- ", g.edges['repay'].data['e_0'].shape)
        if g.edges['repay'].data['e_0'].shape[0] > 0:
            g.edges['repay'].data['x'] = self.repay_encoder(g.edges['repay'].data['e_0'])
        else:
            g.edges['repay'].data['x'] = torch.Tensor([])

        g.nodes['mIdx'].data['x'] = self.mId_psi.repeat(g.batch_size, 1)

        # --- graph computing ---
        # 1、初始化 mIdx node

        if len(g.edges['repay'].data['x']) > 0:

            g['repay'].nodes['lcq'].data['x_'] = self.repay_att_uW(g['repay'].nodes['lcq'].data['x'])
            g['repay'].edges['repay'].data['x_'] = self.repay_att_eW(g.edges['repay'].data['x'])
            g['repay'].apply_edges(fn.u_dot_e('x_', 'x_', 'a'))
            g['repay'].edata["a"] = edge_softmax(g['repay'], g['repay'].edata["a"])

            g['repay'].apply_edges(fn.copy_u('x', 'xSrc'))
            repay_x_xSrc = torch.cat([g.edges['repay'].data['x'], g.edges['repay'].data['xSrc']], dim=-1)  # [, 512]
            g.edges['repay'].data['x'] = self.repay_dense(repay_x_xSrc)  # [, 256]

            g['repay'].edata["ae"] = g['repay'].edata["a"] * g['repay'].edata["x"]

            g['repay'].update_all(fn.copy_e("ae", "att"), fn.sum('att', 'x_'))
            # --- 61月份位置嵌入 ---
            g['repay'].nodes['mIdx'].data['x'] = g['repay'].nodes['mIdx'].data['x'] + g['repay'].nodes['mIdx'].data['x_']

        # --- mIdx node 初始化完成 ---

        # --- lcq -> tG 初始化---
        g['tAgg'].nodes['tGroup'].data['x'] = self.conv_tg(
            g['tAgg'],
            (g['tAgg'].nodes['lcq'].data['x'], torch.zeros([g['tAgg'].num_nodes('tGroup'), 256]).float())
        )
        # --- lcq <-> tG end ---
        # --- lcq -> org 初始化 ---
        g['orgAgg'].nodes['org'].data['x'] = self.conv_org(
            g['orgAgg'],
            (g['orgAgg'].nodes['lcq'].data['x'], torch.zeros([g['orgAgg'].num_nodes('org'), 256]).float())
        )

        # --- occ <-> hrf ---

        # --- loop ---
        # for _ in range(3):
        for _ in range(2):
            # 1. 从 其他节点 -> 更新 lcq

            g.nodes['lcq'].data['h_tg'] = self.lcq_tg_att_0(g['inv_tAgg'],
                                                            (g.nodes['tGroup'].data['x'], g.nodes['lcq'].data['x']))

            g.nodes['lcq'].data['h_org'] = self.lcq_org_att_0(g['inv_orgAgg'],
                                                              (g.nodes['org'].data['x'], g.nodes['lcq'].data['x']))

            g.nodes['lcq'].data['h_mId'] = self.lcq_mId_att_0(g['inv_repay'],
                                                              (g.nodes['mIdx'].data['x'], g.nodes['lcq'].data['x']))

            # update lcq
            lcq_others = self.lcq_other_dense(
                torch.cat([
                    g.nodes['lcq'].data['h_tg'],
                    g.nodes['lcq'].data['h_org'],
                    g.nodes['lcq'].data['h_mId']
                ], dim=-1)
            )  # [, 128]

            g.nodes['lcq'].data['x'] = self.lcq_update_dense(
                torch.cat([
                    g.nodes['lcq'].data['x'],
                    lcq_others
                ], dim=-1)
            )  # [, 128]

            # 2. 从 lcq -> 更新 其他节点

            tg_tg = self.tLine_att_0(g['tLine'],
                                     (g.nodes['tGroup'].data['x'], g.nodes['tGroup'].data['x']))  # , 128
            tg_lcq = self.tg_lcq_att_0(g['tAgg'],
                                       (g.nodes['lcq'].data['x'], g.nodes['tGroup'].data['x']))  # , 128
            tg_others = self.tg_other_dense(
                torch.cat([tg_tg, tg_lcq], dim=-1)
            )  # p[, 128]

            g.nodes['tGroup'].data['x'] = self.tg_update_dense(
                torch.cat([
                    g.nodes['tGroup'].data['x'],
                    tg_others
                ], dim=-1)
            )  # , 128

            # lcq -> org
            org_lcq = self.org_lcq_att_0(g['orgAgg'],
                                         (g.nodes['lcq'].data['x'], g.nodes['org'].data['x']))
            g.nodes['org'].data['x'] = self.org_update_dense(
                torch.cat([
                    g.nodes['org'].data['x'],
                    org_lcq
                ], dim=-1)
            )  # [, 128]

            # lcq -> mId
            mId_lcq = self.mId_lcq_att_0(g['repay'],
                                         (g.nodes['lcq'].data['x'], g.nodes['mIdx'].data['x']))
            g.nodes['mIdx'].data['x'] = self.mId_update_dense(
                torch.cat([
                    g.nodes['mIdx'].data['x'],
                    mId_lcq
                ], dim=-1)
            )  # [, 128]

        repay_v = dgl.readout_nodes(g, 'x', op='mean', ntype='mIdx')  # [b, d]
        tg_v_mean = dgl.readout_nodes(g, 'x', op='mean', ntype='tGroup')  # [b, d]
        tg_v_sum = dgl.readout_nodes(g, 'x', op='sum', ntype='tGroup')  # [b, d]
        org_v_mean = dgl.readout_nodes(g, 'x', op='mean', ntype='org')  # [b, d]
        org_v_sum = dgl.readout_nodes(g, 'x', op='sum', ntype='org')  # [b, d]

        graph_v = torch.cat([repay_v, tg_v_mean, tg_v_sum, org_v_mean, org_v_sum], dim=-1)  # [b, 5d]
        graph_v = self.graph_dense(graph_v)  # [b, 2d]

        pro_v = torch.cat([identity_v, reside_v, occupation_v, hrf_v, occ_hrf_v, hrf_occ_v], dim=-1)  # [b, 6d]
        pro_v = self.pro_dense(pro_v)  # [b, 2d]

        all_v = torch.cat([graph_v, pro_v, summary_v], dim=-1)  # [b, 5d]

        logits = self.cls(all_v)  # [b, 1]

        return logits
