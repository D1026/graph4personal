import configparser
import cpca
import jieba
import datetime
import numpy as np
import copy
from collections import defaultdict

from tokenizer import MyBertTokenizer

key2config = {"education": "Zx_BasicEducation", "degree": "Zx_BasicDegree", "gender": "Zx_BasicGender",
              "maritalStatus": "Zx_BasicMaritalStatus", "employmentType": "Zx_BasicEmploymentType",
              "email_type": "email_type",
              "resideStatus": "Zx_ResideStatus",
              "companyType": "Zx_CompanyType",
              "occupation": "Zx_Occupation", "position": "Zx_Position", "postTitle": "Zx_PostTitle",
              "gjjStatus": "Zx_GjjStatus",
              "phoneType": "Zx_PhoneType",

              "queryReason": "Zx_QueryReason", "queryOrgType": "Zx_QueryOrgType", "queryOperator": None,

              "transactionType": "Zx_LoanTransactionType", "loanAccStatus": "Zx_LoanAccStatus",
              "loanOrgType": "Zx_LoanOrgType", "loanRepayMode": "Zx_LoanRepayMode", "loanCurrency": "Zx_LoanCurrency",
              "fiveclasscode": "Zx_LoanFiveclasscode", "loanPledge": "Zx_LoanPledge",
              "loanPersonAccountType": "Zx_LoanPersonAccountType",

              "repaymentStatus": "Zx_CreditRepaymentStatus", "creditPledge": "Zx_CreditPledge",
              "creditOrgType": "Zx_CreditOrgType",
              "creditCurrency": "Zx_CreditCurrency", "creditAccStatus": "Zx_CreditAccStatus",
              "creditPersonAccountType": "Zx_CreditPersonAccountType"
              }


class GenFea(object):
    def __init__(self):

        self.config = self.getConfig()
        # for identity part
        self.edu2ids = self.values2id("education")
        # print(edu2ids)
        self.degree2ids = self.values2id("degree")
        # print(degree2ids)
        self.gender2ids = self.values2id("gender")
        # print(gender2ids)
        self.maris2ids = self.values2id("maritalStatus")
        # print(maris2ids)
        self.empTp2ids = self.values2id("employmentType")
        # print(empTp2ids)
        self.emlTp2ids = self.values2id("email_type")
        # print(emlTp2ids)

        # for reside part
        self.resiStat2id = self.values2id("resideStatus")

        # for occupation part
        self.compTp2id = self.values2id("companyType")
        self.occup2id = self.values2id("occupation")
        self.position2id = self.values2id("position")
        self.posTt2id = self.values2id("postTitle")

        # for hrf part
        self.gjjStat2id = self.values2id("gjjStatus")

        # std addr Dict
        self.dtct_dict, self.city_dict, self.prov_dict = self.std3Addr2ids()

        # for tokenizer
        self.jiebaDic, self.tokenizer = self.get_jiebaDic_tokenizer()

        self.loanList_keep_len = 64
        self.creditList_keep_len = 30
        self.queryList_keep_len = 30
        self.queryReason2id = self.values2id("queryReason")
        self.queryOrgType2id = self.values2id("queryOrgType")

        self.loanAccStatus2id = self.values2id("loanAccStatus")
        self.transactionType2id = self.values2id("transactionType")
        self.loanOrgType2id = self.values2id("loanOrgType")
        self.loanRepayMode2id = self.values2id("loanRepayMode")
        self.loanCurrency2id = self.values2id("loanCurrency")
        self.fiveclasscode2id = self.values2id("fiveclasscode")
        self.loanPledge2id = self.values2id("loanPledge")
        self.loanPersonAccountType2id = self.values2id("loanPersonAccountType")

        self.creditAccStatus2id = self.values2id("creditAccStatus")
        self.repaymentStatus2id = self.values2id("repaymentStatus")
        self.creditPledge2id = self.values2id("creditPledge")
        self.creditOrgType2id = self.values2id("creditOrgType")
        self.creditCurrency2id = self.values2id("creditCurrency")
        self.creditPersonAccountType2id = self.values2id("creditPersonAccountType")

        # --- for onLine env ---
        self.tk = tk = jieba.Tokenizer()
        tk.tmp_dir = './'
        tk.initialize()
        # tk.load_userdict('./data/MyModel/path/to/my_userdict')

    # -.-.-.-.- initial tools -.-.-.-.-
    def values2id(self, keyName):
        this_dict = dict(self.config[key2config[keyName]])
        this_vlaues = sorted(list(set(this_dict.values())))
        value2ids = dict()
        for id, val in enumerate(this_vlaues, start=0):
            value2ids[val] = id
        return value2ids

    def getConfig(self):
        config = configparser.ConfigParser()
        config.read("./data_trans.cfg", encoding='utf-8')
        return config

    def std3Addr2ids(self):
        """
        :return: { stdAddr : id [1:),
                   stdCode : id [1:)
                    }
        """
        prov_dict = dict()
        city_dict = dict()
        dtct_dict = dict()
        prov_i = 1
        city_i = 1
        dtct_i = 1
        prov = ''
        city = ''
        with open('./addr_dict.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                code = line[0].strip()
                name = line[1].strip()
                if code[-4:] == '0000':
                    prov_dict[name] = prov_i
                    prov_dict[code[:2]] = prov_i
                    prov_i += 1

                    prov = name
                    city = ''
                    dtct_dict[prov] = dtct_i
                    dtct_dict[code] = dtct_i
                    dtct_i += 1
                    continue
                if code[-2:] == '00':
                    city = name
                    city_dict[prov + city] = city_i
                    city_dict[code[:4]] = city_i
                    city_i += 1

                    city = name
                    dtct_dict[prov + city] = dtct_i
                    dtct_dict[code] = dtct_i
                    dtct_i += 1
                    continue
                else:
                    dtct_dict[prov + city + name] = dtct_i
                    dtct_dict[code] = dtct_i
                    dtct_i += 1

        return dtct_dict, city_dict, prov_dict

    def get_jiebaDic_tokenizer(self):
        jieba_idx = 1
        with open('./jieba_words.txt', 'r') as f:
            lines = f.readlines()
        jieba_word2id = dict()
        for l in lines:
            w, c = l.strip().split()
            jieba_word2id[w] = jieba_idx
            jieba_idx += 1

        # if jieba cut word not in jieba word dict, then using bert tokenizer. there initialize a bert tokenizer
        tokenizer = MyBertTokenizer('./chinese-vocab.txt', idx_start=jieba_idx)

        return jieba_word2id, tokenizer

    # -.-.-.-.- function tools -.-.-.-.-
    def getMonthIndex(self, m_start, m_end):
        """

        :param m_start: str like "201906"
        :param m_end: str like "202012"
        :return: int
        """
        y_s, m_s = int(m_start[:4]), int(m_start[4:6])
        y_e, m_e = int(m_end[:4]), int(m_end[4:6])

        delt_m = 0

        delt_m += (y_e - y_s) * 12
        delt_m += (m_e - m_s)

        return delt_m

    def text2ids(self, text):
        # create jieba word-idx dict
        words = list(jieba.cut(text))
        ids = []
        for word in words:
            if word in self.jiebaDic:
                ids.append(self.jiebaDic[word])
            else:
                ids.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize([word])[0]))
        return ids

    def stdAddr_words(self, textList):
        """
        用于处理报告中的自然语言文本（包括公司名称），返回标准行政区域地址ID+文本id 序列；

        :param textList: a list of text(address or company name etc,)
        :return: standard address ID (省,市,区/县）+ 地址分词ID -> 3 + 19 = 22 dim
        """
        # print("len(textList): ", len(textList))
        # print("textList: ", textList)
        if textList == ["", "", "", "", ""]:
            textList = ["None", "None", "None", "None", "None"]

        df = cpca.transform(textList, open_warning=False)

        # df['省'] 覆盖直辖市前，提取 prov ID
        prov = df["省"].tolist().copy()
        stdProvIds = []
        for i in range(len(prov)):
            stdProvIds.append(self.prov_dict.get(prov[i], 0))

        # 市级 ID， 直辖市不在city dict中， 仅有地市
        city = df["市"].tolist().copy()
        stdCityIds = []
        for i in range(len(city)):
            stdCityIds.append(self.city_dict.get((prov[i] + city[i]), 0))

        df['省'].replace("北京市", "", inplace=True)
        df['省'].replace("上海市", "", inplace=True)
        df['省'].replace("天津市", "", inplace=True)
        df['省'].replace("重庆市", "", inplace=True)

        # 区县级ID
        prov = df["省"].tolist().copy()
        city = df["市"].tolist().copy()
        area = df["区"].tolist().copy()
        dizhi = df["地址"].fillna("").tolist().copy()
        # stdAddrDict = stdAddr2ids()
        stdAdrrIds = []
        for i in range(len(prov)):
            stdAdrrIds.append(self.dtct_dict.get((prov[i] + city[i] + area[i]), 0))

        # print(dizhi)
        addrIds = [self.text2ids(_) for _ in dizhi]

        maxLen_txt = 19
        addrIds_padd = []
        for ids in addrIds:
            if len(ids) < maxLen_txt:
                temp = ids + [0] * (maxLen_txt - len(ids))
            else:
                temp = ids[-maxLen_txt:]
            addrIds_padd.append(temp.copy())
        # print(np.array(stdAdrrIds).reshape(-1, 1))
        result = np.concatenate([np.array(stdProvIds).reshape(-1, 1),
                                 np.array(stdCityIds).reshape(-1, 1),
                                 np.array(stdAdrrIds).reshape(-1, 1),
                                 np.array(addrIds_padd)], axis=1)  # [5, 22]
        result = result.tolist()
        return result

    # -.-.-.-.- generating api -.-.-.-.-
    def genProfileFea(self, x):
        """
        :param config:
        :return: proFile feature
        """
        xIdentity, xReside, xOccup, xHfr, prcid = x

        # *-*-*-*-*-* proMidData Identity d= 2 + 6 + 3 + 22x2 = 55 *-*-*-*-*
        identity_dict = {
            "zxAge": 0,
            "zxAgeRound": 0,

            "education": 0,
            "degree": 0,
            "gender": 0,
            "maritalStatus": 0,
            "employmentType": 0,
            "email_type": 0,

            "prc_addrL1": 0,
            "prc_addrL2": 0,
            "prc_addrL3": 0,
            "contactAddr": [0] * 22,
            "resideAddress": [0] * 22
        }
        numFea = []  # 2
        numName = []
        idFea = []  # 8
        idName = []

        # --- numerical type : total 2---
        """ "zxAge", "zxAgeRound" -> 2
        """
        numFea.extend([xIdentity.get("zxAge", 0), xIdentity.get("zxAgeRound", 0)])
        numName.extend(["zxAge", "zxAgeRound"])

        # --- id type : total 6 ---
        """ "education", "degree", "gender", "maritalStatus", "employmentType", "email_type"
        -> 6+2 
        id 类 异常在解析代码中已经处理。
        """

        idFea.append(self.edu2ids[xIdentity.get("education")])
        idName.append("education")
        idFea.append(self.degree2ids[xIdentity.get("degree")])
        idName.append("degree")
        idFea.append(self.gender2ids[xIdentity.get("gender")])
        idName.append("gender")
        idFea.append(self.maris2ids[xIdentity.get("maritalStatus")])
        idName.append("maritalStatus")
        idFea.append(self.empTp2ids[xIdentity.get("employmentType")])
        idName.append("employmentType")

        idFea.append(self.emlTp2ids[xIdentity.get("email_type")])
        idName.append("email_type")

        """"prcid, contactAddress", "residenceAddress" -> 3 + 22 + 22
        """
        xIdentity_contactAddress = xIdentity.get("contactAddress")
        xIdentity_residenceAddress = xIdentity.get("residenceAddress")
        if xIdentity_contactAddress == "" and xIdentity_residenceAddress == "":
            xIdentity_contactAddress = "None"
            xIdentity_residenceAddress = "None"

        # prcid
        # dtct_dict, city_dict, prov_dict = std3Addr2ids()
        prcid_provId = self.prov_dict.get(prcid[:2], 0)
        prcid_cityId = self.city_dict.get(prcid[:4], 0)
        prcid_dtctId = self.dtct_dict.get(prcid[:6], 0)

        contactAddr_ids, residenceAddr_ids = self.stdAddr_words([xIdentity_contactAddress, xIdentity_residenceAddress])

        idFea.append(prcid_provId)
        idName.append("prcid_provId")
        idFea.append(prcid_cityId)
        idName.append("prcid_cityId")
        idFea.append(prcid_dtctId)
        idName.append("prcid_dtctId")

        idFea.extend(contactAddr_ids)
        idName.append("contactAddr_ids_3+19=22")
        idFea.extend(residenceAddr_ids)
        idName.append("residenceAddr_ids_3+19=22")

        # return numFea, numName, idFea, idName

        # *-*-*-*-*-* proMidData Reside d = 5, 5X 22, 5 = 120 *-*-*-*-*-*
        """

        """

        resi_updt = []
        resi_addr = []
        resi_stat = []
        for resi in xReside:
            resi_updt.append(resi.get("infoUpdateDays"))
            resi_addr.append(resi.get("resideAddr"))
            resi_stat.append(self.resiStat2id[resi.get("resideStatus")])

        if len(xReside) < 5:
            resi_updt.extend([-9999] * (5 - len(xReside)))
            resi_stat.extend([0] * (5 - len(xReside)))
            resi_addr.extend(["None"] * (5 - len(xReside)))  # 全 "" 空字符串 list, cpca 解析会报错

        else:
            resi_updt = resi_updt[:5].copy()
            resi_stat = resi_stat[:5].copy()
            resi_addr = resi_addr[:5].copy()

        resi_addr_ids = self.stdAddr_words(resi_addr)  # [5, 22]

        # return resi_updt, resi_addr_ids, resi_stat

        # *-*-*-*-*-* proMidData Occupation: d = 5, 5,5,5,5, 5X22, 5X22 = 245 *-*-*-*-*-*
        """

        """
        Occup_infoUpdateDays = []

        Occup_companyType = []
        Occup_occupation = []
        Occup_position = []
        Occup_postTitle = []

        addrText = []
        nameText = []
        Occup_companyAddress = []
        Occup_companyName = []

        for occup in xOccup:
            Occup_infoUpdateDays.append(occup.get("infoUpdateDays"))

            Occup_companyType.append(self.compTp2id[occup.get("companyType")])
            #
            # print("Occup_companyType->", self.compTp2id[occup.get("companyType")])

            Occup_occupation.append(self.occup2id[occup.get("occupation")])
            Occup_position.append(self.position2id[occup.get("position")])
            Occup_postTitle.append(self.posTt2id[occup.get("postTitle")])

            addrText.append(occup.get("companyAddress"))
            nameText.append(occup.get("companyName"))

        if len(xOccup) < 5:
            Occup_infoUpdateDays.extend([-9999] * (5 - len(xOccup)))

            Occup_companyType.extend([0] * (5 - len(xOccup)))
            Occup_occupation.extend([0] * (5 - len(xOccup)))
            Occup_position.extend([0] * (5 - len(xOccup)))
            Occup_postTitle.extend([0] * (5 - len(xOccup)))

            addrText.extend(["None"] * (5 - len(xOccup)))
            nameText.extend(["None"] * (5 - len(xOccup)))
        else:
            Occup_infoUpdateDays = Occup_infoUpdateDays[:5].copy()
            Occup_companyType = Occup_companyType[:5].copy()
            Occup_occupation = Occup_occupation[:5].copy()
            Occup_position = Occup_position[:5].copy()
            Occup_postTitle = Occup_postTitle[:5].copy()

            addrText = addrText[:5].copy()
            nameText = nameText[:5].copy()

        Occup_companyAddress = self.stdAddr_words(addrText)  # [5, 22]
        Occup_companyName = self.stdAddr_words(nameText)  # [5, 22]

        # return Occup_infoUpdateDays, Occup_companyType, Occup_occupation, Occup_position, Occup_postTitle, Occup_companyAddress, Occup_companyName

        # *-*-*-*-*-* proMidData HfrList: d = 5,5,5,5, 5,5, 5X 22 = 140 *-*-*-*-*-*
        Hfr_infoUpdateDays = []
        Hfr_depositAmount = []
        Hfr_personalDepositRatio = []
        Hfr_unitDepositRatio = []

        Hfr_ginsengLand = []
        Hfr_gjjStatus = []

        payUnit_text = []
        Hfr_paymentUnit = []

        for hrf in xHfr:
            Hfr_infoUpdateDays.append(hrf.get("infoUpdateDays"))
            if hrf.get("depositAmount").replace(",", "").isdigit():
                Hfr_depositAmount.append(int(hrf.get("depositAmount").replace(",", "")))
            else:
                Hfr_depositAmount.append(0)
            if hrf.get("personalDepositRatio")[-1] == "%" and hrf.get("personalDepositRatio")[:-1].isdigit():
                Hfr_personalDepositRatio.append(int(hrf.get("personalDepositRatio")[:-1]) * 0.01)
            else:
                Hfr_personalDepositRatio.append(0)
            if hrf.get("unitDepositRatio")[-1] == "%" and hrf.get("unitDepositRatio")[:-1].isdigit():
                Hfr_unitDepositRatio.append(int(hrf.get("unitDepositRatio")[:-1]) * 0.01)
            else:
                Hfr_unitDepositRatio.append(0)

            # 一代报告 ginsengLand 为文本, 二代为行政区域编码
            if hrf.get("ginsengLand").isdigit():
                Hfr_ginsengLand.append(self.dtct_dict.get(hrf.get("ginsengLand"), 0))
            else:
                text = hrf.get("ginsengLand")
                gL_df = cpca.transform([text], open_warning=False)
                gL_df['省'].replace("北京市", "", inplace=True)
                gL_df['省'].replace("上海市", "", inplace=True)
                gL_df['省'].replace("天津市", "", inplace=True)
                gL_df['省'].replace("重庆市", "", inplace=True)

                gL = gL_df["省"][0] + gL_df["市"][0] + gL_df["区"][0]

                Hfr_ginsengLand.append(self.dtct_dict.get(gL, 0))

            Hfr_gjjStatus.append(self.gjjStat2id[hrf.get("gjjStatus")])

            payUnit_text.append(hrf.get("paymentUnit"))

        if len(xHfr) < 5:
            Hfr_infoUpdateDays.extend([-9999] * (5 - len(xHfr)))
            Hfr_depositAmount.extend([0] * (5 - len(xHfr)))
            Hfr_personalDepositRatio.extend([0] * (5 - len(xHfr)))
            Hfr_unitDepositRatio.extend([0] * (5 - len(xHfr)))

            Hfr_ginsengLand.extend([0] * (5 - len(xHfr)))
            Hfr_gjjStatus.extend([0] * (5 - len(xHfr)))

            payUnit_text.extend(["None"] * (5 - len(xHfr)))
        else:
            Hfr_infoUpdateDays = Hfr_infoUpdateDays[:5].copy()
            Hfr_depositAmount = Hfr_depositAmount[:5].copy()
            Hfr_personalDepositRatio = Hfr_personalDepositRatio[:5].copy()
            Hfr_unitDepositRatio = Hfr_unitDepositRatio[:5].copy()
            Hfr_ginsengLand = Hfr_ginsengLand[:5].copy()
            Hfr_gjjStatus = Hfr_gjjStatus[:5].copy()
            payUnit_text = payUnit_text[:5].copy()

        Hfr_paymentUnit = self.stdAddr_words(payUnit_text)

        # return Hfr_infoUpdateDays, Hfr_depositAmount, Hfr_personalDepositRatio, Hfr_unitDepositRatio, \
        #        Hfr_ginsengLand, Hfr_gjjStatus, \
        #        Hfr_paymentUnit
        flat_resi_addr_ids = []
        for _ in resi_addr_ids:
            flat_resi_addr_ids.extend(_)
        # print(len(flat_resi_addr_ids))
        flat_Occup_companyAddress = []
        for _ in Occup_companyAddress:
            flat_Occup_companyAddress.extend(_)
        # print(len(flat_Occup_companyAddress))
        flat_Occup_companyName = []
        for _ in Occup_companyName:
            flat_Occup_companyName.extend(_)
        # print(len(flat_Occup_companyName))
        flat_Hfr_paymentUnit = []
        for _ in Hfr_paymentUnit:
            flat_Hfr_paymentUnit.extend(_)
        # print(len(flat_Hfr_paymentUnit))

        res_fea = numFea + idFea + \
                  resi_updt + resi_stat + flat_resi_addr_ids + \
                  Occup_infoUpdateDays + Occup_companyType + Occup_occupation + Occup_position + Occup_postTitle + flat_Occup_companyAddress + flat_Occup_companyName + \
                  Hfr_infoUpdateDays + Hfr_depositAmount + Hfr_personalDepositRatio + Hfr_unitDepositRatio + Hfr_ginsengLand + Hfr_gjjStatus + flat_Hfr_paymentUnit

        # 2+6+3+22*2 + 5+5+5X22 + 5+5+5+5+5+5X22+5X22 + 5+5+5+5+5+5+5X22 = 560
        return res_fea

    def genQueryFea(self, x):
        qList = []
        tList = []
        orgList = []
        for q in x:
            temp = []
            t = q.get("queryDeltDays")
            tList.append(t)

            orgList.append(q.get("queryOperator"))

            temp.append(t)
            temp.append(self.queryReason2id[q.get("queryReason")])
            temp.append(self.queryOrgType2id[q.get("queryOrgType")])
            qList.append(temp.copy())

        # qList.sort(key=lambda x_: x_[4], reverse=True)
        # qList, tList = np.array(qList)[:, 4:].tolist(), np.array(qList)[:, :4].tolist()

        # 无需 padding
        # if origlen < self.queryList_keep_len:
        #     padLen = self.queryList_keep_len - origlen
        #     for _ in range(padLen):
        #         qList.append([-9999, 0, 0])
        #         tList.append((-61, -61, 0, 0))
        #         orgList.append("none")
        # else:
        #     qList = qList[:self.queryList_keep_len]
        #     tList = tList[:self.queryList_keep_len]
        #     orgList = orgList[:self.queryList_keep_len]

        # query 为空/<10 时填充一个, 为了避免 没有 tGroup 节点，query 较少时 填充一个
        if len(qList) < 16:
            qList.append([-9999, 0, 0])
            tList.append(-9999)
            orgList.append("none")

        if len(qList) > 128:
            qList = qList[:128]
            tList = tList[:128]
            orgList = orgList[:128]

        origlen = len(qList)

        return qList, origlen, tList, orgList

    def genLoanFea(self, x, reportDate):
        """
        '_pred' means this field can be uesd in pretraining prediction
        :param x:
        :param reportDate:
        :param cofig:
        :return:

        todo: 还款边 中 如果没有 * 状态，则通过 'startDate' 字段 生成一个loan/credit 发生边 -> done
        """
        loanFea_list = []
        tList = []
        orgList = []
        repays_list = []
        return_name2idx = {
            "startDays": 0,
            "endDays": 1,
            "infoUpdateDays": 2,
            "repayDays": 3,
            "overdueDataList$overdueMonth": 18,

            "amount": 4,
            "balance": 5,
            "currentOverdueNum": 6,
            "currentOverdueAmount": 7,
            "overdue31To60Days": 8,
            "overdue61To90Days": 9,
            "overdue91To180Days": 10,
            "overdue180Days": 11,
            "remainRepayNum": 12,
            "totalNum": 13,
            "thisMonthRepayAmount": 14,
            "thisMonthActualRepayAmount": 15,

            "overdueDataList$overdueAmount": 16,
            "overdueDataList$overdueContMonths": 17,

            "accStatus": 19,
            "fiveclasscode": 20,
            "loanCurrency": 21,
            "loanPledge": 22,
            "loanRepayMode": 23,
            "loanOrgType_pred": 24,
            "transactionType": 25,
            "personAccountType": 26
        }

        for loan in x:

            # for repay graph
            repays = []  # [month_idx, repayStat, overdueAmt(以负值表示）, speAmt]
            year5RepayStatus = loan.get("year5RepayStatus")
            specTraDataList = loan.get("specTraDataList")
            speDict = dict()  # { str: int}

            while specTraDataList:
                spd = specTraDataList.pop()
                speDict[spd["specTraDate"][:6]] = int(spd["specTraAmount"])  # { "201906": 4521, ...}

            start_flag = 0
            # --- V2
            if year5RepayStatus:
                for rsDict in year5RepayStatus:
                    month = rsDict["month"]
                    mon_idx = self.getMonthIndex(reportDate, month)
                    rpStat = rsDict["repayStatus"].strip()
                    if rpStat == "*":
                        start_flag = 1
                    odAmt = -int(rsDict["repayAmount"])
                    spRepay = 0
                    if month in speDict:
                        spRepay = speDict[month]

                    repays.append([mon_idx, rpStat, odAmt, spRepay])  # -, str, -, +
            # --- V1
            else:
                month24RepayIdx = loan.get("month24RepayIdx")
                month24RepayStatus = loan.get("month24RepayStatus")
                overdueDataList = loan.get("overdueDataList")

                if month24RepayIdx != "None" and isinstance(month24RepayIdx, list) and isinstance(month24RepayStatus,
                                                                                                  str):
                    month24RepayStatus = month24RepayStatus.strip().split()
                else:
                    month24RepayIdx = []
                    month24RepayStatus = []

                if "*" in month24RepayStatus:
                    start_flag = 1

                if len(month24RepayIdx) != len(month24RepayStatus):
                    # print(month24RepayIdx)
                    # print(month24RepayStatus)
                    print("还款序列长度 与 索引 不等..")
                else:
                    # 还款状态
                    for i_ in range(len(month24RepayIdx)):
                        repays.append([month24RepayIdx[i_], month24RepayStatus[i_], 0, 0])

                    # 逾期
                    for od in overdueDataList:
                        overdueMonth = od["overdueMonth"]
                        overdueContMonths = od["overdueContMonths"]
                        overdueAmount = od["overdueAmount"]

                        if overdueMonth == "None":
                            continue

                        m_idx = self.getMonthIndex(reportDate, overdueMonth)
                        flag = 0
                        for i in range(len(repays)):
                            if repays[i][0] == m_idx:
                                repays[i][2] = -int(overdueAmount)
                                flag = 1
                        if flag == 0:
                            if overdueContMonths > "99": overdueContMonths = "99"
                            repays.append([m_idx, overdueContMonths, -int(overdueAmount), 0])
                    # 提前还款
                    for sp_m in speDict.keys():
                        sp_midx = self.getMonthIndex(reportDate, sp_m)
                        flag = 0
                        for i in range(len(repays)):
                            if repays[i][0] == sp_midx:
                                repays[i][3] = speDict[sp_m]
                                flag = 1

                        if flag == 0:
                            repays.append([sp_midx, "C", 0, speDict[sp_m]])
            # print("repays in loan:", repays)

            if start_flag == 0:
                start_date = loan.get("startDate")
                if start_date.isdigit():
                    start_date = start_date[:6]
                else:
                    start_date = "199901"
                m_idx = self.getMonthIndex(reportDate, start_date)
                repays.append([m_idx, "*", 0, 0])

            repays_list.append(copy.deepcopy(repays))

            loanFea = []
            t = loan.get("startDays")
            tList.append(t)
            # if t < -1800:
            #     t = -1800
            # if t > 0:
            #     t = 0
            # down = t // 30
            # up = down+1
            # down_w = (up*30 - t) / 30
            # up_w = (t-down*30) / 30

            # print("t->", t)
            # print("down, up->", down, up)

            # tList.append((down, up, down_w, up_w))
            # loanFea.extend((down, up, down_w, up_w))

            orgList.append(loan.get("orgName"))

            loanFea.append(t)
            loanFea.append(loan.get("endDays"))
            loanFea.append(loan.get("infoUpdateDays"))
            loanFea.append(loan.get("repayDays"))

            loanFea.append(int(loan.get("amount")) if loan.get("amount").isdigit() else 0)
            loanFea.append(int(loan.get("balance")) if loan.get("balance").isdigit() else 0)
            loanFea.append(int(loan.get("currentOverdueNum")) if loan.get("currentOverdueNum").isdigit() else 0)
            loanFea.append(int(loan.get("currentOverdueAmount")) if loan.get("currentOverdueAmount").isdigit() else 0)
            loanFea.append(int(loan.get("overdue31To60Days")) if loan.get("overdue31To60Days").isdigit() else 0)
            loanFea.append(int(loan.get("overdue61To90Days")) if loan.get("overdue61To90Days").isdigit() else 0)
            loanFea.append(int(loan.get("overdue91To180Days")) if loan.get("overdue91To180Days").isdigit() else 0)
            loanFea.append(int(loan.get("overdue180Days")) if loan.get("overdue180Days").isdigit() else 0)
            loanFea.append(int(loan.get("remainRepayNum")) if loan.get("remainRepayNum").isdigit() else 0)
            loanFea.append(int(loan.get("totalNum")) if loan.get("totalNum").isdigit() else 0)
            loanFea.append(int(loan.get("thisMonthRepayAmount")) if loan.get("thisMonthRepayAmount").isdigit() else 0)
            loanFea.append(
                int(loan.get("thisMonthActualRepayAmount")) if loan.get("thisMonthActualRepayAmount").isdigit() else 0)

            def merge_overdueDataList(json_list):
                maxAmount = 0
                maxCountMonths = 0
                lastMonth = "0"
                if json_list:
                    for item in json_list:
                        amount = int(item.get("overdueAmount", "0").replace(",", "")
                                     if item.get("overdueAmount", "0").replace(",", "").isdigit() else 0
                                     )
                        months = int(item.get("overdueContMonths", "0").replace(",", "")
                                     if item.get("overdueContMonths", "0").replace(",", "").isdigit() else 0
                                     )
                        mon = item.get("overdueMonth", "0").replace(".", "")
                        if amount > maxAmount:
                            maxAmount = amount
                        if months > maxCountMonths:
                            maxCountMonths = months

                        if mon.isdigit() and len(mon) == 6 and mon > lastMonth:
                            lastMonth = mon
                recentMonths = -9999
                if len(lastMonth) == 6:
                    eMon = datetime.datetime(int(lastMonth[:4]), int(lastMonth[4:6]), 1)
                    sMon = datetime.datetime(int(reportDate[:4]), int(reportDate[4:6]), int(reportDate[6:8]))
                    recentMonths = (eMon - sMon).days // 30

                return (maxAmount, maxCountMonths, recentMonths)

            overdueDataList = loan.get("overdueDataList", "None")
            if overdueDataList in ("None", ""):
                loanFea.extend([0, 0, -9999])
            elif isinstance(overdueDataList, list):
                loanFea.extend(merge_overdueDataList(overdueDataList))
            else:
                print("type of overdueDataList-> ", type(overdueDataList))
                loanFea.extend([0, 0, -9999])

            # id type
            loanFea.append(self.loanAccStatus2id[loan.get("loanAccStatus")])
            loanFea.append(self.fiveclasscode2id[loan.get("fiveclasscode")])
            loanFea.append(self.loanCurrency2id[loan.get("loanCurrency")])
            loanFea.append(self.loanPledge2id[loan.get("loanPledge")])
            loanFea.append(self.loanRepayMode2id[loan.get("loanRepayMode")])
            loanFea.append(self.loanOrgType2id[loan.get("loanOrgType")])
            loanFea.append(self.transactionType2id[loan.get("transactionType")])
            loanFea.append(self.loanPersonAccountType2id[loan.get("personAccountType")])

            loanFea_list.append(loanFea.copy())

        # loanFea_list.sort(key=lambda x_: x_[4], reverse=True)
        # loanFea_list, tList = np.array(loanFea_list)[:, 4:].tolist(), np.array(loanFea_list)[:, :4].tolist()

        # 无需 padding
        # if len(loanFea_list) < self.loanList_keep_len:
        #     padLen = self.loanList_keep_len - len(loanFea_list)
        #     for _ in range(padLen):
        #         loanFea_list.append([-9999, -9999, -9999, -9999,
        #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                              0, 0, -9999,
        #                              0, 0, 0, 0, 0, 0, 0
        #                              ])
        #         tList.append((-61, -61, 0, 0))
        #         orgList.append("none")
        # else:
        #     loanFea_list = loanFea_list[:self.loanList_keep_len]
        #     tList = tList[:self.loanList_keep_len]

        # 全空时，填充一个
        if len(loanFea_list) < 32:
            loanFea_list.append([-9999, -9999, -9999, -9999,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, -9999,
                                 0, 0, 0, 0, 0, 0, 0, 0
                                 ])
            tList.append(-9999)
            orgList.append("none")

        if len(loanFea_list) > 256:
            loanFea_list = loanFea_list[:256]
            tList = tList[:256]
            orgList = orgList[:256]
            repays_list = repays_list[:256]

        # 按 特征类型 排列， days:0-4, num 5-18, type: 19-26
        if loanFea_list:
            loanFea_list_temp = np.concatenate([
                np.array(loanFea_list)[:, :4], np.array(loanFea_list)[:, 18:19],
                np.array(loanFea_list)[:, 4:18],
                np.array(loanFea_list)[:, 19:]
            ], axis=1)
            loanFea_list = loanFea_list_temp.tolist()
        else:
            pass

        origlen = len(loanFea_list)

        return loanFea_list, origlen, tList, orgList, repays_list

    def genCreditFea(self, x, reportDate):
        """

        :param x:
        :param reportDate:
        :return:

        todo: 还款边 中 如果没有 * 状态，则通过 'startDate' 字段 生成一个loan/credit 发生边 -> done
        """
        creditFea_list = []
        tList = []
        orgList = []
        repays_list = []
        return_name2id = {
            "startDays": 0,
            "infoUpdateDays": 1,
            "lastRepayDays": 2,
            "billingDays": 3,
            "overdueDataList$overdueMonth": 17,

            "creditAmount": 4,
            "shareAmount": 5,
            "usedMmount": 6,
            "thisMonthRepayAmount": 7,
            "thisMonthActualRepayAmount": 8,
            "averageUsage": 9,
            "maxUsedCreditLimit": 10,
            "currentOverdueNum": 11,
            "currentOverdueAmount": 12,
            "principalAmount": 13,
            "balance": 14,

            "overdueDataList$overdueAmount": 15,
            "overdueDataList$overdueContMonths": 16,

            "accStatus": 18,
            "creditPledge": 19,
            "creditOrgType": 20,
            "creditCurrency": 21,
            "repaymentStatus": 22,
            "personAccountType": 23
        }

        for cred in x:

            # for repay graph
            # for repay graph
            repays = []  # [month_idx, repayStat, overdueAmt(以负值表示）, speAmt]
            year5RepayStatus = cred.get("year5RepayStatus")
            # specTraDataList = cred.get("specTraDataList")
            # speDict = dict()  # { str: int}

            # while specTraDataList:
            #     spd = specTraDataList.pop()
            #     speDict[spd["specTraDate"][:6]] = int(spd["specTraAmount"])  # { "201906": 4521, ...}

            start_flag = 0
            # --- V2
            if year5RepayStatus:
                for rsDict in year5RepayStatus:
                    month = rsDict["month"]
                    mon_idx = self.getMonthIndex(reportDate, month)
                    rpStat = rsDict["repayStatus"].strip()
                    if rpStat == "*":
                        start_flag = 1
                    odAmt = -int(rsDict["repayAmount"])
                    spRepay = 0
                    # if month in speDict:
                    #     spRepay = speDict[month]

                    repays.append([mon_idx, rpStat, odAmt, spRepay])  # -, str, -, +
            # --- V1
            else:
                month24RepayIdx = cred.get("month24RepayIdx")
                month24RepayStatus = cred.get("month24RepayStatus")
                overdueDataList = cred.get("overdueDataList")

                if month24RepayIdx != "None" and isinstance(month24RepayIdx, list) and isinstance(month24RepayStatus,
                                                                                                  str):
                    month24RepayStatus = month24RepayStatus.strip().split()
                else:
                    month24RepayIdx = []
                    month24RepayStatus = []

                if "*" in month24RepayStatus:
                    start_flag = 1

                if len(month24RepayIdx) != len(month24RepayStatus):
                    print("还款序列长度 与 索引 不等..")
                else:
                    # 还款状态
                    for i_ in range(len(month24RepayIdx)):
                        repays.append([month24RepayIdx[i_], month24RepayStatus[i_], 0, 0])

                    # 逾期
                    for od in overdueDataList:
                        overdueMonth = od["overdueMonth"]
                        overdueContMonths = od["overdueContMonths"]
                        overdueAmount = od["overdueAmount"]

                        if overdueMonth == "None":
                            continue

                        m_idx = self.getMonthIndex(reportDate, overdueMonth)
                        flag = 0
                        for i in range(len(repays)):
                            if repays[i][0] == m_idx:
                                repays[i][2] = -int(overdueAmount)
                                flag = 1
                        if flag == 0:
                            if overdueContMonths > "99": overdueContMonths = "99"
                            repays.append([m_idx, overdueContMonths, -int(overdueAmount), 0])
                    # 提前还款
                    # for sp_m in speDict.keys():
                    #     sp_midx = self.getMonthIndex(reportDate, sp_m)
                    #     flag = 0
                    #     for i in range(len(repays)):
                    #         if repays[i][0] == sp_midx:
                    #             repays[i][3] = speDict[sp_m]
                    #             flag = 1
                    #
                    #     if flag == 0:
                    #         repays.append([sp_midx, "C", 0, speDict[sp_m]])
            # print("repays in credit", repays)
            if start_flag == 0:
                start_date = cred.get("startDate")
                if start_date.isdigit():
                    start_date = start_date[:6]
                else:
                    start_date = "199901"
                m_idx = self.getMonthIndex(reportDate, start_date)
                repays.append([m_idx, "*", 0, 0])
            repays_list.append(copy.deepcopy(repays))

            credFea = []

            t = cred.get("startDays")
            tList.append(t)
            # if t < -1800:
            #     t = -1800
            # if t > 0:
            #     t = 0
            # down = t // 30
            # up = down+1
            # down_w = (up*30 - t) / 30
            # up_w = (t-down*30) / 30
            #
            # tList.append((down, up, down_w, up_w))
            # credFea.extend((down, up, down_w, up_w))
            orgList.append(cred.get("orgName"))

            credFea.append(t)
            credFea.append(cred.get("infoUpdateDays"))
            credFea.append(cred.get("lastRepayDays"))
            credFea.append(cred.get("billingDays"))

            credFea.append(int(cred.get("creditAmount")) if cred.get("creditAmount").isdigit() else 0)

            # credFea.append(int(cred.get("shareAmount")) if cred.get("shareAmount").isdigit() else 0)
            # -- 2021.11.29 -- checking ks diff
            if cred.get("shareAmount").isdigit():
                credFea.append(int(cred.get("shareAmount")))
            else:
                credFea.append(int(cred.get("creditAmount")) if cred.get("creditAmount").isdigit() else 0)

            # -- end check --

            credFea.append(int(cred.get("usedMmount")) if cred.get("usedMmount").isdigit() else 0)
            credFea.append(int(cred.get("thisMonthRepayAmount")) if cred.get("thisMonthRepayAmount").isdigit() else 0)
            credFea.append(
                int(cred.get("thisMonthActualRepayAmount")) if cred.get("thisMonthActualRepayAmount").isdigit() else 0)
            credFea.append(int(cred.get("averageUsage")) if cred.get("averageUsage").isdigit() else 0)
            credFea.append(int(cred.get("maxUsedCreditLimit")) if cred.get("maxUsedCreditLimit").isdigit() else 0)
            credFea.append(int(cred.get("currentOverdueNum")) if cred.get("currentOverdueNum").isdigit() else 0)
            credFea.append(int(cred.get("currentOverdueAmount")) if cred.get("currentOverdueAmount").isdigit() else 0)
            credFea.append(int(cred.get("principalAmount")) if cred.get("principalAmount").isdigit() else 0)
            credFea.append(int(cred.get("balance")) if cred.get("balance").isdigit() else 0)

            def merge_overdueDataList(json_list):
                maxAmount = 0
                maxCountMonths = 0
                lastMonth = "0"
                if json_list:
                    for item in json_list:
                        amount = int(item.get("overdueAmount", 0))
                        months = int(item.get("overdueContMonths", 0))
                        mon = item.get("overdueMonth", "0").replace(".", "")
                        if amount > maxAmount:
                            maxAmount = amount
                        if months > maxCountMonths:
                            maxCountMonths = months

                        if mon.isdigit() and len(mon) == 6 and mon > lastMonth:
                            lastMonth = mon
                recentMonths = -9999
                if len(lastMonth) == 6:
                    eMon = datetime.datetime(int(lastMonth[:4]), int(lastMonth[4:6]), 1)
                    sMon = datetime.datetime(int(reportDate[:4]), int(reportDate[4:6]), int(reportDate[6:8]))
                    recentMonths = (eMon - sMon).days // 30

                return (maxAmount, maxCountMonths, recentMonths)

            overdueDataList = cred.get("overdueDataList", "None")
            if overdueDataList in ("None", ""):
                credFea.extend([0, 0, -9999])
            elif isinstance(overdueDataList, list):
                credFea.extend(merge_overdueDataList(overdueDataList))
            else:
                print("type of overdueDataList-> ", type(overdueDataList))
                credFea.extend([0, 0, -9999])

            # id type
            credFea.append(self.creditAccStatus2id[cred.get("creditAccStatus")])
            credFea.append(self.creditPledge2id[cred.get("creditPledge")])
            credFea.append(self.creditOrgType2id[cred.get("creditOrgType")])
            credFea.append(self.creditCurrency2id[cred.get("creditCurrency")])
            credFea.append(self.repaymentStatus2id[cred.get("repaymentStatus")])
            credFea.append(self.creditPersonAccountType2id[cred.get("personAccountType")])

            creditFea_list.append(credFea.copy())

        # creditFea_list.sort(key=lambda x_: x_[4], reverse=True)
        # creditFea_list, tList = np.array(creditFea_list)[:, 4:].tolist(), np.array(creditFea_list)[:, :4].tolist()

        # 无需 padding
        # if origLen < self.creditList_keep_len:
        #     padLen = self.creditList_keep_len - origLen
        #     for _ in range(padLen):
        #         creditFea_list.append([
        #             -9999, -9999, -9999, -9999,
        #             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #             0, 0, -9999,
        #             0, 0, 0, 0, 0
        #         ])
        #         tList.append((-61, -61, 0, 0))
        #         orgList.append("none")
        # else:
        #     creditFea_list = creditFea_list[:self.creditList_keep_len]
        #     tList = tList[:self.creditList_keep_len]
        #     orgList = orgList[:self.creditList_keep_len]

        # 按 特征类型 排列， days:0-4, num 5-17, type: 19-25

        # 贷记卡记录为空时填充一个
        if len(creditFea_list) < 16:
            creditFea_list.append([
                -9999, -9999, -9999, -9999,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, -9999,
                0, 0, 0, 0, 0, 0
            ])
            tList.append(-9999)
            orgList.append("none")

        if len(creditFea_list) > 128:
            creditFea_list = creditFea_list[:128]
            tList = tList[:128]
            orgList = orgList[:128]
            repays_list = repays_list[:128]

        if creditFea_list:
            creditFea_list_temp = np.concatenate([
                np.array(creditFea_list)[:, :4], np.array(creditFea_list)[:, 17:18],
                np.array(creditFea_list)[:, 4:17],
                np.array(creditFea_list)[:, 18:]
            ], axis=1)

            creditFea_list = creditFea_list_temp.tolist()
        else:
            pass

        origLen = len(creditFea_list)

        return creditFea_list, origLen, tList, orgList, repays_list

    def genGraphFea_v3(self, query_x, loan_x, credit_x, reportDate):
        """
        异构图中，共 4 类节点：
            长序列（loan, credit, query): 编号 0 ～ l_origlen + c_origLen + q_origlen
            还款时间节点：编号 [0～60]，代表过去第 n 个月发生
            机构：不定数目， 0～x
            时间连通聚类：不定数目， 0～x

        update: 1、增加了 还款记录 逾期的状态类别
                todo：2、贷记卡还款 去掉 *
        """
        l_list, l_origlen, l_tList, l_orgList, l_repays_list = self.genLoanFea(loan_x, reportDate)
        c_list, c_origLen, c_tList, c_orgList, c_repays_list = self.genCreditFea(credit_x, reportDate)
        q_list, q_origlen, q_tList, q_orgList = self.genQueryFea(query_x)

        lcq_node_num = l_origlen + c_origLen + q_origlen

        lcq_list = l_list + c_list + q_list
        lcq_org_list = l_orgList + c_orgList + q_orgList
        lcq_tlist = l_tList + c_tList + q_tList

        #  --- edge loan/credit repays -> month ---
        """
        return: lc2m_src, lc2m_dst, lc2m_edge
        """
        repay_node_num = 61  # 0 ~ 60
        lc2m_src = []  # 还款边/起点：loan, credit
        lc2m_dst = []  # 终点：近60月
        """
        边特征：还款状态，逾期金额，提前还款金额 （ v1: month24RepayStatus(lc)，overdueDataList(lc)，specTraDataList（l)
        v2: year5RepayStatus(lc), year5RepayStatus(lc), specTraDataList(l)

        """
        lc2m_edge = []

        repaySt2edgeIdx = {
            "*": 0,
            "N": 1,
            "C": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
        }
        for i in range(5, 100):
            repaySt2edgeIdx[str(i)] = 7

        # --- 二代 还款 特征解析 ---
        for i in range(len(l_repays_list)):
            for m_idx, r_st, odAmt, spAmt in l_repays_list[i]:
                if r_st in ["*", "N", "C"] + [str(_) for _ in range(1, 100)]:
                    lc2m_src.append(i)

                    if -60 <= m_idx <= 0:
                        lc2m_dst.append(-m_idx)
                    elif m_idx < -60:
                        lc2m_dst.append(60)
                    else:
                        lc2m_dst.append(0)

                    lc2m_edge.append([repaySt2edgeIdx[r_st], odAmt, spAmt])

        for i in range(len(c_repays_list)):
            for m_idx, r_st, odAmt, spAmt in c_repays_list[i]:
                if r_st in ["*", "N", "C"] + [str(_) for _ in range(1, 100)]:
                    lc2m_src.append(i + l_origlen)

                    if -60 <= m_idx <= 0:
                        lc2m_dst.append(-m_idx)
                    elif m_idx < -60:
                        lc2m_dst.append(60)
                    else:
                        lc2m_dst.append(0)

                    lc2m_edge.append([repaySt2edgeIdx[r_st], odAmt, spAmt])

        # return lc2m_src, lc2m_dst, lc2m_edge

        # 时间连通域 图, 得到被时间 聚类的节点 clusters
        t2idx = []
        t_agg_num = 0
        for i in range(len(lcq_tlist)):
            t2idx.append([lcq_tlist[i], i])
        t2idx.sort(key=lambda x_: x_[0])

        clusters = []
        group = set()
        for i in range(1, len(t2idx)):
            t_, idx_ = t2idx[i - 1]
            t_n, idx_n = t2idx[i]
            if (t_n - t_) <= 31:
                group.add(idx_)
                group.add(idx_n)
                continue
            elif len(group) > 1:
                clusters.append(copy.deepcopy(group))
                group.clear()
                continue
            else:
                group.clear()

        if len(group) > 1:
            clusters.append(copy.deepcopy(group))

        tg_id = 0
        tg_src = []
        tg_dst = []
        while clusters:
            nodes = clusters.pop()
            for n_id in nodes:
                tg_src.append(n_id)
                tg_dst.append(tg_id)
            tg_id += 1
        tg_node_num = tg_id

        # 贷款、贷记卡、查询 机构图
        org_src = []
        org_dst = []
        d_org = defaultdict(list)
        for i in range(len(lcq_org_list)):
            d_org[lcq_org_list[i]].append(i)

        org_node_id = 0
        for k in d_org.keys():
            for lcd_id in d_org[k]:
                org_src.append(lcd_id)
                org_dst.append(org_node_id)

            org_node_id += 1
        org_node_num = org_node_id

        return repay_node_num, lcq_node_num, tg_node_num, org_node_num, \
               l_list, c_list, q_list, \
               lc2m_src, lc2m_dst, lc2m_edge, \
               tg_src, tg_dst, \
               org_src, org_dst

    def genSummryFea(self, x, reportDate):
        summryFea = []
        name2index = {
            'zx_ct_firstHousingLoanMonth': 'None',
            'zx_ct_firstCommercialLoanMonth': 'None',
            'zx_ct_firstOtherLoanMonth': 'None',
            'zx_ct_firstLoanMonth': 'None',
            'zx_ct_firstDebitMonth': 'None',
            'zx_ct_firstSemiCreditMonth': 'None',

            'zxLast1MonthsCreditCardApprovalSum': 0,
            'zxLast1MonthsLoanApprovalSum': 0,
            'zxLast1MonthsOrgCreditCardApprovalSum': 0,
            'zxLast1MonthsOrgLoanApprovalSum': 0,
            'zxLast2YearsGuarApproSum': 0,
            'zxLast2YearsLoanMangeSum': 0,
            'zxLast2YearsSpeMerchApproSum': 0,
            'zxRcy1MSelfQuery': 0,
            'zx_ct_housingLoanNum': 0,
            'zx_ct_commercialLoanNum': 0,
            'zx_ct_otherLoanNum': 0,
            'zx_ct_debitAccountNum': 0,
            'zx_ct_semiCreditNum': 0,
            'zx_ct_declareNum': 0,
            'zx_ct_objectionLabelNum': 0,
            'zx_ols_loanInstitutionsNum': 0,
            'zx_ols_num': 0,
            'zx_ols_contractAmount': 0,
            'zx_ols_balance': 0,
            'zx_ols_averageRepayment': 0,
            'zx_ncs_cardlLegalIssuerNum': 0,
            'zx_ncs_cardIssuerNum': 0,
            'zx_ncs_accountNum': 0,
            'zx_ncs_totalCredit': 0,
            'zx_ncs_maxCredit': 0,
            'zx_ncs_minCredit': 0,
            'zx_ncs_usedMmount': 0,
            'zx_ncs_averageUsage': 0,
            'zx_oods_overdueLoan_num': 0,
            'zx_oods_overdueLoan_monthNum': 0,
            'zx_oods_overdueLoan_overdueAmount': 0,
            'zx_oods_overdueLoan_longestOverdueMonth': 0,
            'zx_oods_overdueCredit_accountNum': 0,
            'zx_oods_overdueCredit_monthNum': 0,
            'zx_oods_overdueCredit_overdueAmount': 0,
            'zx_oods_overdueCredit_longestOverdueMonth': 0
        }

        def getDelteMonths(k):
            val = x.get(k)
            if val != "None" and val.isdigit() and len(val) == 6:
                eMon = datetime.datetime(int(val[:4]), int(val[4:6]), 1)
                sMon = datetime.datetime(int(reportDate[:4]), int(reportDate[4:6]), int(reportDate[6:8]))
                return (eMon - sMon).days // 30
            else:
                return -9999

        summryFea.append(getDelteMonths("zx_ct_firstHousingLoanMonth"))
        summryFea.append(getDelteMonths("zx_ct_firstCommercialLoanMonth"))
        summryFea.append(getDelteMonths("zx_ct_firstOtherLoanMonth"))
        summryFea.append(getDelteMonths("zx_ct_firstLoanMonth"))
        summryFea.append(getDelteMonths("zx_ct_firstDebitMonth"))
        summryFea.append(getDelteMonths("zx_ct_firstSemiCreditMonth"))

        summryFea.append(x.get("zxLast1MonthsCreditCardApprovalSum"))
        summryFea.append(x.get("zxLast1MonthsLoanApprovalSum"))
        summryFea.append(x.get("zxLast1MonthsOrgCreditCardApprovalSum"))
        summryFea.append(x.get("zxLast1MonthsOrgLoanApprovalSum"))
        summryFea.append(x.get("zxLast2YearsGuarApproSum"))
        summryFea.append(x.get("zxLast2YearsLoanMangeSum"))
        summryFea.append(x.get("zxLast2YearsSpeMerchApproSum"))
        summryFea.append(x.get("zxRcy1MSelfQuery"))
        summryFea.append(x.get("zx_ct_housingLoanNum"))
        summryFea.append(x.get("zx_ct_commercialLoanNum"))
        summryFea.append(x.get("zx_ct_otherLoanNum"))
        summryFea.append(x.get("zx_ct_debitAccountNum"))
        summryFea.append(x.get("zx_ct_semiCreditNum"))
        summryFea.append(x.get("zx_ct_declareNum"))
        summryFea.append(x.get("zx_ct_objectionLabelNum"))
        summryFea.append(x.get("zx_ols_loanInstitutionsNum"))
        summryFea.append(x.get("zx_ols_num"))
        summryFea.append(x.get("zx_ols_contractAmount"))
        summryFea.append(x.get("zx_ols_balance"))
        summryFea.append(x.get("zx_ols_averageRepayment"))
        summryFea.append(x.get("zx_ncs_cardlLegalIssuerNum"))
        summryFea.append(x.get("zx_ncs_cardIssuerNum"))
        summryFea.append(x.get("zx_ncs_accountNum"))
        summryFea.append(x.get("zx_ncs_totalCredit"))
        summryFea.append(x.get("zx_ncs_maxCredit"))
        summryFea.append(x.get("zx_ncs_minCredit"))
        summryFea.append(x.get("zx_ncs_usedMmount"))
        summryFea.append(x.get("zx_ncs_averageUsage"))
        summryFea.append(x.get("zx_oods_overdueLoan_num"))
        summryFea.append(x.get("zx_oods_overdueLoan_monthNum"))
        summryFea.append(x.get("zx_oods_overdueLoan_overdueAmount"))
        summryFea.append(x.get("zx_oods_overdueLoan_longestOverdueMonth"))
        summryFea.append(x.get("zx_oods_overdueCredit_accountNum"))
        summryFea.append(x.get("zx_oods_overdueCredit_monthNum"))
        summryFea.append(x.get("zx_oods_overdueCredit_overdueAmount"))
        summryFea.append(x.get("zx_oods_overdueCredit_longestOverdueMonth"))

        return summryFea