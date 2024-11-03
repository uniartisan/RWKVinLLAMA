# 首先定义所有变量
entity_types = "ORGANIZATION,DATE,NUMBER"
tuple_delimiter = "||"
completion_delimiter = "<完成>"
language = "中文"
input_text = """贷款市场报价利率（Loan Prime Rate, LPR）是由具有代表性的报价行，根据本行对最优质客户的贷款利率，以公开市场操作利率（主要指中期借贷便利利率）加点形成的方式报价，由中国人民银行授权全国银行间同业拆借中心计算并公布的基础性的贷款参考利率，各金融机构应主要参考LPR进行贷款定价。 现行的LPR包括1年期和5年期以上两个品种 [1]。LPR市场化程度较高，能够充分反映信贷市场资金供求情况，使用LPR进行贷款定价可以促进形成市场化的贷款利率，提高市场利率向信贷利率的传导效率。
2020年8月12日，中国工商银行、中国建设银行、中国农业银行、中国银行和中国邮政储蓄银行五家国有大行同时发布公告，于8月25日起对批量转换范围内的个人住房贷款，按照相关规则统一调整为LPR（贷款市场报价利率）定价方式。 [2]
最新贷款市场报价利率（LPR）：2024年10月21日，1年期LPR为3.10%，5年期以上LPR为3.60%，均较此前下降0.25个百分点。
"""

# 使用双大括号来转义格式字符串中的大括号
Entity_Relation_Extraction_Prompt = """
-目标-
提供一段相关文本和实体类型列表，从文本中识别指定类型的所有实体，并分析这些实体之间的关联关系。
-步骤-

1. 实体识别。对每个识别出的实体，需要提取以下信息：

entity_name：实体名称（使用原文形式，中文无需大写）
entity_type：从以下类型中选择：[{entity_types}]
entity_description：详细描述实体的特征和行为
格式要求：(T{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)


2. 关系识别。从第1步识别的实体中，找出所有存在明确关联的实体对(source_entity, target_entity)。
对每对关联实体，提取以下信息：

source_entity：起始实体名称（来自第1步的识别结果）
target_entity：目标实体名称（来自第1步的识别结果）
relationship_description：说明两个实体之间存在关联的具体原因
relationship_strength：用1-10的整数表示关系强度
格式要求：(R{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

-要求-

1. 使用{language}语言输出包含所有实体和关系的列表。
2. 如需翻译成{language}，仅翻译描述性内容，保持其他格式和标识符不变！
3. 先输出所有实体，再输出这些实体间的关系！
4. 所有的实体和关联实体都来自相关文本，不要输出文本中没有的实体和关系！
5. 确保关联实体的起始实体和目标实体都来自实体识别的结果！
6. 实体和关联实体不要重复！
7. 最后添加{completion_delimiter}表示完成。

######################
-示例-
######################
示例 1：
实体类型: ORGANIZATION,PERSON
文本：
维丹蒂斯中央机构定于周一和周四召开会议。该机构计划于周四下午1:30（太平洋时间）发布最新政策决定，随后将举行新闻发布会，由中央机构主席马丁·史密斯回答提问。投资者预计市场战略委员会将维持基准利率在3.5%-3.75%区间不变。
######################
输出：
(T{tuple_delimiter}维丹蒂斯中央机构{tuple_delimiter}ORGANIZATION{tuple_delimiter}维丹蒂斯的中央金融管理机构，负责制定利率政策)
(T{tuple_delimiter}马丁·史密斯{tuple_delimiter}PERSON{tuple_delimiter}中央机构现任主席，负责政策解读与沟通)
(T{tuple_delimiter}市场战略委员会{tuple_delimiter}ORGANIZATION{tuple_delimiter}负责制定货币政策和利率决策的核心委员会)
(R{tuple_delimiter}马丁·史密斯{tuple_delimiter}维丹蒂斯中央机构{tuple_delimiter}作为主席领导中央机构，并将主持新闻发布会{tuple_delimiter}9){completion_delimiter}
######################
示例 2：
实体类型: ORGANIZATION
文本：
科技环球(TG)股票在周四全球交易所首日交易中表现亮眼。不过IPO专家提醒，这家半导体企业的上市表现并不能代表其他新上市公司的市场前景。
科技环球原本是一家上市公司，2014年被远景控股私有化收购。这家知名芯片设计公司称其产品已覆盖85%的高端智能手机市场。
######################
输出：
(T{tuple_delimiter}科技环球{tuple_delimiter}ORGANIZATION{tuple_delimiter}全球交易所上市公司，在高端智能手机芯片市场占据主导地位)
(T{tuple_delimiter}远景控股{tuple_delimiter}ORGANIZATION{tuple_delimiter}曾收购科技环球的投资公司)
(R{tuple_delimiter}科技环球{tuple_delimiter}远景控股{tuple_delimiter}自2014年起成为其私有化后的母公司{tuple_delimiter}5){completion_delimiter}
######################
示例 3：
实体类型: ORGANIZATION,GEO,PERSON
文本：
五名在菲鲁扎巴德被囚禁8年、被普遍视为人质的奥雷利亚公民正在返回祖国途中。
这次由昆塔拉斡旋的交换在菲鲁扎巴德80亿美元资金转入昆塔拉首都克罗哈拉的金融机构后最终完成。
交换行动始于菲鲁扎巴德首都提鲁齐亚，四名男性和一名女性（同时具有菲鲁扎巴德国籍）已登上前往克罗哈拉的包机。
奥雷利亚高级官员在机场迎接了他们，目前他们正在前往奥雷利亚首都卡希昂的途中。
获释的奥雷利亚人包括：39岁的商人萨姆·纳马拉（此前被关押在提鲁齐亚的阿拉米亚监狱）、59岁的记者杜克·巴塔格拉尼，以及53岁同时拥有布拉提纳斯国籍的环保人士梅吉·塔兹巴。
######################
输出：
(T{tuple_delimiter}菲鲁扎巴德{tuple_delimiter}GEO{tuple_delimiter}扣押奥雷利亚人质的国家)
(T{tuple_delimiter}奥雷利亚{tuple_delimiter}GEO{tuple_delimiter}正在营救本国人质的国家)
(T{tuple_delimiter}昆塔拉{tuple_delimiter}GEO{tuple_delimiter}促成人质交换的调解国)
(T{tuple_delimiter}提鲁齐亚{tuple_delimiter}GEO{tuple_delimiter}菲鲁扎巴德首都，人质被关押地)
(T{tuple_delimiter}克罗哈拉{tuple_delimiter}GEO{tuple_delimiter}昆塔拉首都)
(T{tuple_delimiter}卡希昂{tuple_delimiter}GEO{tuple_delimiter}奥雷利亚首都)
(T{tuple_delimiter}萨姆·纳马拉{tuple_delimiter}PERSON{tuple_delimiter}曾被关押在阿拉米亚监狱的奥雷利亚商人)
(T{tuple_delimiter}阿拉米亚监狱{tuple_delimiter}GEO{tuple_delimiter}提鲁齐亚市的监狱)
(T{tuple_delimiter}杜克·巴塔格拉尼{tuple_delimiter}PERSON{tuple_delimiter}被扣押的奥雷利亚记者)
(T{tuple_delimiter}梅吉·塔兹巴{tuple_delimiter}PERSON{tuple_delimiter}被扣押的双重国籍环保人士)
(R{tuple_delimiter}菲鲁扎巴德{tuple_delimiter}奥雷利亚{tuple_delimiter}就人质问题进行谈判与交换{tuple_delimiter}2)
(R{tuple_delimiter}昆塔拉{tuple_delimiter}奥雷利亚{tuple_delimiter}作为中间方推动人质交换进程{tuple_delimiter}2)
(R{tuple_delimiter}昆塔拉{tuple_delimiter}菲鲁扎巴德{tuple_delimiter}担任人质交换的调解方{tuple_delimiter}2)
(R{tuple_delimiter}萨姆·纳马拉{tuple_delimiter}阿拉米亚监狱{tuple_delimiter}曾在此监狱服刑{tuple_delimiter}8)
(R{tuple_delimiter}萨姆·纳马拉{tuple_delimiter}梅吉·塔兹巴{tuple_delimiter}同批获释的人质{tuple_delimiter}2)
(R{tuple_delimiter}萨姆·纳马拉{tuple_delimiter}杜克·巴塔格拉尼{tuple_delimiter}同批获释的人质{tuple_delimiter}2)
(R{tuple_delimiter}梅吉·塔兹巴{tuple_delimiter}杜克·巴塔格拉尼{tuple_delimiter}同批获释的人质{tuple_delimiter}2)
(R{tuple_delimiter}萨姆·纳马拉{tuple_delimiter}菲鲁扎巴德{tuple_delimiter}被该国扣押为人质{tuple_delimiter}2)
(R{tuple_delimiter}梅吉·塔兹巴{tuple_delimiter}菲鲁扎巴德{tuple_delimiter}被该国扣押为人质{tuple_delimiter}2)
(R{tuple_delimiter}杜克·巴塔格拉尼{tuple_delimiter}菲鲁扎巴德{tuple_delimiter}被该国扣押为人质{tuple_delimiter}2){completion_delimiter}
######################
-实际数据-
######################
实体类型: {entity_types}
文本：{input_text}
######################
输出：
"""

# 使用format方法格式化字符串
prompt = Entity_Relation_Extraction_Prompt.format(
    entity_types=entity_types,
    input_text=input_text,
    language=language,
    tuple_delimiter=tuple_delimiter,
    completion_delimiter=completion_delimiter
)

print(prompt)