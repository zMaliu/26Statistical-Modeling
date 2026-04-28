## 一、当前可直接使用的数据表

### 1. 主面板

- 路径：`scheme2_main_city_panel.csv`
- 用途：作为整体数据底板，提供广东 21 市 `2015—2023` 年的基础城市—年份信息。
- 规模：`189` 个城市—年份观测值。

### 2. AI 集聚测度样本

- 路径：`scheme2_ai_measurement_panel.csv`
- 用途：用于构建和解释人工智能产业集聚指标。
- 规模：广东 `8` 个样本城市，`2020—2023` 年，共 `27` 个城市—年份观测值。

### 3. 创新支撑环境评价样本

- 路径：`scheme2_innovation_support_panel_upgraded.csv`
- 用途：用于构建创新支撑环境综合评价模型。
- 规模：广东 `21` 市，`2022—2023` 年，共 `42` 个城市—年份观测值。

### 4. 匹配样本

- 路径：`scheme2_ai_innovation_matched_panel.csv`
- 用途：用于 AI 集聚、创新支撑环境与协调发展之间的辅助性统计检验。
- 规模：广东 `8` 个样本城市，`2022—2023` 年，共 `14` 个城市—年份观测值。

### 5. 分层结果表

- 路径：`scheme2_city_stratification.csv`
- 用途：用于城市分层识别与城市画像分析。

### 6. 聚类诊断结果表

- 路径：`scheme2_cluster_diagnostics.csv`
- 用途：用于四象限分层的统计验证。

### 7. 辅助性统计结果表

- 描述统计：`scheme2_descriptive_statistics.csv`
- 相关矩阵：`scheme2_correlation_matrix.csv`
- 回归结果：`scheme2_regression_summary.csv`

---

## 二、论文需要采用的模型

---

## 模型 1：人工智能产业集聚测度模型

### 研究目的

构建城市—年份层面的人工智能产业集聚指标，识别广东样本城市之间 AI 集聚水平的差异及其阶段性变化。

### 使用数据表

- `scheme2_ai_measurement_panel.csv`

### 核心变量

- `ai_agglomeration_composite`：人工智能产业集聚综合指标
- `ai_company_count`：样本企业数量
- `ai_hit_company_count`：人工智能关键词命中企业数量
- `ai_hit_ratio`：人工智能命中企业占比
- `ai_keyword_mentions_company_sum`：人工智能关键词命中强度总量
- `ai_lq_company_hit`：基于命中企业数的区位商指标
- `ai_lq_keyword_mass`：基于关键词强度的区位商指标
- `ai_lq_ai_char_mass`：基于人工智能文本字符强度的区位商指标
- `ai_small_sample_flag`：小样本标记

### 需要完成的内容

1. 明确论文中是否直接采用 `ai_agglomeration_composite` 作为最终 AI 集聚指标；
2. 明确该指标由哪些分量构成；
3. 给出该指标的数学表达；
4. **核心测度模型**。

---

## 模型 2：创新支撑环境综合评价模型

### 研究目的

对广东 21 市创新支撑环境进行综合评价，识别不同城市在财政、金融、开放、消费与服务开放等方面的支撑强弱。

### 使用数据表

- `scheme2_innovation_support_panel_upgraded.csv`

### 核心变量

- `innovation_support_entropy_topsis_score`：创新支撑环境熵值—TOPSIS 综合得分
- `innovation_support_entropy_rank_within_year`：创新支撑环境年度排名
- `fiscal_intensity_ratio`：财政强度指标
- `financial_depth_ratio`：金融深度指标
- `fdi_gdp_ratio`：开放强度指标（外资占 GDP 比重）
- `retail_per_capita`：人均消费水平指标
- `service_openness_proxy`：服务开放代理指标

### 需要完成的内容

1. 明确评价指标体系；
2. 明确采用熵值法—TOPSIS 作为主评价模型；
3. 给出标准化、赋权、综合得分的数学表达；
4. **核心评价模型**。

---

## 模型 3：创新支撑环境稳健性评价模型

### 研究目的

检验创新支撑环境综合评价结果是否稳健，避免主评价结果过度依赖单一赋权方式。

### 使用数据表

- `scheme2_innovation_support_panel_upgraded.csv`

### 核心变量

- `innovation_support_pca_score`：创新支撑环境 PCA 得分
- `innovation_support_pca_rank_within_year`：创新支撑环境 PCA 年度排名
- `pca_explained_variance_ratio`：主成分解释方差比
- `fiscal_intensity_ratio`：财政强度指标
- `financial_depth_ratio`：金融深度指标
- `fdi_gdp_ratio`：开放强度指标
- `retail_per_capita`：人均消费水平指标
- `service_openness_proxy`：服务开放代理指标

### 需要完成的内容

1. 明确 PCA 的角色是**稳健性评价模型**，不是主模型；
2. 给出 PCA 得分的数学表达；
3. 说明 PCA 与熵值—TOPSIS 结果如何对照；
4. **稳健性检验模型**。

---

## 模型 4：城市分层识别模型

### 研究目的

识别不同城市在“AI 集聚—创新支撑”结构中的位置，解释城市之间的协同与错配关系。

### 使用数据表

- `data/processed/analysis_ready/scheme2_city_stratification.csv`

### 核心变量

- `ai_agglomeration_mean`：AI 集聚均值
- `innovation_support_mean`：创新支撑环境均值
- `coordination_capacity_mean`：协调发展均值
- `ai_level`：AI 集聚高低分层
- `innovation_support_level`：创新支撑高低分层
- `city_quadrant`：城市所属象限类型

### 当前已有分层结果

- 高集聚—高支撑：中山、珠海
- 高集聚—低支撑：佛山、惠州
- 低集聚—高支撑：广州、深圳
- 低集聚—低支撑：东莞、汕头

### 需要完成的内容

1. 明确四象限分层的划分依据；
2. 说明分层使用的两个维度；
3. 给出四象限分层的规则表达或数学表达；
4. **核心解释模型**。

---

## 模型 5：城市分层验证模型

### 研究目的

验证四象限分层是否具有一定统计合理性，使城市分层不只停留在经验划分层面。

### 使用数据表

- `data/processed/analysis_ready/scheme2_cluster_diagnostics.csv`

### 核心变量

- `k`：聚类类别数
- `silhouette_score`：轮廓系数
- `inertia`：类内离差平方和

### 需要完成的内容

1. 明确采用 K-means 作为分层验证模型；
2. 给出聚类目标函数；
3. 说明其角色是**验证模型**，不是主模型；
4. **分层验证模型**。

---

## 模型 6：描述统计模型

### 研究目的

刻画核心变量的基本分布特征，为后续评价、分层和辅助性回归提供统计画像。

### 使用数据表

- `scheme2_descriptive_statistics.csv`
- 必要时回到：
  - `scheme2_ai_measurement_panel.csv`
  - `scheme2_innovation_support_panel_upgraded.csv`
  - `scheme2_ai_innovation_matched_panel.csv`

### 核心变量

- `ai_agglomeration_composite`：人工智能产业集聚综合指标
- `innovation_support_entropy_topsis_score`：创新支撑环境综合得分
- `innovation_support_pca_score`：创新支撑环境稳健性得分
- `coordination_capacity_composite`：协调发展综合指标

### 需要完成的内容

1. 明确论文中描述统计的变量范围；
2. 明确描述统计应包含的统计量；
3. **基础统计模型**。

---

## 模型 7：相关性分析模型

### 研究目的

识别 AI 集聚、创新支撑环境与协调发展之间的线性关系方向和强弱，为后续结构解释和辅助性回归提供支持。

### 使用数据表

- `scheme2_correlation_matrix.csv`
- 必要时回到 `scheme2_ai_innovation_matched_panel.csv`

### 核心变量

- `ai_agglomeration_composite`：人工智能产业集聚综合指标
- `innovation_support_substitute_index`：创新支撑替代综合指数
- `innovation_support_entropy_topsis_score`：创新支撑环境综合得分
- `coordination_capacity_composite`：协调发展综合指标

### 需要完成的内容

1. 明确采用 Pearson 相关分析；
2. 给出相关系数的数学表达；
3. 明确重点解释哪些变量之间的关系；
4. **关系识别模型**。

---

## 模型 8：辅助性回归模型

### 研究目的

对 AI 集聚、创新支撑环境与协调发展之间的关系进行支持性检验，但不承担强因果识别任务。

### 使用数据表

- `scheme2_ai_innovation_matched_panel.csv`
- 结果参照：`scheme2_regression_summary.csv`

### 核心变量

#### 被解释变量

- `coordination_capacity_composite`：协调发展综合指标

#### 核心解释变量

- `ai_agglomeration_composite`：人工智能产业集聚综合指标

#### 创新支撑变量

- `innovation_support_substitute_index`：创新支撑替代综合指数
- `fiscal_intensity_ratio`：财政强度指标
- `financial_depth_ratio`：金融深度指标
- `fdi_gdp_ratio`：开放强度指标
- `retail_per_capita`：人均消费水平指标
- `service_openness_proxy`：服务开放代理指标

#### 控制变量

- `gdp`：地区生产总值
- `gdp_per_capita`：人均地区生产总值
- `population`：常住人口规模
- `retail_sales`：社会消费品零售总额
- `fdi_actual_used`：实际利用外资额
- `financial_deposit_loan`：金融机构存贷款规模
- `fiscal_expenditure`：财政支出规模
- `secondary_industry_share`：第二产业占比
- `tertiary_industry_share`：第三产业占比

### 需要完成的内容

1. 给出辅助性回归模型的统一数学表达；
2. 明确至少整理三类回归形式：
   - AI 单变量模型
   - AI + 创新支撑综合得分模型
   - AI + 创新支撑组成项模型
3. 明确说明该模型仅作支持性统计检验，不作因果识别；
4. **辅助检验模型**。
