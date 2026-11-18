# 生物医学骨干网络研究进展综述：从 CNN、Transformer 到状态空间模型与基础模型

> **检索与方法说明（可复现性记录）**
> - 数据库：PubMed（E-utilities esearch + esummary）、arXiv（Atom API）
> - 时间范围：2023-01-01 至 2026-12-31（按出版日期）；个别奠基性工作（如 Swin UNETR, 2022）作为背景引用
> - 检索日期：2026-11-18
> - 检索式与命中量（原始 JSON 存于 `sources/`）：
>   - 骨干/基础模型 × 医学图像：258 篇
>   - Vision Transformer × 医学图像：557 篇
>   - Mamba/状态空间模型 × 生物医学：183 篇
>   - 现代 CNN 骨干 × 医学图像：247 篇
>   - 基础模型/Transformer × ECG/EEG 等生理信号：961 篇
>   - Segment Anything × 医学图像：226 篇
> - 筛选方式：按相关性排序取各组前列记录，人工核对标题、期刊、年份、DOI/PMID 后纳入；所有引用元数据均来自上述 API 返回的原始记录。

---

## 1. 引言

骨干网络（backbone network）是深度学习模型的特征提取核心，其架构设计直接决定了下游任务（分割、分类、检测、配准等）的性能上限。在生物医学领域，数据具有维度高（3D CT/MRI 体数据、全切片病理图像 WSI）、标注稀缺、模态多样（影像、信号、文本）、类别不平衡显著等特点，对骨干网络的感受野、长程依赖建模能力与数据效率提出了远高于自然图像的要求。

2023 年以来，该领域经历了三条明显的技术演进脉络：其一，以 MedNeXt [1] 为代表的"现代化 CNN"证明纯卷积架构经合理扩展仍具竞争力；其二，Vision Transformer（ViT）与 Swin Transformer 在医学图像分析中迅速普及 [2, 3]，并成为 3D 分割的主流编码器 [4]；其三，以 Mamba 为代表的状态空间模型（State Space Model, SSM）凭借线性复杂度长序列建模能力，在 2024 年后成为高分辨率医学图像分割的新热点 [5, 6]。与此同时，以 Segment Anything Model（SAM）及其医学适配版本（MedSAM 等）为代表的**基础模型（foundation model）**范式 [7, 8]，以及 ECG/EEG 领域的信号基础模型 [9, 10]，正在把"骨干网络"的概念从单一任务特征提取器扩展为"预训练-适配"的通用表征平台。

本综述聚焦近 3 年（2023–2026）文献，系统梳理主流骨干架构的设计逻辑、在生物医学图像与信号中的典型应用，并讨论当前挑战与未来方向。

## 2. 主流骨干网络架构分析

### 2.1 CNN 及其现代化变体

卷积神经网络凭借局部归纳偏置和平移等变性，在小样本医学数据上长期占据主导。近 3 年的关键进展是"ConvNet 现代化"路线：MedNeXt [1]（2023）将 ConvNeXt 的大卷积核、倒置瓶颈等设计引入 3D 医学分割，并借鉴 Transformer 训练策略（长余弦调度、大 kernel 深度卷积），在多项 CT/MRI 分割基准上超越同期 Transformer 模型，证明卷积骨干在标注受限场景下的数据效率优势。CI-UNet [11]（2024）进一步将 ConvNeXt 块与跨维注意力融合用于通用医学图像分割。此类工作的共同启示是：架构之争中，训练策略与归纳偏置的匹配往往比"卷积还是注意力"本身更关键。

### 2.2 Vision Transformer 与 Swin 系列

ViT 通过自注意力实现全局感受野，弥补了 CNN 长程依赖建模的不足。在医学影像中，纯 ViT 受限于二次复杂度与数据需求，因此**层级化、带移位窗口的 Swin Transformer** 成为事实标准：Swin UNETR [4]（2022，MICCAI 脑瘤分割冠军方案）以 Swin 块构建 3D 编码器，是后续大量 3D 分割工作的骨干蓝本。近 3 年的改进主要沿三个方向展开：

- **效率化**：如 EViT-UNet [12]（2025）面向移动/边缘设备设计高效 ViT-UNet；
- **频域与抗混叠**：如 Spec-ViT [3]（2026）引入小波变换以抗混叠、去噪，提升医学图像分类鲁棒性；
- **半监督与低标注**：如 Semi-SwinUNeTR [13]（2026）在有限标注下训练 3D Swin-UNet。

综述性工作 [2]（2023）也指出，ViT 类骨干在小数据集上易过拟合，混合 CNN-Transformer 架构通常是更稳妥的选择。

### 2.3 Mamba 与状态空间模型

Mamba 类 SSM 以线性复杂度实现长序列选择性状态建模，天然契合高分辨率体数据与长信号序列。2024 年出现了两个奠基性医学适配工作：**U-Mamba** [5] 将 SSM 块嵌入 U-Net 编码器并与卷积块混合，兼顾局部纹理与长程依赖；**SegMamba** [6] 则面向 3D 体数据设计全尺度 SSM 编码器，在 CRC-500 等 3D 分割数据集上验证了吞吐与显存优势。此后 SSM 骨干迅速扩散：前列腺 MR 分割的混合 Mamba U-Net [14]（2024）、脑 MR-CT 可变形配准的 Mamba 网络 [15]（2025）等。整体趋势是从"替代注意力"转向"与卷积/注意力混合"的务实路线。

### 2.4 大规模基础模型：从 SAM 到视觉-语言骨干

2023 年后，"预训练大骨干 + 轻量适配"成为新范式。MedSAM [7]（Nature Communications, 2024）在大规模医学掩膜数据上微调 SAM，成为医学交互式分割的标志性工作；Medical SAM Adapter [8]（2025）则以参数高效适配方式将 SAM 迁移到多种医学模态；多篇综述系统评估了 SAM 系模型在医学图像中的能力边界与失效模式 [16]。在 CT 领域，Merlin [17]（Nature, 2026）构建了 CT 视觉-语言基础模型与配套数据集，将影像表征与报告文本对齐；生成式方向亦有自改进医学图像生成基础模型（Nature Medicine, 2025）[18]。基础模型正在把骨干网络的角色从"任务专属特征器"转变为"可复用的通用表征底座"。

### 2.5 架构对比小结

| 架构家族 | 代表工作 | 核心优势 | 主要局限 | 典型场景 |
|---|---|---|---|---|
| 现代 CNN | MedNeXt [1], CI-UNet [11] | 数据效率高、小样本稳健 | 长程依赖弱 | 小/中规模 3D 分割 |
| ViT / Swin | Swin UNETR [4], Spec-ViT [3] | 全局建模、易做多模态融合 | 数据与算力需求大 | 3D 分割、分类、配准 |
| Mamba / SSM | U-Mamba [5], SegMamba [6] | 线性复杂度、长序列/高分辨率友好 | 生态较新、工程成熟度低 | 高分辨率分割、长信号 |
| 基础模型 | MedSAM [7], Merlin [17] | 零样本/少样本迁移、通用表征 | 领域鸿沟、部署成本高 | 交互式分割、跨模态任务 |

## 3. 在生物医学图像与信号中的应用

### 3.1 医学图像分割

分割是骨干网络竞争最激烈的战场。3D 体数据方面，Swin UNETR [4]、SegMamba [6]、MedNeXt [1] 分别代表了 Transformer、SSM、现代 CNN 三条路线在脑瘤、腹部器官等基准上的最高水平；交互式分割则由 MedSAM [7]、ProtoSAM-3D [19]、SAM2 的 3D 腹部器官零样本扩展 [20] 等推动，显著降低了标注成本。专病场景亦有大量定制骨干，如胎儿 MRI 脑组织分割 [21]、卵巢超声多尺度分割 [22] 等。

### 3.2 分类、检测与配准

分类任务中，层级化多尺度 ViT 被用于 MRI 脑瘤检测与分型 [23]，视网膜 OCT 的元数据增强对比学习 [24] 展示了预训练骨干在眼科筛查中的价值；配准方面，Mamba 骨干被引入脑 MR-CT 可变形配准 [15] 与多模态配准的双流架构 [25]，利用长程建模改善大形变对齐。

### 3.3 生物医学信号：ECG 与 EEG

一维生理信号的骨干网络近 3 年同样经历了"基础模型化"：ECG 方向，Vaid 等 [10]（npj Digital Medicine, 2023）以视觉 Transformer 骨干预训练心电基础模型提升诊断性能；2025 年相继出现了基于逾千万条心电记录构建的基础模型（NEJM AI）[9] 与开源 ECG-FM [26]。EEG 方向，EEG Conformer [27]（2023）将卷积与 Transformer 结合用于脑电解码；睡眠 EEG 自监督基础模型 [28] 与图掩码自编码器 GMAEEG [29] 则探索了信号表征的通用化路径。与影像相比，信号基础模型更强调跨被试、跨设备的泛化与自监督预训练。

## 4. 挑战与展望

1. **数据规模与标注瓶颈**：医学标注昂贵，基础模型范式（预训练 + 适配/提示）与半监督骨干（如 Semi-SwinUNeTR [13]）是主流解法，但预训练数据与目标域的分布鸿沟仍未根本解决。
2. **架构选择与可复现性**：CNN、ViT、Mamba 各有适用边界，文献中混杂的基准与实现差异使横向比较困难；社区需要统一的评测协议与开源基准。
3. **3D 与高分辨率计算成本**：体数据与 WSI 对显存和吞吐要求苛刻，SSM 的线性复杂度与高效 ViT 是重要方向，但工程成熟度仍待提升。
4. **泛化与可靠性**：跨中心、跨设备、跨人群的域偏移是临床落地的核心障碍；视网膜血管的域泛化研究 [30] 提示骨干层面的几何/物理先验可能是破局点之一。
5. **多模态与临床闭环**：Merlin [17] 等视觉-语言模型预示着影像-报告-文本联合骨干的兴起；未来骨干网络将越来越多地承担"多模态临床表征底座"的角色，并需在隐私、可解释性与监管合规框架下验证。

**展望**：短期内，混合架构（CNN + 注意力 + SSM）与参数高效适配将主导实用系统；中期看，信号与影像各自的领域基础模型会继续扩大规模并走向开源（如 ECG-FM [26]）；长期看，统一影像、信号、文本与组学的多模态生物医学基础骨干，以及与之配套的严格临床验证体系，是该领域最值得期待的方向。

---

## 参考文献

1. Roy S, Koehler G, Ulrich C, Baumgartner M, Petersen J, et al. MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation. arXiv:2303.09975, 2023.
2. Zhang Y, Wang J, Gorriz JM, et al. Deep Learning and Vision Transformer for Medical Image Analysis. *Journal of Imaging*, 2023. DOI: 10.3390/jimaging9070147.
3. Zhang X, Wang L, Rao Y, et al. Spec-ViT: A Vision Transformer With Wavelet for Anti-Aliasing and Denoising in Medical Image Classification. *IEEE Journal of Biomedical and Health Informatics*, 2026. DOI: 10.1109/JBHI.2025.3640432.
4. Hatamizadeh A, Nath V, Tang Y, Yang D, Roth H, et al. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv:2201.01266, 2022.
5. Ma J, Li F, Wang B. U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation. arXiv:2401.04722, 2024.
6. Xing Z, Ye T, Yang Y, Liu G, Zhu L, et al. SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation. arXiv:2401.13560, 2024.
7. Ma J, He Y, Li F, et al. Segment anything in medical images. *Nature Communications*, 2024. DOI: 10.1038/s41467-024-44824-z.
8. Wu J, Wang Z, Hong M, et al. Medical SAM adapter: Adapting segment anything model for medical image segmentation. *Medical Image Analysis*, 2025. DOI: 10.1016/j.media.2025.103547.
9. Li J, Aguirre AD, Moura V, et al. An Electrocardiogram Foundation Model Built on over 10 Million Recordings. *NEJM AI*, 2025. DOI: 10.1056/aioa2401033.
10. Vaid A, Jiang J, Sawant A, et al. A foundational vision transformer improves diagnostic performance for electrocardiograms. *npj Digital Medicine*, 2023. DOI: 10.1038/s41746-023-00840-9.
11. Zhang Z, Wen Y, Zhang X, et al. CI-UNet: melding convnext and cross-dimensional attention for robust medical image segmentation. *Biomedical Engineering Letters*, 2024. DOI: 10.1007/s13534-023-00341-4.
12. Li X, Zhu W, Dong X, et al. EViT-UNet: U-Net like efficient vision transformer for medical image segmentation on mobile and edge devices. *Proc. IEEE ISBI*, 2025. DOI: 10.1109/ISBI60581.2025.10981108.
13. Tian Y, Wang Z, Guo L, et al. Semi-SwinUNeTR: Towards 3D Swin Vision Transformer-Based UNet for Medical Image Segmentation with Limited Annotations. *Bioengineering*, 2026. DOI: 10.3390/bioengineering13060695.
14. Du Q, Wang L, Chen H, et al. A mixed Mamba U-net for prostate segmentation in MR images. *Scientific Reports*, 2024. DOI: 10.1038/s41598-024-71045-7.
15. Wang Y, Guo T, Yuan W, et al. Mamba-based deformable medical image registration with an annotated brain MR-CT dataset. *Computerized Medical Imaging and Graphics*, 2025. DOI: 10.1016/j.compmedimag.2025.102566.
16. Zhang Y, Shen Z, Jiao R, et al. Segment anything model for medical image segmentation: Current applications and future directions. *Computers in Biology and Medicine*, 2024. DOI: 10.1016/j.compbiomed.2024.108238.
17. Blankemeier L, Kumar A, Cohen JP, et al. Merlin: a computed tomography vision-language foundation model and dataset. *Nature*, 2026. DOI: 10.1038/s41586-026-10181-8.
18. Wang J, Wang K, Yu Y, et al. Self-improving generative foundation model for synthetic medical image generation and clinical applications. *Nature Medicine*, 2025. DOI: 10.1038/s41591-024-03359-y.
19. Shen Y, Dreizin D, Inigo B, et al. ProtoSAM-3D: Interactive semantic segmentation in volumetric medical imaging via a Segment Anything Model and mask-level prototypes. *Computerized Medical Imaging and Graphics*, 2025. DOI: 10.1016/j.compmedimag.2025.102501.
20. Yamagishi Y, Hanaoka S, Kikuchi T, et al. Using Segment Anything Model 2 for Zero-Shot 3D Segmentation of Abdominal Organs in Computed Tomography Scans. *JMIR AI*, 2025. DOI: 10.2196/72109.
21. Huang X, Liu Y, Li Y, et al. Deep Learning-Based Multiclass Brain Tissue Segmentation in Fetal MRIs. *Sensors*, 2023. DOI: 10.3390/s23020655.
22. Wang Y, Zhang J, Han J, et al. Ovarian Ultrasound Image Segmentation Algorithm with Fused Multi-Scale Features. *Critical Reviews in Biomedical Engineering*, 2025. DOI: 10.1615/CritRevBiomedEng.v53.i1.40.
23. Sankari C, Jamuna V, Kavitha AR, et al. Hierarchical multi-scale vision transformer model for accurate detection and classification of brain tumors in MRI-based medical imaging. *Scientific Reports*, 2025. DOI: 10.1038/s41598-025-23100-0.
24. Holland R, Leingang O, Bogunović H, et al. Metadata-enhanced contrastive learning from retinal optical coherence tomography images. *Medical Image Analysis*, 2024. DOI: 10.1016/j.media.2024.103296.
25. Hu Y, Xu LX, Sun J, et al. A Dual-Stream Mamba With Contrastive Representation for Multimodal Deformable Registration. *Proc. IEEE EMBC*, 2025. DOI: 10.1109/EMBC58623.2025.11253168.
26. McKeen K, Masood S, Toma A, et al. ECG-FM: an open electrocardiogram foundation model. *JAMIA Open*, 2025. DOI: 10.1093/jamiaopen/ooaf122.
27. Song Y, Zheng Q, Liu B, et al. EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2023. DOI: 10.1109/TNSRE.2022.3230250.
28. Ogg M, Coon WG. Self-Supervised Transformer Model Training for a Sleep-EEG Foundation Model. *bioRxiv*, 2024. DOI: 10.1101/2024.01.18.576245.
29. Fu Z, Zhu H, Zhao Y, et al. GMAEEG: A Self-Supervised Graph Masked Autoencoder for EEG Representation Learning. *IEEE Journal of Biomedical and Health Informatics*, 2024. DOI: 10.1109/JBHI.2024.3443651.
30. Hu D, Li H, Liu H, et al. Domain generalization for retinal vessel segmentation via Hessian-based vector field. *Medical Image Analysis*, 2024. DOI: 10.1016/j.media.2024.103164.

---

## 附：工具使用声明

本文由 Kimi Code 调用 Scientific Agent Skills 中的 `paper-lookup`、`literature-review`、`scientific-writing` 技能辅助完成：文献检索基于 PubMed/arXiv 官方 API 的真实返回（原始记录存于 `sources/`），正文由 AI 起草，**所有引用元数据已经 API 记录核验，但事实性论断仍需作者逐条对照原文确认后方可用于正式投稿**。

> Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065, 2026. https://doi.org/10.48550/arXiv.2609.00065
