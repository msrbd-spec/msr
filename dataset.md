নিচে thyroid nodule আর cervical lymph node (metastasis) — দুটোর জন্যই publicly available ultrasound dataset গুলো দিলাম। প্রায় সবগুলোই segmentation mask আকারে থাকে, তাই annotation থেকে bounding box বানিয়ে COCO detection format এ convert করে নিতে হবে (mask → contour → bbox, তারপর COCO JSON schema তে সাজানো)।

## Thyroid Nodule Datasets

**TN-SCUI 2020** — সবচেয়ে বড় thyroid nodule ultrasound dataset (৪৫৫৪ images, benign/malignant mask + label)
- Link: https://tn-scui2020.grand-challenge.org/Source_links/

**DDTI (Digital Database Thyroid Image)** — ৬৩৭ images, pixel-level mask, classic benchmark dataset
- Link: http://cimalab.intec.co/applications/thyroid/

**TN3K + TG3K** (TRFE-Net paper থেকে) — TN3K এ ৩৪৯৩ nodule-segmentation images, TG3K এ পুরো gland segmentation এর জন্য ৩৫৮৩ images। ready-made train/test split JSON সহ পাওয়া যায়
- Link: https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation
- Mirror doc: https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md

**TN5000** — নতুন dataset, detection-ready (PASCAL VOC style annotation, তাই COCO তে convert করা আরও সহজ হবে), ৫০০০ images, biopsy-confirmed label সহ
- Paper: https://www.nature.com/articles/s41597-025-05757-4

## Cervical Lymph Node (Metastasis) Datasets

**LymphUs** — এই মুহূর্তে সবচেয়ে relevant, কারণ এটা specifically PTC (papillary thyroid carcinoma) patients-দের cervical lymph node metastasis ultrasound dataset, ৩৩৮ patient, segmentation mask + malignant/benign label সহ, দুই center থেকে multicenter data
- Kaggle: https://www.kaggle.com/datasets/aliabbasianardakani/a-multicenter-lymph-node-ultrasound-image-database
- Paper: https://pubmed.ncbi.nlm.nih.gov/41940127/

এটা ছাড়া cervical lymph node metastasis এর জন্য fully standalone public dataset বেশ কম — বেশিরভাগ গবেষণায় single-institution retrospective data ব্যবহার হয়েছে যেগুলো public না (যেমন Hunan Cancer Hospital এর FADLM study, বা multicenter radiomics study গুলো)। তাই LymphUs-ই এখন practically একমাত্র সরাসরি download-যোগ্য cervical LN metastasis segmentation dataset।

## একটা practical suggestion
যেহেতু তুমি thyroid nodule + cervical LN metastasis দুটোই একসাথে detect করতে চাও:
- Thyroid nodule অংশের জন্য TN3K/TN-SCUI/DDTI মিলিয়ে বড় combined dataset বানানো যায় (paper গুলোতেও এভাবেই করা হয়)
- LymphUs থেকে LN mask গুলো secondary class হিসেবে যোগ করে একটাই multi-class COCO dataset (thyroid_nodule, lymph_node_metastasis) বানানো যেতে পারে
- মূল challenge হবে class imbalance — thyroid nodule dataset অনেক বড়, LN dataset তুলনামূলক ছোট, তাই training এ oversampling/augmentation লাগবে

চাইলে আমি DDTI/TN3K/LymphUs এর mask → COCO bbox conversion script লিখে দিতে পারি, বলো।
