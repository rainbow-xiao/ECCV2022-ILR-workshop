# Google Universal Image Embedding Challenge 2nd Place Solution

#### [Competition on kaggle](https://www.kaggle.com/competitions/google-universal-image-embedding/)
#### [Instance-Level Recognition workshop](https://ilr-workshop.github.io/ECCVW2022/)
#### [My kaggle profile](https://www.kaggle.com/w3579628328)
#### [Inference notebook](https://www.kaggle.com/code/w3579628328/2nd-place-solution)

## HARDWARE & SOFTWARE

Ubuntu 18.04.3 LTS

CPU: AMD EPYC 7543 32-Core Processor

GPU: 6 * NVIDIA A40 PCIe, Memory: 48G

Python: 3.8

Pytorch: 1.9.0+cu111

## Data Preparation
1. Download all data from the data source below:

    [Aliproducts](https://tianchi.aliyun.com/competition/entrance/231780/introduction)
    
    [Art_MET](https://www.kaggle.com/datasets/dschettler8845/the-met-dataset)
    
    [DeepFashion(Consumer-to-shop)](https://www.kaggle.com/datasets/sangamman/deepfashion-consumer-to-shop-training)
    [DeepFashion2(hard-triplets)](https://www.kaggle.com/datasets/sangamman/deepfashion2-hard-triplets)
    [Fashion200K](https://www.kaggle.com/datasets/mayukh18/fashion200k-dataset)
    [ICCV 2021 LargeFineFoodAI](https://www.kaggle.com/competitions/largefinefoodai-iccv-recognition/data)
    [Food Recognition 2022](https://www.kaggle.com/datasets/sainikhileshreddy/food-recognition-2022)
    [JD_Products_10K](https://www.kaggle.com/c/products-10k)
    [Landmark2021](https://www.kaggle.com/competitions/landmark-retrieval-2021)
    [Grocery Store](https://github.com/marcusklasson/GroceryStoreDataset)
    [rp2k](https://www.pinlandata.com/rp2k_dataset/)
    [Shopee](https://www.kaggle.com/competitions/shopee-product-matching)
    [Stanford_Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
    [Stanford_Products](https://cvgl.stanford.edu/projects/lifted_struct/)

2. Run **Get_Data.ipynb** to create a csv file to corresponds to images for each dataset.

3. Run **Data_preprocessing.ipynb** to filter out classes with less than 3 images, and resize all images to 224.

4. Run **Data_Merge.ipynb** to merge all the csvs, and do sampling and resamping. Will get final_data_224_sample_balance_fold.csv finally. 

## Model Preparation
1. Pre-trained ViT-H-14 from [open_clip](https://github.com/mlfoundations/open_clip)

2. Get the visual module:
```bash
import open_clip
import torch
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir='./pretrained_models')
model_visual = model.visual
torch.save(model_visual.state_dict(), './pretrained_models/ViT_H_14_2B_vision_model.pt')
```

## Training
1. All configurations for **ViT-H-14-Visual** can be found in ./GUIE/config_clip_224.py

2. Training:
```bash
!CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python -m torch.distributed.launch --nproc_per_node=6 \
./GUIE/train.py \
--csv-dir ./final_data_224_sample_balance_fold.csv \
--config-name 'vit_224' \
--image-size 224 \
--batch-size 32 \
--num-workers 10 \
--init-lr 1e-4 \
--n-epochs 10 \
--cpkt_epoch 10 \
--n_batch_log 300 \
--warm_up_epochs 1 \
--fold 1
```

## Contact
Feel free to contact me if you have questions, email: 3579628328@qq.com
