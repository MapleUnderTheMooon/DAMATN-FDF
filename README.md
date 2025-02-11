# DAMATN-FDF

# Usage

1. Clone the repo:
```
git clone https://github.com/MapleUnderTheMooon/DAMATN-FDF.git 
cd DAMATN-FDF
```
2. Put the data in [data/2018LA_Seg_Training Set](https://github.com/yulequan/UA-MT/tree/master/data/2018LA_Seg_Training%20Set).


```
data/
└── 2018LA_Seg_Training Set/
├── test.list
└── train.list
```



3. Train the model
```
cd code
bash ./train.sh
```

4. Test the model
```
python my_test_LA.py --model DAMATN-FDF
```
Our pre-trained models are saved in the model dir [DAMATN-FDF_model](https://github.com/MapleUnderTheMooon/DAMATN-FDF/tree/main/model) (both 8 labeled images and 16 labeled images), and the pretrained SASSNet and UAMT model can be download from [SASSNet_model](https://github.com/kleinzcy/SASSnet/tree/master/model) and [UA-MT_model](https://github.com/yulequan/UA-MT/tree/master/model). The other comparison method can be found in [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

