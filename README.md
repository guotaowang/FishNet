<h1 align="center">🌟WinDB🌟 HMD-Free and Distortion-Free Panoptic Video Fixation Learning (TPAMI 2025)</h1>

<div align='center'>
    <a href='https://scholar.google.com/citations?user=eJIysC8AAAAJ' target='_blank'><strong>Guotao Wang</strong></a><sup> 1</sup>,&thinsp;
    <a href='http://chenglizhaochen.cn/' target='_blank'><strong>Chenglizhao Chen</strong></a><sup> 2, 6</sup>,&thinsp;
    <a href='https://dblp.org/pid/94/5679.html' target='_blank'><strong>Aimin Hao</strong></a><sup> 1</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?hl=en&user=NOcejj8AAAAJ&view_op=list_works&sortby=pubdate' target='_blank'><strong>Hong Qin</strong></a><sup> 3</sup>,&thinsp;
    <a href='https://dengpingfan.github.io/' target='_blank'><strong>Deng-Ping Fan</strong></a><sup> 4, 5</sup>,&thinsp;
</div>

<div align='center'>
    <sup>1 </sup>Beihang University&ensp;  <sup>2 </sup>China University of Petroleum&ensp;  <sup>3 </sup>Stony Brook University&ensp; <sup>4 </sup>Nankai University&ensp; 
    <br />
    <sup>5 </sup>Nankai International Advanced Research Institute (SHENZHEN FUTIAN)&ensp;  <sup>6 </sup>Sichuan Provincial Key Laboratory of Criminal Examination&ensp; 
</div>

   

📄 [**Arxiv**](https://arxiv.org/pdf/2305.13901) | 🌐 [**中文**]()

<div align="center">
  <img width="1000" height="210" src="https://github.com/guotaowang/WinDB/blob/main/Figs/1732611209391.gif"/>
</div>

The existing HMD-based fixation collection method for panoptic data has a critical limitation --- **blind zoom**, results in the collected fixations being insufficient to train deep models to accurately predict which regions in a given panoptic are most important.   

## News :newspaper:
* **`Nov 29, 2024`:**  We uploaded the [FishNet]() model weight.
* **`Nov 23, 2024`:** Our WinDB is now officially released [online]() on TPAMI journal.
* **`Nov 27, 2024`:** We released and uploaded the Chinese version of our paper to my [Baidu Netdisk]().
* **`Mar 7, 2024`:**  We released FishNet codes, [Baidu Netdisk]().
* **`Mar 7, 2024`:**  We released PanopticVideo-300 dataset, [Baidu Netdisk]().
* **`Sep 27, 2023`:**  We released WinDB codes, [Baidu Netdisk]().
* **`May 23, 2023`:**  We released our paper on [arXiv](https://arxiv.org/abs/2305.13901).

---

## 🎣 FishNet Architecture  

<div align="center">
  <img width="1000" height="480" src="https://github.com/guotaowang/WinDB/blob/main/Figs/1732611673627.gif"/>
</div>

<p align="center"><b>Fig.</b> The detailed network architecture of our FishNet.</p>   

**A** focuses on performing ERP-based global feature embedding to achieve panoptic perception and avoid visual distortion.  
**B** catches fixation shifting by refocusing the network to avoid the compression problem of shifted fixations in SOTA models.  
**C** makes the network fully aware of the fixation shifting mechanism to ensure that the network is sensitive to fixation shifting.  

<div align="center">
<img width="400" height="250" src="https://github.com/guotaowang/WinDB/blob/main/Figs/1732614599303.gif"/> <img width="400" height="170" src="https://github.com/guotaowang/WinDB/blob/main/Figs/1732614640036.gif"/>
</div>

<p align="center"><b>Fig.</b> Detailed calculation of the spherical distance. <b>Fig.</b> Visualizing of the ``shifting-aware feature enhancing''.</p>  

### 🛠️ Key Steps for FishNet (CODE: https://github.com/guotaowang/FishNet/tree/main)

1. **Training Process**  
   ```bash
   python train.py
   ```

2. **Inference Process**  
   ```bash
   python test.py
   ```

3. **Model Weight**  
   - [Model.pt](https://pan.baidu.com/s/1LeiX-p9YsAhrqTd2jq0Dfw) (97.9 MB) 

4. **Results**  
   - Results are stored in the output directory.

<div align="center">
  <img width="800" height="350" src="https://github.com/guotaowang/WinDB/blob/main/Figs/1732611730375.gif"/>
</div>

---

### 📊 Evaluation

1. **Score of Each Testing Set Clip**  
   ```bash
   MatricsOfMyERP.m
   ```

2. **Score of Entire Testing Set**  
   ```bash
   MatricsOfMyALLERP.m
   ```

<div align="center">
  <img width="600" height="290" src="https://github.com/guotaowang/WinDB/blob/main/Figs/1732611752777.gif"/>
</div>

---

## 📜 Citation  
If you use **WinDB**, please cite the following paper:

```bibtex
@article{wang2023windb,
  title={WinDB: HMD-free and Distortion-free Panoptic Video Fixation Learning},
  author={Wang, Guotao and Chen, Chenglizhao and Hao, Aimin and Qin, Hong and Fan, Deng-Ping},
  journal={arXiv preprint arXiv:2305.13901},
  year={2023}
}
```
