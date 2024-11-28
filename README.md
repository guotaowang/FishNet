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
