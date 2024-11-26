
### 🎣 FishNet   

<div align="center">
  <img width="1000" height="410" src="https://github.com/guotaowang/FishNet/blob/main/Fig/Net.gif"/>
</div>

<p align="center"><b>Fig.</b> The detailed network architecture of our Fixation Shifting Network (FishNet). Our FishNet has three major components:</p>

1. **Component A**: Focuses on performing ERP-based global feature embedding to achieve panoptic perception and avoid visual distortion.  
2. **Component B**: Catches fixation shifting in PanopticVideo-300 by refocusing the network to avoid the compression problem of shifted fixations in SOTA models.  
3. **Component C**: Makes the network fully aware of and learns the fixation shifting mechanism to ensure that the network is sensitive to fixation shifting.  

---

### 🛠️ Key Steps

1. **Training Process**  
   ```bash
   python train.py
   ```

2. **Inference Process**  
   ```bash
   python test.py
   ```

3. **Model Weight**  
   - `Model.pt` (97.9 MB)

4. **Results**  
   - Results are saved in the output directory.

---

### 📊 Evaluation

1. **Score of Every Testing Set Clip**  
   ```bash
   MatricsOfMyERP.m
   ```

2. **Score of the Entire Testing Set**  
   ```bash
   MatricsOfMyALLERP.m
   ```
