### FishNet   
<div align=center><img width="1000" height="410" src="https://github.com/guotaowang/FishNet/blob/main/Fig/Net.gif"/></div>
<p align="center">
 Fig. The detailed network architecture of our Fixation Shifting Network (FishNet). Our FishNet has three major components. Component A
focuses on performing ERP-based global feature embedding to achieve panoptic perception and avoid visual distortion. B catches fixation shifting in
PanopticVideo-300 by refocusing the network to avoid the compression problem of shifted fixations in SOTA models. C makes the network fully aware
of and learns the fixation shifting behind mechanism to ensure that the network is sensitive to fixation shifting.  </p>     

  * 1) The **Training** Process    
     ```Python train.py```  
  * 2) The **Inference** Process    
     ```Python test.py```  
  * 3) The **Model Weight**   
     Model.pt (97.9MB)
  * 4) Results  
     Results  
### Evaluation  
  * 1) The score of every **Testing set clip**  
  ```MatricsOfMyERP.m```  
  * 2) The score of the **All testing set**   
  ```MatricsOfMyALLERP.m```
