### FishNet
<div align=center><img width="600" height="400" src="https://github.com/guotaowang/FishNet/tree/main/Fig/Net.gif"/></div>
<p align="center">
Fig. The motivation of the newly proposed model.   
Subfigures A and B illustrate the “fixation shifting” phenomenon — very common in our set.   
Our model has devised “a very simple yet effective” architecture, which performs spatiotemporal self-attention to alleviate the fixation shifting-induced longdistance misalignment problem. </p>     

  * 1) The **Training** Process    
     ```Python main.py --- Train=True```  
  * 2) The **Inference** Process    
     ```Python main.py --- Test=True```  
  * 3) The **Model Weight**   
     Model.pt (51.2MB)
  * 4) Results  
     Results  
### Evaluation  
  * 1) The score of every **Testing set clip**  
  ```MatricsOfMyERP.m```  
  * 2) The score of the **All testing set**   
  ```MatricsOfMyALLERP.m```
