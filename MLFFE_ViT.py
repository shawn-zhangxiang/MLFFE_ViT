import torch
from torch import nn
import torch.nn.functional as F
import pywt
import torchvision.models as models
import pytorch_wavelets.dwt.lowlevel as lowlevel

class DWTForward(nn.Module):
    
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
       
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        for j in range(self.J):
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh

class MLDWT_ViT(nn.Module): 
    def __init__(self, image_size=224, num_classes=2):
                        
        super().__init__() 
        
        self.xf1 = DWTForward(J=1, mode='zero', wave='haar')  
        self.xf2 = DWTForward(J=2, mode='zero', wave='haar')
        self.xf3 = DWTForward(J=3, mode='zero', wave='haar')
    
        weights = models.ViT_B_32_Weights.DEFAULT
        self.vit_model = models.vit_b_32(weights=weights)

        for param in self.vit_model.parameters():
            param.requires_grad = False

        self.vit_model.heads = nn.Sequential(nn.Linear(in_features=768, 
                                          out_features=num_classes))
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, int(image_size/4),3, 1, 1),
            nn.Conv2d(int(image_size/4), int(image_size/2), 3, 1, 1),
            nn.Conv2d(int(image_size/2), image_size, 4, 4),
            nn.GELU(),
            nn.BatchNorm2d(image_size)
            )
        
        self.reduce_ch = nn.Conv2d(image_size, int(image_size/4), 1)
         
        self.classification_block = nn.Sequential(
            nn.BatchNorm2d(672),
            nn.MaxPool2d(2, 2), 
            nn.Flatten(), 
            nn.Linear(672*3*3, 1024),
            nn.Dropout(p=0.3),
            nn.PReLU(), 
            nn.Linear(in_features= 1024, out_features=672), 
            nn.Dropout(p=0.3),
            nn.PReLU(), 
            nn.Linear(in_features=672, out_features=2)
        )
        
        self.conv2d=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)        
        
    def forward(self, img):
        
        x = self.conv_layer(img)  
        b, c, h, w = x.shape     
        
        x = self.reduce_ch(x)   
     
        Y1, Yh = self.xf1(x)    
        Y2, Yh = self.xf2(x)    
        Y3, Yh = self.xf3(x)   
    
        x1 = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))   
        x2 = torch.reshape(Yh[1], (b, int(c*3/4), int(h/4), int(w/4)))   
        x3 = torch.reshape(Yh[2], (b, int(c*3/4), int(h/8), int(w/8)))  
       
        x1 = torch.cat((Y1,x1), dim = 1)    
        x1 = F.adaptive_avg_pool2d(x1, (7, 7)) 
        
        x2 = torch.cat((Y2,x2), dim = 1)      
        x2 = F.adaptive_avg_pool2d(x2, (7, 7)) 
        
        x3 = torch.cat((Y3,x3), dim = 1)
        x = torch.cat((x1,x2,x3), dim = 1)  
       
        cls_out=self.classification_block(x) 
       
        vit_input=self.conv2d(img)
        v_out=self.vit_model(vit_input)
        
        f_out=cls_out+ v_out
       
        return f_out
