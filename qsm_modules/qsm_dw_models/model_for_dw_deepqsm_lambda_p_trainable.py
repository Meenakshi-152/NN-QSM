
import torch

import torch.nn as nn

class DeepQSM(nn.Module):
    
    def __init__(self):
        super(DeepQSM, self).__init__()
        
        
        # (inp channels, Out Channels, Kernel Size, Stride, Padding)
        
        # Down 1
        
        self.conv10 = nn.Sequential(nn.Conv3d(1,16,3,1,1),nn.BatchNorm3d(16),
        nn.ReLU())
        
        self.conv11 = nn.Sequential(nn.Conv3d(16,16,3,1,1),nn.BatchNorm3d(16),
        nn.ReLU())
        
        self.down1  = nn.MaxPool3d(2,2)
        
        # Down 2
        
        self.conv20 = nn.Sequential(nn.Conv3d(16,32,3,1,1),nn.BatchNorm3d(32),
        nn.ReLU())
        
        self.conv21 = nn.Sequential(nn.Conv3d(32,32,3,1,1),nn.BatchNorm3d(32),
        nn.ReLU())
        
        self.down2  = nn.MaxPool3d(2,2)

        # Down 3
        
        self.conv30 = nn.Sequential(nn.Conv3d(32,64,3,1,1),nn.BatchNorm3d(64),
        nn.ReLU())
        
        self.conv31 = nn.Sequential(nn.Conv3d(64,64,3,1,1),nn.BatchNorm3d(64),
        nn.ReLU())
        
        self.down3  = nn.MaxPool3d(2,2)

         # Down 4
        
        self.conv40 = nn.Sequential(nn.Conv3d(64,128,3,1,1),nn.BatchNorm3d(128),
        nn.ReLU())
        
        self.conv41 = nn.Sequential(nn.Conv3d(128,128,3,1,1),nn.BatchNorm3d(128),
        nn.ReLU())
        
        self.down4  = nn.MaxPool3d(2,2)

        # Middle

        self.convm1 = nn.Sequential(nn.Conv3d(128,256,3,1,1),nn.BatchNorm3d(256),
        nn.ReLU())
        
        self.convm2 = nn.Sequential(nn.Conv3d(256,256,3,1,1),nn.BatchNorm3d(256),
        nn.ReLU())

        # Up 1

        self.up1   = nn.ConvTranspose3d(256,128,2,2)

        self.conv50 = nn.Sequential(nn.Conv3d(256,128,3,1,1),nn.BatchNorm3d(128),
        nn.ReLU())
        
        self.conv51 = nn.Sequential(nn.Conv3d(128,128,3,1,1),nn.BatchNorm3d(128),
        nn.ReLU())

        # Up 2

        self.up2   = nn.ConvTranspose3d(128,64,2,2)

        self.conv60 = nn.Sequential(nn.Conv3d(128,64,3,1,1),nn.BatchNorm3d(64),
        nn.ReLU())
        
        self.conv61 = nn.Sequential(nn.Conv3d(64,64,3,1,1),nn.BatchNorm3d(64),
        nn.ReLU())

        # Up 3

        self.up3   = nn.ConvTranspose3d(64,32,2,2)

        self.conv70 = nn.Sequential(nn.Conv3d(64,32,3,1,1),nn.BatchNorm3d(32),
        nn.ReLU())
        
        self.conv71 = nn.Sequential(nn.Conv3d(32,32,3,1,1),nn.BatchNorm3d(32),
        nn.ReLU())

        # Up 4

        self.up4   = nn.ConvTranspose3d(32,16,2,2)

        self.conv80 = nn.Sequential(nn.Conv3d(32,16,3,1,1),nn.BatchNorm3d(16),
        nn.ReLU())
        
        self.conv81 = nn.Sequential(nn.Conv3d(16,16,3,1,1),nn.BatchNorm3d(16),
        nn.ReLU())

        # Final

        self.out = nn.Conv3d(16,1,1,1)


        
    def forward(self,x):

      # Down
        
        x10 = self.conv10(x)
        x11 = self.conv11(x10)
        
        xd1 = self.down1(x11)
        
        x20 = self.conv20(xd1)
        x21 = self.conv21(x20)
        
        xd2 = self.down2(x21)

        x30 = self.conv30(xd2)
        x31 = self.conv31(x30)
        
        xd3 = self.down3(x31)

        x40 = self.conv40(xd3)
        x41 = self.conv41(x40)
        
        xd4 = self.down4(x41)

      # Middle

        xm1 = self.convm1(xd4)
        xm2 = self.convm2(xm1)

      # Up

        xu1  = self.up1(xm2)   

        xc1  = torch.cat((xu1,x41),dim = 1)

        x50 = self.conv50(xc1)
        x51 = self.conv51(x50)

        xu2  = self.up2(x51)
      
        xc2  = torch.cat((xu2,x31),dim = 1)

        x60 = self.conv60(xc2)
        x61 = self.conv61(x60)

        xu3  = self.up3(x61)
      
        xc3  = torch.cat((xu3,x21),dim = 1)

        x70 = self.conv70(xc3)
        x71 = self.conv71(x70)

        xu4  = self.up4(x71)
      
        xc4  = torch.cat((xu4,x11),dim = 1)

        x80 = self.conv80(xc4)
        x81 = self.conv81(x80)

        Out = self.out(x81)        
        
        return Out

    def getnumberofparams(self,net):
      pp=0
      for p in (net.parameters()):
        nn=1
        for s in p.size():
          nn=nn*s
        pp+=nn      
      return pp
        

def tic():
        # Homemade version of matlab tic and toc functions
        import time
        global startTime_for_tictoc
        startTime_for_tictoc = time.time()
    
def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        
        
if __name__ == '__main__':
    net = DeepQSM()
    net=net.float()        
    #net=net.cuda(1)
    
    x = torch.randn(1,2,176,176,160, dtype=torch.float)    
    x=x.float()
    #x=x.cuda(1)
    
    print('input' + str(x.size()))
    print(x.dtype)
    tic()
    with torch.no_grad():
        y = net(x)
    toc()
    print('output'+str(y.size()))        
    print(net.getnumberofparams(net))
        
        
        
