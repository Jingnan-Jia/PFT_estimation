from lung_function.modules.networks.models_pcd.pointnet2_reg import PointNet2_reg_extractor
from torch import nn
import torch
import torch.nn.functional as F
from lung_function.modules.path import PFTPath
import os
from lung_function.modules.networks.x3d import create_x3d_extractor



    
class CombinedNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_class = len(args.target.split('-'))
        nb_feature_ct = 192
        nb_feature_pcd = 1024
        
        if args.combined_by_add:
            self.nb_feature = nb_feature_ct
            self.fc192 = nn.Linear(nb_feature_pcd, nb_feature_ct)
            self.bn192 = nn.InstanceNorm1d(192) 
        else:
            self.nb_feature = nb_feature_ct + nb_feature_pcd
        
        ct_net_name, pcd_net_name = args.net.split('-')
        
        if ct_net_name == 'x3d_m':
            self.ct_net_extractor = create_x3d_extractor(input_channel=1)
        else:
            raise Exception(f'Unknown ct net name {ct_net_name}')
        
        if pcd_net_name == 'pointnet2_reg':
            self.pcd_net_extractor = PointNet2_reg_extractor(in_channel=args.in_channel, 
                                                             npoint_base=args.npoint_base, 
                                                             radius_base=args.radius_base, 
                                                             nsample_base=args.nsample_base)
        else:
            raise Exception(f"unknown pcd net name {pcd_net_name}")

        if args.pretrained_ct in ['ct', 'video']:
            
            if args.pretrained_ct == 'ct':
                if 'x3d_m' == ct_net_name:
                    pre_trained_ids = [2751,2760,2765,2771]
                else:
                    raise Exception(f'this net does not have pretrianed weights {ct_net_name}')

                pre_trained_id = pre_trained_ids[args.fold-1]
                mypath = PFTPath(pre_trained_id, check_id_dir=False, space=args.ct_sp)
                if os.path.exists(mypath.model_fpath):
                        ckpt = torch.load(mypath.model_fpath)
                        if isinstance(ckpt, dict) and 'model' in ckpt:
                            model = ckpt['model']
                        else:
                            model = ckpt
                        # model_fpath need to exist
                        self.ct_net_extractor.load_state_dict(model, strict=False)
                        print(f"load net from {mypath.model_fpath}")
                else:
                    raise Exception(f"model does not exist at {mypath.model_fpath}")
            else:  # TODO: load weights from videso
                net_video = torch.hub.load( 'facebookresearch/pytorchvideo', ct_net_name, pretrained=True)
                self.ct_net_extractor.load_state_dict(net_video.state_dict(), strict=False)
            
        if args.pretrained_pcd in ['vessel_skeleton_pcd', 'lungmask_pcd', 'modelnet40_pcd', 'vessel_surface_pcd']:
            if args.pretrained_pcd == 'vessel_skeleton_pcd' and pcd_net_name == 'pointnet2_reg':
                pre_trained_ids = [2336, 2343, 2346, 2350]
                pre_trained_id = pre_trained_ids[args.fold-1]
                mypath = PFTPath(pre_trained_id, check_id_dir=False, space=args.ct_sp)
                
                if os.path.exists(mypath.model_fpath):
                        ckpt = torch.load(mypath.model_fpath)
                        if isinstance(ckpt, dict) and 'model' in ckpt:
                            model = ckpt['model']
                        else:
                            model = ckpt
                        # model_fpath need to exist
                        self.pcd_net_extractor.load_state_dict(model, strict=False)
                        print(f"load net from {mypath.model_fpath}")
                else:
                    raise Exception(f"model does not exist at {mypath.model_fpath}")
            else:
                raise Exception(f"unset model fpath for {args.pre_trained_pcd}")
            
        
        
        self.fc1 = nn.Linear(self.nb_feature, 512)
        self.bn1 = nn.InstanceNorm1d(512) 
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.InstanceNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)
        
        self.norm192 = nn.InstanceNorm1d(192) 
        self.norm1024 = nn.InstanceNorm1d(1024) 
        
        

        

    def forward(self, ct, pcd, out_features=False):  # x shape: (B,N)
        B = ct.shape[0]  # Batch, 3+1, N

        ct_features = self.ct_net_extractor(ct).reshape(B, 1, -1)
        ct_features_norm = self.norm192(ct_features)
        
        pcd_features = self.pcd_net_extractor(pcd).reshape(B, 1, -1)
        pcd_features_norm = self.norm1024(pcd_features)

        if self.nb_feature == 1216:  # concatenation
            pcd_features_to_192 = self.bn192(self.fc192(pcd_features))
            all_features = pcd_features_to_192 + ct_features_norm
        else:
            all_features = torch.concatenate((ct_features_norm, pcd_features_norm), axis=2)
        
        x = all_features.view(B, self.nb_feature)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        out = self.fc3(x)


        return out, ct_features, pcd_features
    
