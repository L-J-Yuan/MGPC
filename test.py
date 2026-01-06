import argparse
import torch
from dataset.MGPC_1M import MGPC_1M
from utils.metrics import CD, CD_sqrt, F_score
from models.MGPC import MGPC, MGPC_2048
from utils.misc import seed_all, worker_init_fn
from collections import OrderedDict
from tqdm import tqdm
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


def test(args):

    device = args.device
    seed_all(args.seed)

    # mdoel
    net = MGPC()

    # load ckpt
    ckpt = torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if k.startswith('vae.'):
            continue
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=True)
    net.to(device)
    net.eval()


    # load dataset
    test_dset = MGPC_1M(args.dataset_path, "test")
    test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=args.test_batch_size,
                                        shuffle = False, 
                                        drop_last = False,
                                        num_workers = args.num_workers,
                                        worker_init_fn=worker_init_fn)

    cd_l2_list, cd_l1_list, Fscore_list = [], [], []

    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc=f"Test:", mininterval=10)
        for data in progress_bar:
            x = data[0].to(device)
            gt = data[1].to(device)
            rgb = data[2].to(device)
            text = data[3]
            data_path = data[4]
            
            recons, _, _ = net.predict(x, rgb, text)

            # evaluation metrics
            if args.num_points != 8192:
                gt = gather_operation(gt.permute(0, 2, 1).contiguous(), 
                                          furthest_point_sample(gt, args.num_points)).permute(0, 2, 1).contiguous()
            cd_l2 = 1000 * CD(recons, gt)
            cd_l1 = 1000 * CD_sqrt(recons, gt)
            fscore = F_score(recons, gt, th=0.01)


            if not isinstance(fscore, torch.Tensor):
                fscore = torch.tensor(fscore, device=device)

            cd_l2_list.append(cd_l2.item())
            cd_l1_list.append(cd_l1.item())
            Fscore_list.append(fscore.item())
    
        avg_cd_l2 = sum(cd_l2_list) / len(cd_l2_list)
        avg_cd_l1 = sum(cd_l1_list) / len(cd_l1_list)
        avg_Fscore = sum(Fscore_list) / len(Fscore_list)
        print("Avg CD_L2: ", avg_cd_l2)
        print("Avg CD_L1: ", avg_cd_l1)
        print("Avg F1: ", avg_Fscore)
    
        return avg_cd_l2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/path_to/mgpc_ckpt.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--dataset_path', type=str, default="/path_to/MGPC-1M/")
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=8192)
    args = parser.parse_args()
    test(args)
