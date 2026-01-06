import argparse, os
import torch
import numpy as np
import open3d as o3d
from torchvision import transforms
from PIL import Image

from models.MGPC import MGPC, MGPC_2048
from utils.misc import resume_model



def data_processing(pc_path, rgb_path, category, device):
    pc_input = np.load(pc_path)
    pc_input = torch.from_numpy(pc_input).float().unsqueeze(0).to(device)
    
    rgb_normalize = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(rgb_path).convert('RGB')
    image = rgb_normalize(image).unsqueeze(0).to(device)

    prompt = [f"This is a/an \"{category}\""]

    return pc_input, image, prompt



def demo(args):
    device = args.device

    # data
    input_dir = args.input_dir
    pc_path = os.path.join(input_dir, "input.npy")
    rgb_path = os.path.join(input_dir, "cropped.png")
    folders = input_dir.rstrip('/').split('/')
    category = folders[-2]
    pc_input, image, prompt = data_processing(pc_path, rgb_path, category, device)

    # model
    net = MGPC(args).to(device)
    resume_model(net, args.model_path)
    net.eval()

    with torch.no_grad():
        recons, recons_low, recons_coarse = net.predict(pc_input, image, prompt)
        
    # visualization
    if args.visualize:
        pred = recons[0].detach().cpu().numpy()
        input = pc_input[0].detach().cpu().numpy()
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pred)
        pcd_input = o3d.geometry.PointCloud()
        pcd_input.points = o3d.utility.Vector3dVector(input)
        pcd_input.paint_uniform_color([0.43, 0.72, 0.76])
        pcd_pred.paint_uniform_color([0.79, 0.54, 0.53])
        o3d.visualization.draw_geometries([pcd_input, axis], window_name="input")
        o3d.visualization.draw_geometries([pcd_pred, axis], window_name="pred")
        o3d.visualization.draw_geometries([pcd_pred, pcd_input, axis], window_name="pred and input")
        # save_path = os.path.join(input_dir, "pred.npy")
        # np.save(save_path, pred)








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/path_to/mgpc_ckpt.pt')
    parser.add_argument('--input_dir', type=str, default="./demo/controller/view2")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()
    demo(args)