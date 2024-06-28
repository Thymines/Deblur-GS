#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import fnmatch
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from typing import Union


import open3d as o3d

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    test_image: np.array
    image_path: str
    image_name: str
    depth_image:  Union[np.array, None]
    depth_mask: Union[np.array, None]
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(path, cam_extrinsics, cam_intrinsics, images_folder)->list[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        test_image_path = os.path.join(path, 'images_test',os.path.basename(extr.name))
        if not os.path.exists(test_image_path):
            test_image_path = image_path
        test_image = Image.open(test_image_path)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,test_image=test_image,
                              image_path=image_path, image_name=image_name, width=width, height=height,depth_image=None,depth_mask=None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals = None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(path, cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        # _max, _min = xyz.max(0), xyz.min(0)
        # _len = _max - _min

        
        # _xyz, _rgb = __create_random_ply(None)
        # _xyz = (_xyz+1.3)/2*_len+_min

        # xyz = np.concatenate([xyz,_xyz])
        # rgb = np.concatenate([rgb,_rgb])
        
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],depth_image=None,depth_mask=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readHoloLensSceneInfo(path, eval, llffhold=8,voxel_size=1e-2):

    ply_path = os.path.join(path,'points.ply')
    pcd = None
    if not os.path.exists(ply_path):

        pcds = o3d.geometry.PointCloud()
        # 读取所有的深度相机所得到的点云文件
        for file in [name for name in os.listdir(os.path.join(path,"Depth Long Throw")) if fnmatch.fnmatch(name,'*.ply')]:
            pcd = o3d.io.read_point_cloud(os.path.join(path,"Depth Long Throw",file))
            pcd_down = pcd.voxel_down_sample(voxel_size = voxel_size)
            pcds = pcds + pcd_down

        pcds = pcds.voxel_down_sample(voxel_size=voxel_size)


        _points = np.array(pcds.points)
        #_points[:0] = -_points[:0]
        #_points[:1] = -_points[:1]
        #_points[:2] = -_points[:2]

        _colors = np.array(pcds.colors)
        _colors = (_colors*255.).astype(np.uint8)

        storePly(ply_path,_points,_colors,np.array(pcds.normals))
        pcd = BasicPointCloud(_points,np.array(pcds.colors),np.array(pcds.normals))
    else: 
        pcd = fetchPly(ply_path)


    camera_infos = []

    stem = os.path.basename(path)
    pv_info_file = [f for f in os.listdir(path) if fnmatch.fnmatch(f,r'*_pv.txt')][0]
    pv_info_file = os.path.join(path,pv_info_file)
    pack = readHoloLensPVCameraIntrinsics(pv_info_file)
    ox,oy,cx,cy,fx,fy =pack[0],pack[1],pack[2],pack[3],pack[4],pack[5]

    #fovx = focal2fov(focal_x,fx)
    #fovy = focal2fov(focal_y,fy)

    for id, focal, mat in readHoloLensPVCameraToWorldTransform(pv_info_file):

        c2w = mat
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # T[1:3]*=-1
        # ?

        fovx = focal2fov(focal[0],cx)
        fovy = focal2fov(focal[1],cy)
        try:
            img_path = os.path.join(path,"PV",id+'.png')
            img = Image.open(img_path)
            camera_info = CameraInfo(id,R,T,fovy,fovx,img,img,img_path,id+'.png',None,None,cx,cy)
            camera_infos.append(camera_info)
        except:
            print(f'ignore image: {id}')
            continue

    if eval:
        train_cam_infos = [c for idx, c in enumerate(camera_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(camera_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = camera_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(pcd,train_cam_infos, test_cam_infos,nerf_normalization, ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "HoloLens": readHoloLensSceneInfo
}


# def __create_random_ply(ply_path):
#     # Since this data set has no colmap data, we start with random points
#         num_pts = 100_000
#         print(f"Generating random point cloud ({num_pts})...")
        
#         # We create random points inside the bounds of the synthetic Blender scenes
#         xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
#         shs = np.random.random((num_pts, 3)) / 255.0
#         # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
#         shs = SH2RGB(shs) * 255
#         if ply_path is not None:
#             storePly(ply_path, xyz, shs)

#         return xyz,shs

def readHoloLensDepth(path):
    import fnmatch
    import cv2
    for file in [x for x in os.listdir(path) if fnmatch.fnmatch(x,'*_proj.png')]:
        image_path = os.path.join(path, file)
        img = cv2.imread(image_path)
        mask = np.where(img!=0)
        yield img, mask



# 读取可定位相机的RGB图像（hint，RGB图像的数量远多于深度图的数量）
def readHoloLensPV(path):
    import fnmatch
    for file in [x for x in os.listdir(path) if fnmatch.fnmatch(x,'*.png')]:
        image_path = os.path.join(path, file)
        # image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        yield os.path.basename(file), image

# 从深度相机的rig2world读取对应图片的变换矩阵(深度图)
def readHoloLensDepthTransform(file):
    import csv
    with open(file,'r') as f:
        lines = csv.reader(f)
        for line in lines:
            index = line[0]
            mat = [float(str_value) for str_value in line[1:]]
            mat = np.asarray(mat)
            mat = mat.reshape(4,4)
            
            yield index, mat


# def readHoloLensDepthCameraExtrinsics(file):
#     with open(file) as f:
#         text = f.readline()
#         data = [int(i) for i in text.split(',')]
#         data = np.asarray(data).reshape(4,4)
#     return data


def readHoloLensDepthCameraIntrinsics(file):
    # HoloLens不直接提供相机内参，而是通过查找表LUT转换得到
    def read_lut_file(filepath):
        with open(filepath, mode='rb') as depth_file:
            lut = np.frombuffer(depth_file.read(),dtype='f')
            lut = np.reshape(lut,(-1,3))
        return lut
    
    def fit_linearity(x_array, y_array):
        m = len(x_array)
        sum_x = np.sum(x_array)
        sum_y = np.sum(y_array)
        sum_xy = np.sum(x_array*y_array)
        sum_xx = np.sum(x_array**2)
        b = (sum_y*sum_xx-sum_x*sum_xy)/(m*sum_xx-sum_x**2)
        k = (m*sum_xy-sum_x*sum_y)/(m*sum_xx-sum_xx**2)
        return k, b
    
    def fit_camera_intrinsics(lut, h,w):
        where = np.where(lut[:,2]!=0)
        lut = lut[where]
        xc = lut[:,0]/lut[:,2]
        yc = lut[:,1]/lut[:,2]
        u=np.arange(0.5,w,1,float)
        u = np.tile(u,h)
        u = u[where]
        v = np.arange(0.5,h,1,float)
        v = v.repeat(w)
        v = v[where]
        fx, x0 = fit_linearity(xc,u)
        fy, y0 = fit_linearity(yc,v)

        intrinsics = np.array([[fx,0,x0],[0,fy,y0],[0,0,1]])
        return intrinsics
    
    lut = read_lut_file(file)
    intrinsics = fit_camera_intrinsics(lut, 428, 760)
    return intrinsics

# 读取PV相机的内参
# 依次是：主点x、主点y；图像宽、高；焦距
def readHoloLensPVCameraIntrinsics(path):
    with open(path) as f:
        txt = f.readline()
        return [float(x) for x in txt.split(',')]
    
def readHoloLensPVCameraToWorldTransform(path):
    with open(path) as f:
        s = f.readlines()

        for _s in s[1:] :
            if _s.isspace():
                continue
            txt = _s.split(',')
            index = txt[0]
            focal = (float(txt[1]),float(txt[2]))
            mat = np.asarray([float(x) for x in txt[3:]]).reshape(4,4)
            # mat[:3, 1:3] *= -1
            ## 转换了相机坐标系（但是不确定效果？）
            yield index, focal ,mat
            

def connection_depth_with_rgb_files(path):
    import re
    pattern = re.compile(R'(\d{18,}) rgb\\(\d{18,})_proj.png')

    #with open(os.path.join(path,'depth.txt')) as f_depth:
    with open(os.path.join(path,'rgb.txt')) as f_rgb:
        rgbs = f_rgb.read()
        result = pattern.findall(rgbs)
        depth_index = [x[0] for x in result]
        pv_index = [x[1] for x in result]

        return depth_index, pv_index
    
def generatePointCloud(depth):
    xyz = None
    rgb = None
    return xyz, rgb
    



   