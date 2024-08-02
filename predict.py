import os
from re import X
import numpy as np
import torch
import SimpleITK as sitk
from timm.models import create_model
from dataset import get_dataloader, CarotidDataset, CarotidTestDataset
from main import get_argparse
from net.unet import Unet3D
from typing import Union, Tuple, List
from scipy.ndimage.filters import gaussian_filter

def maybe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit
    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).
    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


class Predictor():
    def __init__(self, model):

        self.input_shape_must_be_divisible_by = None 
        self.net = model

        self.conv_op = None  # nn.Conv2d or nn.Conv3d
        self.num_classes = 2  # number of channels in the output

        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None


    def __call__(self, model, dataloader, datapath, savepath):
        torch.cuda.empty_cache()

        maybe_mkdir(datapath)
        maybe_mkdir(savepath)
        for item in dataloader:
            
            sub_i = dataloader.dataset.pathlist[item['Id']].split("/")[-1]
            print(sub_i, " processing...")

            data = item['image'].squeeze(axis=0).cuda()
            
            patch_size = (128, 112, 128)

            steps = self._compute_steps_for_sliding_window(patch_size, tuple(data.shape[1:]), 0.5)
            
            steps = [step if step != [] else [0] for step in steps ]
            
            segs, prob = self._get_segmap(data, patch_size, steps)
            sub_name = sub_i.split("_")[0]

            roi_img = sitk.GetImageFromArray(segs)
            sitk.WriteImage(roi_img, f"{savepath}/{sub_i}.nii.gz")
            prob = prob.astype(np.float16)
            np.savez_compressed(f"{savepath}/{sub_name}", prob=prob)


    def get_device(self):
        if next(self.net.parameters()).device.type == "cpu":
            return "cpu"
        else:
            return next(self.net.parameters()).device.index


    def _get_segmap(self, data, patch_size, steps):

        data, slicer = pad_nd_image(data.cpu().numpy(), patch_size, "constant", None, True, None)

        mirror_axes = (0, 1, 2)
    
        gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)
        gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
        gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)
        gaussian_importance_map = gaussian_importance_map.cpu().numpy()

        add_for_nb_of_preds = gaussian_importance_map

        aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, True,
                        gaussian_importance_map)[0]

                    predicted_patch = predicted_patch.detach().cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
            range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])

        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        region_lu  = class_probabilities[0]
        region_ow  = class_probabilities[1]
        predicted_segmentation = np.zeros(class_probabilities.shape[1:])

        predicted_segmentation[region_ow > 0.5] = 1
        predicted_segmentation[region_lu > 0.5] = 2


        return predicted_segmentation, class_probabilities


    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                            do_mirroring: bool = True,
                                            mult: np.ndarray or torch.tensor = None) -> torch.tensor:
            assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

            # if cuda available:
            #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
            #   we now return a cuda tensor! Not numpy array!

            x = maybe_to_torch(x)
            result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                    dtype=torch.float)

            if torch.cuda.is_available():
                x = to_cuda(x, gpu_id=self.get_device())
                result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

            if mult is not None:
                mult = maybe_to_torch(mult)
                if torch.cuda.is_available():
                    mult = to_cuda(mult, gpu_id=self.get_device())

            if do_mirroring:
                mirror_idx = 8
                num_results = 2 ** len(mirror_axes)
            else:
                mirror_idx = 1
                num_results = 1

            for m in range(mirror_idx):

                if m == 0:
                    pred = self.inference_apply_nonlin(self.net(x))
                    result_torch += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self.net(torch.flip(x, (4, ))))
                    result_torch += 1 / num_results * torch.flip(pred, (4,))

                if m == 2 and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self.net(torch.flip(x, (3, ))))
                    result_torch += 1 / num_results * torch.flip(pred, (3,))

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self.net(torch.flip(x, (4, 3))))
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3))

                if m == 4 and (0 in mirror_axes):
                    pred = self.inference_apply_nonlin(self.net(torch.flip(x, (2, ))))
                    result_torch += 1 / num_results * torch.flip(pred, (2,))

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self.net(torch.flip(x, (4, 2))))
                    result_torch += 1 / num_results * torch.flip(pred, (4, 2))

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self.net(torch.flip(x, (3, 2))))
                    result_torch += 1 / num_results * torch.flip(pred, (3, 2))
                
                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self.net(torch.flip(x, (4, 3, 2))))
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

                    pred = 0

            if mult is not None:
                result_torch[:, :] *= mult

            return result_torch


    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map


    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps


def maybe_mkdirs(args):
    results_path = args.pred_path
    dataset = args.test_path.split("/")[-1]
    path_ = results_path

    if args.nb_classes == 2: classes = "01.2class"
    else: classes = "02.4class"
    fold = f"fold{args.fold}"
    sublist = [classes, dataset, args.model, args.mode + "_" + args.name, fold]
    for p in sublist:
        path_ = f"{path_}{p}/"
        maybe_mkdir(path_)

    return path_



if __name__ == "__main__":

    args = get_argparse()

    model = create_model(
        args.model,
        pretrained=False,
        out_channels=2).to('cuda')

    pred_path = maybe_mkdirs(args)
    print(pred_path)

    ## load model weight
    model.load_state_dict(torch.load(args.state_path + f"/{args.model}_{args.mode}-{args.name}_fold{args.fold}/best_model.pth"))
    test_dataloader = get_dataloader(CarotidTestDataset, path=args.test_path, phase="test")

    predictor = Predictor(model)
    predictor(model, test_dataloader, args.test_path, pred_path)
