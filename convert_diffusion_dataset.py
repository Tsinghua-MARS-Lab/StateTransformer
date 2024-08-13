# This is used to train the Diffusion Key Point Decoder separately.
# Need to generate the training set and testing set for DiffusionKeyPointDecoder using runner.py
    #   by setting generate_diffusion_dataset_for_key_points_decoder = True and specify diffusion_feature_save_dir first.
# After training the Diffusion Key Point Decoder using this file, one can use the trained Diffusion Key Point Decoder as key_point_decoder for 
    # Model type TrajectoryGPTDiffusionKPDecoder by setting the model-name to be pretrain/scratch-diffusion_KP_decoder_gpt 
                                                    # and set the key_points_diffusion_decoder_load_from to be the best_model.pth file that is generated and saved by this runner_diffusionKPdecoder.py program.
import torch
import os
import argparse

def map_pdm_dataset(args):
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from transformer4planning.preprocess.pdm_vectorize import pdm_vectorize
    from datasets import Dataset
    from runner import load_dataset
    map_api = dict()
    for map in ['sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']:
        map_api[map] = get_maps_api(map_root=args.map_dir,
                            map_version="nuplan-maps-v1.0",
                            map_name=map
                            )
    datapath = args.saved_dataset_folder
    def map_func(sample):
        return pdm_vectorize(sample, datapath, map_api, False)
    dataset = load_dataset(os.path.join(datapath, "index"), args.split)
    dataset = dataset.map(map_func, num_proc=args.num_proc)
    Dataset.save_to_disk(dataset, os.path.join(args.savedir, args.dataset_name, args.split))
    print("done")

def yield_diffusion_dataset(shards, root):
    for shard in shards:
        label_name = "future_key_points_"
        hidden_state_name = "future_key_points_hidden_state_"
        if not os.path.exists(os.path.join(root, label_name + str(shard) + ".pth")):
            print(f'{os.path.join(root, label_name + str(shard) + ".pth")} does not exist!')
            continue
        if not os.path.exists(os.path.join(root, hidden_state_name + str(shard) + ".pth")):
            print(f'{os.path.join(root, hidden_state_name + str(shard) + ".pth")} does not exist!')
            continue
        label_in_batch = torch.load(os.path.join(root, label_name + str(shard) + ".pth"))
        hidden_state_in_batch = torch.load(os.path.join(root, hidden_state_name + str(shard) + ".pth"))
        assert label_in_batch.shape[0] == hidden_state_in_batch.shape[0]
        for i in range(label_in_batch.shape[0]):
            item = dict(
                label = label_in_batch[i],
                hidden_state = hidden_state_in_batch[i]
            )
            yield item

def generate_arrow_dataset(args):
    from datasets import Dataset
    items = os.listdir(args.data_dir)
    shards = set()
    for item in items:
        if item.endswith(".pth"):
            shards.add(int(item.split("_")[-1].split(".")[0]))
    print(len(shards))
    assert len(shards) == len(items) // 2
    print(f"generating dataset from {args.data_dir}, with {len(shards)} items.")
    shards = list(shards)
    dataset = Dataset.from_generator(
        yield_diffusion_dataset,
        gen_kwargs={"shards": shards, "root": args.data_dir},
        writer_batch_size=10,
        cache_dir=args.save_dir,
        num_proc=args.num_proc,
    )
    dataset.save_to_disk(os.path.join(args.save_dir, args.dataset_name))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument("--save_dir", 
                        type=str, 
                        default=None, 
                        help="the dir to save the generated arrow dataset")
    parser.add_argument("--data_dir",
                        type=str,
                        default=None,
                        help="original hidden state dataset path, usually stored in pth format")
    parser.add_argument("--num_proc", 
                        type=int, 
                        default=1,
                        help="number of processes to use in generation")
    parser.add_argument("--dataset_name",
                        type=str,
                        default="train",
                        help="the name of the generated dataset")
    parser.add_argument("--map_dir",
                        type=str,
                        default="/public/MARS/datasets/nuPlan/nuplan-maps-v1.1"
                        )
    parser.add_argument("--saved_dataset_folder",
                        type=str,
                        default="/localdata_ssd/nuplan/online_float32_opt",
                        help="default online dataset, same as that used in runner.py")
    parser.add_argument("--split",
                        type=str,
                        default="train")
    args = parser.parse_args()
    # map_pdm_dataset(args)
    generate_arrow_dataset(args)
    # main()
