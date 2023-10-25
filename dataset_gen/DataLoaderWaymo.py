import os
import glob
import tensorflow as tf

class WaymoDL:
    def __init__(self, data_path=None, mode="train"):
        
        assert data_path is not None and mode in ["train", "val", "test"]

        self.data_root = data_path["WAYMO_DATA_ROOT"]
        self.split_dir = data_path["SPLIT_DIR"][mode]
       
        self.data_path = os.path.join(self.data_root, self.split_dir)
        src_files = glob.glob(os.path.join(self.data_path, '*.tfrecord*'))
        src_files.sort()
        
        self.global_file_names = src_files
        self.total_file_num = len(self.global_file_names)

    def get_next_file(self, specify_file_index=None):
        if specify_file_index is None:
            self.current_file_index += 1
            file_index = self.current_file_index
        else:
            # for non-sequential parallel loading
            file_index = specify_file_index
        if not file_index < self.total_file_num:
            print('index exceed total file number', file_index, self.total_file_num)
            self.end = True
            return None

        if os.path.getsize(self.global_file_names[file_index]) > 0:
            data = tf.data.TFRecordDataset(self.global_file_names[file_index], compression_type='')
        else:
            print('empty file', self.global_file_names[file_index])
            return None
        
        return data, self.global_file_names[file_index].split("/")[-1]