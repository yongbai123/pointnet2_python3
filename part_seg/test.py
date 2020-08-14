import datetime
import argparse
import importlib
import os
import sys
import tensorflow as tf
import numpy as np
from matplotlib import cm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import part_dataset_all_normal as part_dataset
import show3d_balls
output_dir = os.path.join(BASE_DIR, './test_results')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--category', default='Airplane', help='Which single class to train on [default: Airplane]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
FLAGS = parser.parse_args()


MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model) # import network module
NUM_CLASSES = 50
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
TEST_DATASET = part_dataset.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test', return_cls_label=True)

def print_log(msg, stream=None):
    formatted_msg = "[%s] %s" % (str(datetime.datetime.now()), msg)
    print(formatted_msg)
    if stream is not None:
        stream.write(formatted_msg)


def output_color_point_cloud(data, seg, out_file, color_map):
    with open(out_file, 'w') as f:
        for i in range(len(seg)):
            color = color_map(seg[i])
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def get_model(batch_size, num_point):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl)#, end_points)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}
        return sess, ops

def inference(sess, ops, pc, batch_size):
    ''' pc: BxNx3 array, return BxN pred '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]//batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    for i in range(num_batches):
        feed_dict = {ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
                     ops['is_training_pl']: False}
        batch_logits = sess.run(ops['pred'], feed_dict=feed_dict)
        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
    return np.argmax(logits, 2)

if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    total = 0
    total_acc = 0
    color_map = cm.get_cmap('viridis', NUM_CLASSES)
    total_seen = 0
    SIZE = len(TEST_DATASET)
    total_acc_iou = 0.0

    classes = TEST_DATASET.classes
    total_per_cat_iou = {cat:0 for cat in classes.keys()}
    total_per_cat_acc = {cat:0 for cat in classes.keys()}
    total_per_cat_seen = {cat:0 for cat in classes.keys()}

    for i in range(SIZE):
        print_log(">>>> running sample " + str(i) + "/" + str(SIZE))

        ps, normal, seg, current_cls = TEST_DATASET[i]
        ps = np.hstack((ps, normal))
        sess, ops = get_model(batch_size=1, num_point=ps.shape[0])
        segp = inference(sess, ops, np.expand_dims(ps, 0), batch_size=1)
        segp = segp.squeeze()
        
        total += segp.shape[0]
        total_acc += np.mean(seg == segp)
        total_seen +=1 

        mask = np.int32(seg == segp)

        total_iou = 0.0

        seg_classes = TEST_DATASET.seg_classes
        current_cls_name = list(classes)[current_cls[0]]
        iou_oids = seg_classes[current_cls_name] 
        

        for oid in iou_oids:
            n_pred = np.sum(segp == oid)
            n_gt = np.sum(seg == oid)
            n_intersect = np.sum(np.int32(seg == oid) * mask)
            n_union = n_pred + n_gt - n_intersect
            if n_union == 0:
                total_iou += 1
            else:
                total_iou += n_intersect * 1.0 / n_union

        avg_iou = total_iou / len(iou_oids)

        total_per_cat_iou[current_cls_name] += avg_iou 
        total_per_cat_acc[current_cls_name] += np.mean(mask)
        total_per_cat_seen[current_cls_name] += 1
        total_acc_iou += avg_iou
        
        print_log("IoU: %f" % (total_acc_iou / total_seen))
        print_log("Accuracy: %f" % (total_acc / total_seen))
        print_log("Current class: %s" % (current_cls_name))
        print_log("Current class avg: %f" % (total_per_cat_acc[current_cls_name] / total_per_cat_seen[current_cls_name]))
        print_log("Current class iou: %f" % (total_per_cat_iou[current_cls_name] / total_per_cat_seen[current_cls_name]))

        output_color_point_cloud(ps, seg, './test_results/gt_%d.obj' % (i), color_map)
        output_color_point_cloud(ps, segp, './test_results/pred_%d.obj' % (i), color_map)
        output_color_point_cloud(ps, segp == seg, './test_results/diff_%d.obj' % (i), lambda eq: (0, 1, 0) if eq else (1, 0, 0))

    accuracy = total_acc / total_seen
    iou = total_acc_iou / total_seen
    with open('./test_results/metrics.txt', 'w') as f:
        print_log("Accuracy: %f \n" % accuracy, stream=f)
        print_log("IoU: %f \n" % iou, stream=f)
        for c in total_per_cat_iou:
            acc = 0
            iou = 0
            if total_per_cat_seen[c] != 0:
                acc = total_per_cat_acc[c] / total_per_cat_seen[c]
                iou = total_per_cat_iou[c] / total_per_cat_seen[c]
            print_log("%s avg: %f\n" % (c, acc), stream=f)
            print_log("%s iou: %f\n" % (c, iou), stream=f)
