import os
from glob import glob

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from jpeg_eigen import jpeg_recompress_pil, jpeg_feature


def main():
    # Parameters ---
    ps_root = 'data/ps/'
    raw_root = 'data/raw/'
    pil_root = 'data/pil/'
    ps_features_path = 'data/ps.npy'
    pil_features_path = 'data/pil.npy'

    raw_file_list = glob(raw_root + '*.png')
    ps_file_list = [os.path.join(ps_root, os.path.split(os.path.splitext(file_path)[0])[1]) + '.jpg' for file_path in
                    raw_file_list]
    pil_file_list = [os.path.join(pil_root, os.path.split(os.path.splitext(file_path)[0])[1]) + '.jpg' for file_path in
                     raw_file_list]

    print('Compressing RAW to PIL with PS quantization matrix')
    for raw_file_path, ps_file_path, pil_file_path in tqdm(zip(raw_file_list, ps_file_list, pil_file_list)):
        if not os.path.exists(pil_file_path):
            img_ps = Image.open(ps_file_path)
            qtables_in = img_ps.quantization
            jpeg_recompress_pil(raw_file_path, pil_file_path, qtables_in=qtables_in, )

    if not os.path.exists(ps_features_path):
        print('Extracting features for PS images')
        features_ps = []
        for ps_file_path in tqdm(ps_file_list):
            features_ps += [jpeg_feature(ps_file_path)]
        features_ps = np.stack(features_ps)
        np.save(ps_features_path, features_ps)
    else:
        print('Loading features for PS images')
        features_ps = np.load(ps_features_path)

    if not os.path.exists(pil_features_path):
        print('Extracting features for PIL images')
        features_pil = []
        for pil_file_path in tqdm(pil_file_list):
            features_pil += [jpeg_feature(pil_file_path)]
        features_pil = np.stack(features_pil)
        np.save(pil_features_path, features_pil)
    else:
        print('Loading features for PIL images')
        features_pil = np.load(pil_features_path)

    np.random.seed(197)
    rand_idxs = np.random.permutation(np.arange(len(raw_file_list)))
    train_idxs = rand_idxs[:len(raw_file_list) // 2]
    test_idxs = rand_idxs[len(raw_file_list) // 2:]

    features_train = np.concatenate((features_ps[train_idxs], features_pil[train_idxs]), axis=0)
    labels_train = np.concatenate((np.zeros(len(train_idxs)), np.ones(len(train_idxs))))

    features_test = np.concatenate((features_ps[test_idxs], features_pil[test_idxs]), axis=0)
    labels_test = np.concatenate((np.zeros(len(test_idxs)), np.ones(len(test_idxs))))

    clf = RandomForestClassifier()
    clf.fit(features_train, labels_train)

    pred_test = clf.predict_proba(features_test)[:, 1]

    auc_score = roc_auc_score(labels_test, pred_test)
    print('Test AUC: {:.2f}'.format(auc_score))


if __name__ == '__main__':
    main()
