
import os
import sys
import cv2
import logging
import numpy as np

def logger_init():

    logger = logging.getLogger("PedestranDetect")

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')


    # file_handler = logging.FileHandler("test.log")
    # file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter  # 也可以直接给formatter赋值

    # 为logger添加的日志处理器
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(logging.INFO)

    return logger

def load_data_set(logger):
    logger.info('Checking data path!')
    pwd = os.getcwd()
    logger.info('Current path is:{}'.format(pwd))

    # 提取正样本
    pos_dir = os.path.join(pwd, 'Positive')
    if os.path.exists(pos_dir):
        logger.info('Positive data path is:{}'.format(pos_dir))
        pos = os.listdir(pos_dir)
        logger.info('Positive samples number:{}'.format(len(pos)))

    # 提取负样本
    neg_dir = os.path.join(pwd, 'Negative')
    if os.path.exists(neg_dir):
        logger.info('Negative data path is:{}'.format(neg_dir))
        neg = os.listdir(neg_dir)
        logger.info('Negative samples number:{}'.format(len(neg)))

    # 提取测试集
    test_dir = os.path.join(pwd, 'TestData')
    if os.path.exists(test_dir):
        logger.info('Test data path is:{}'.format(test_dir))
        test = os.listdir(test_dir)
        logger.info('Test samples number:{}'.format(len(test)))

    return pos, neg, test

def load_train_samples(pos, neg):
    pwd = os.getcwd()
    pos_dir = os.path.join(pwd, 'Positive')
    neg_dir = os.path.join(pwd, 'Negative')

    samples = []
    labels = []
    for f in pos:
        file_path = os.path.join(pos_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(1.)

    for f in neg:
        file_path = os.path.join(neg_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(-1.)

    # labels 要转换成numpy数组，类型为np.int32
    labels = np.int32(labels)
    labels_len = len(pos) + len(neg)
    labels = np.resize(labels, (labels_len, 1))

    return samples, labels

def extract_hog(samples, logger):

    train = []
    logger.info('Extracting HOG Descriptors...')
    num = 0.
    total = len(samples)
    for f in samples:
        num += 1.
        logger.info('Processing {} {:2.1f}%'.format(f, num/total*100))
        hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
        # hog = cv2.HOGDescriptor()
        img = cv2.imread(f, -1)
        img = cv2.resize(img, (64,128))
        descriptors = hog.compute(img)
        logger.info('hog feature descriptor size: {}'.format(descriptors.shape))    # (3780, 1)
        train.append(descriptors)

    train = np.float32(train)
    train = np.resize(train, (total, 3780))

    return train

def get_svm_detector(svm):

    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)

def train_svm(train, labels, logger):

    logger.info('Configuring SVM classifier.')
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)

    logger.info('Starting training svm.')
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    logger.info('Training done.')

    pwd = os.getcwd()
    model_path = os.path.join(pwd, 'svm.xml')
    svm.save(model_path)
    logger.info('Trained SVM classifier is saved as: {}'.format(model_path))

    return get_svm_detector(svm)

def test_hog_detect(test, svm_detector, logger):

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    pwd = os.getcwd()
    test_dir = os.path.join(pwd, 'TestData')
    cv2.namedWindow('Detect')
    for f in test:
        file_path = os.path.join(test_dir, f)
        logger.info('Processing {}'.format(file_path))
        img = cv2.imread(file_path)
        rects, _ = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
        for (x,y,w,h) in rects:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.imshow('Detect', img)
        c = cv2.waitKey(0) & 0xff
        if c == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    e1 = cv2.getTickCount()
    logger = logger_init()
    pos, neg, test = load_data_set(logger=logger)
    samples, labels = load_train_samples(pos, neg)
    train = extract_hog(samples, logger=logger)
    logger.info('Size of feature vectors of samples: {}'.format(train.shape))
    logger.info('Size of labels of samples: {}'.format(labels.shape))
    svm_detector = train_svm(train, labels, logger=logger)

    test_hog_detect(test, svm_detector, logger)
    e2 = cv2.getTickCount()  # time
    time2 = (e2 - e1) / cv2.getTickFrequency()
    print('time = ', time2)