import tensorflow as tf
import time
from nets import TrajectoryCNN
import torch.nn as nn
import numpy as np
import torch
from data_provider import datasets_factory_joints_cmu as datasets_factory
import os
from utils import recovercmu_3d as recovercmu_3d
from utils import metrics
import scipy.io as io

# data path
dataset_name='skeleton'
train_data_paths ='cmu_ske/train_cmu_35.npy'
valid_data_paths ='cmu_ske/train_cmu_35.npy'
test_data_paths ='cmu_ske/testset'
save_dir ='checkpoints/cmu/traj_cmu_long_term/v3'
gen_dir ='results/cmu/traj_cmu_long_term/v3'
bak_dir ='backup/cmu/traj_cmu_long_term'
# model parameter
pretrained_model=''
input_length= 10
seq_length =35
joints_number=25
joint_dims=3
stacklength=4
filter_size=3
# opt
lr=0.0001
batch_size=16
max_iterations=3000000
display_interval=10
test_interval=500
snapshot_interval=500
num_save_samples=100000
num_hidden=[64,64,64,64,64]
keep_prob=0.75
print('!!! TrajectoryCNN:', num_hidden)

if not  tf.io.gfile.exists(save_dir):
    tf.io.gfile.makedirs(save_dir)
if not  tf.io.gfile.exists(gen_dir):
    tf.io.gfile.makedirs(gen_dir)
if not tf.io.gfile.exists(bak_dir):
    tf.io.gfile.makedirs(bak_dir)
print ('start training !',time.strftime('%Y-%m-%d %H:%M:%S\n\n\n',time.localtime(time.time())))

#Training
class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc,self).__init__()
    def forward(self,output,target):
        return torch.mean(torch.norm(target-output,p=2,dim=3,keepdim=True))
model =TrajectoryCNN(keep_prob,seq_length,input_length,stacklength,num_hidden,filter_size).cuda()
model =torch.load(bak_dir+'/model.pth')
loss =LossFunc()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
train_time=0
test_time_all =0
err_list = []
min_err = 66.10117324696071
# load data
train_input_handle, test_input_handle = datasets_factory.data_provider(
    dataset_name, train_data_paths, valid_data_paths,
    batch_size ,joints_number, input_length, seq_length, is_training=True)

for itr in range(1,max_iterations+1):
    if train_input_handle.no_batch_left():
        train_input_handle.begin(do_shuffle=True)

    if itr <= 751500:
        train_input_handle.next()
        print('itr: ' + str(itr) + 'passed!')
        continue
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    start_time = time.time()

    model.train()
    ims = train_input_handle.get_batch()
    ims = ims[:, :, 0:joints_number, :]
    pretrain_iter = 0
    if itr < pretrain_iter:
        inputs1 = ims
    else:
        inputs1=ims[:,0:input_length,:,:]
        tem=ims[:,input_length-1]
        tem=np.expand_dims(tem,axis=1)
        tem=np.repeat(tem,seq_length - input_length,axis=1)
        inputs1=np.concatenate((inputs1,tem),axis=1)
    inputs2 = ims[:, input_length:]
    inputs = np.concatenate((inputs1, inputs2), axis=1)
    inputs =torch.FloatTensor(inputs).cuda()
    train_pred = model(inputs)
    batch_loss =loss(train_pred,inputs[:,seq_length:])

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    train_loss += batch_loss.item()

    # inverse the input sequence
    imv1 = ims[:, ::-1]
    if itr >= pretrain_iter:
        imv_rev1 = imv1[:, 0:input_length, :, :]
        tem = imv1[:, input_length - 1]
        tem = np.expand_dims(tem, axis=1)
        tem = np.repeat(tem, seq_length - input_length, axis=1)
        imv_rev1 = np.concatenate((imv_rev1, tem), axis=1)
    else:
        imv_rev1 = imv1
    imv_rev2 = imv1[:, input_length:]
    ims_rev1 = np.concatenate((imv_rev1, imv_rev2), axis=1)
    ims_rev1 =torch.FloatTensor(ims_rev1).cuda()
    train_pred = model(ims_rev1)
    batch_loss =loss(train_pred,ims_rev1[:,seq_length:])

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    train_loss += batch_loss.item()
    train_loss =train_loss/2

    end_time=time.time()
    t=end_time-start_time
    train_time +=t

    if itr % display_interval==0:
        print('itr: ' + str(itr) + ' lr: ' + str(lr) + ' training loss: ' + str(train_loss))
    if itr % test_interval==0:
        model.eval()
        print('train time:' + str(train_time))
        print('test...')
        str1 = 'basketball', 'basketball_signal', 'directing_traffic', 'jumping', 'running', 'soccer', 'walking', 'washwindow'
        res_path=os.path.join(gen_dir,str(itr))
        if not tf.io.gfile.exists(res_path):
            tf.io.gfile.makedirs(res_path)
        avg_mse = 0
        batch_id = 0
        test_time = 0
        joint_mse = np.zeros((25, 38))
        joint_mae = np.zeros((25, 38))
        mpjpe = np.zeros([1, seq_length - input_length])
        mpjpe_l = np.zeros([1, seq_length - input_length])
        img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []
        for i in range(seq_length - input_length):
            img_mse.append(0)
            fmae.append(0)
        f = 0
        for s in str1:
            start_time1 = time.time()
            batch_id = batch_id + 1
            mpjpe1 = np.zeros([1, seq_length - input_length])
            tem = np.load(test_data_paths + '/test_cmu_' + str(seq_length) + '_' + s + '.npy')
            tem = np.repeat(tem, (batch_size) / 8, axis=0)
            test_ims = tem[:, 0:seq_length, :, :]
            test_ims1 = test_ims
            test_ims = test_ims[:, :, 0:joints_number, :]

            test_dat = test_ims[:, 0:input_length, :, :]
            tem = test_dat[:, input_length - 1]
            tem = np.expand_dims(tem, axis=1)
            tem = np.repeat(tem, seq_length - input_length, axis=1)
            test_dat1 = np.concatenate((test_dat, tem), axis=1)
            test_dat2 = test_ims[:, input_length:]
            test_dat = np.concatenate((test_dat1, test_dat2), axis=1)
            test_dat=torch.FloatTensor(test_dat).cuda()
            img_gen = model(test_dat)
            end_time1 =time.time()
            t1 =end_time1-start_time1
            test_time+=t1
            img_gen=img_gen.cpu().detach().numpy()
            gt_frm=test_ims1[:,input_length:]
            img_gen= recovercmu_3d.recovercmu_3d(gt_frm, img_gen)
            for i in range(seq_length - input_length):
                x = gt_frm[:, i, :, ]
                gx = img_gen[:, i, :, ]
                fmae[i] += metrics.batch_mae_frame_float(gx, x)
                mse = np.square(x - gx).sum()
                for j in range(batch_size ):
                    tem1 = 0
                    for k in range(gt_frm.shape[2]):
                        tem1 += np.sqrt(np.square(x[j, k] - gx[j, k]).sum())
                    mpjpe1[0, i] += tem1 / gt_frm.shape[2]
                img_mse[i] += mse
                avg_mse += mse
                real_frm = x
                pred_frm = gx
                for j in range(gt_frm.shape[2]):
                    xi = x[:, j]
                    gxi = gx[:, j]
                    joint_mse[i, j] += np.square(xi - gxi).sum()
                    joint_mae[i, j] += metrics.batch_mae_frame_float1(gxi, xi)

            mpjpe1 = mpjpe1 / (batch_size )
            print('current action mpjpe: ', s)
            for i in mpjpe1[0]:
                print(i)
            mpjpe += mpjpe1
            if f <= 3:
                print('four actions', s)
                mpjpe_l += mpjpe1
            f = f + 1
        test_time_all += test_time
        joint_mae = np.asarray(joint_mae, dtype=np.float32) / batch_id
        joint_mse = np.asarray(joint_mse, dtype=np.float32) / (batch_id * batch_size )
        avg_mse = avg_mse / (batch_id * batch_size )
        print('mse per seq: ' + str(avg_mse))
        mpjpe = mpjpe / (batch_id)
        err_list.append(np.mean(mpjpe))
        print('mean per joints position error: ' + str(np.mean(mpjpe)))
        for i in range(seq_length - input_length):
            print(mpjpe[0, i])
        mpjpe_l = mpjpe_l / 4
        print('mean mpjpe for four actions: ' + str(np.mean(mpjpe_l)))
        for i in range(seq_length - input_length):
            print(mpjpe_l[0, i])
        fmae = np.asarray(fmae, dtype=np.float32) / batch_id
        print('fmae per frame: ' + str(np.mean(fmae)))
        print('current test time:' + str(test_time))
        print('all test time: ' + str(test_time_all))
    if itr % snapshot_interval == 0 and min(err_list) < min_err:
        checkpoint_path = os.path.join(save_dir, 'model.pth')
        torch.save(model,checkpoint_path)
        min_err = min(err_list)
        print('model saving done! ', time.strftime('%Y-%m-%d %H:%M:%S\n\n\n', time.localtime(time.time())))
        with open(save_dir+'/save_itr.txt', 'w') as f:
            f.write(str(itr))
        print('current minimize error is: ', min_err)
    if itr % snapshot_interval == 0:
        with open(bak_dir+'/save_itr.txt', 'w') as f:
            f.write(str(itr))
        bak_path = os.path.join(bak_dir, 'model.pth')
        torch.save(model,bak_path)

    train_input_handle.next()