import tensorflow as tf
import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
from B_code.evaluation import eva
import time
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Record the running time
tic = time.time()

# Solve the error [RuntimeError: tf.placeholder() is not compatible with eager execution]
tf.compat.v1.disable_eager_execution()

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, \
        reg_constant1 = 1.0, reg_constant2 = 1.0, reg_constant3 = 1.0, \
            batch_size = 200, reg = None, \
                denoise = False, model_path = None, restore_path = None, \
                    logs_path = './models_face/logs'):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0

        # Input required to be fed
        self.x = tf.compat.v1.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, [])

        weights = self._initialize_weights()

        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                               mean = 0,
                                               stddev = 0.2,
                                               dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)

        z = tf.reshape(latent, [batch_size, -1])  
        self.z = z

        # 🟡 C_A
        C_A = weights['C_A']
        Self_A = tf.matmul(C_A,z)
        self.C_A = C_A

        # 🟡 C_S  
        Z_S = 0.5*(C_A + tf.transpose(C_A))
        C_S = weights['C_S']
        Self_S = tf.matmul(C_S, Z_S)
        self.C_S = C_S

        latent_c = latent
        self.x_r = self.decoder(latent_c, weights, shape)      

        self.Coef_F = self.attention_fusion(self.C_A, self.C_S, weights)

        self.sim_mat1 = 0.5*(self.Coef_F + tf.transpose(self.Coef_F))
        self.sim_mat1 = self.sim_mat1 - tf.compat.v1.diag(tf.compat.v1.diag_part(self.sim_mat1))
        
        if tf.compat.v1.count_nonzero( tf.reduce_sum(self.sim_mat1,1) ) == batch_size:
            self.sim_mat2 = tf.divide(self.sim_mat1,tf.reduce_sum(self.sim_mat1,1))
        else:
            self.sim_mat2 = self.sim_mat1
        
        self.sim_mat3 = (self.sim_mat2-tf.compat.v1.diag(tf.compat.v1.diag_part(self.sim_mat2))) + tf.eye(self.batch_size)
        
        self.F_weight = self.sim_mat3**2 / tf.reduce_sum(self.sim_mat3, 0)
        self.F_Aug = tf.transpose(( tf.transpose(self.F_weight)  / tf.reduce_sum(self.F_weight, 1)))

        # 👁‍🗨Reconstruction loss
        self.reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))
        # 👁‍🗨Normalization to C_A 
        self.reg_losses = tf.reduce_sum(tf.pow(self.C_A,2.0))
        # 👁‍🗨Reconstruction for obtaining C_A
        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(Self_A, z), 2.0))
        # 👁‍🗨Normalization to C_S
        self.C_S_normalization = tf.reduce_sum(tf.pow(self.C_S,2.0))
        # 👁‍🗨Reconstruction for obtaining C_S
        self.C_S_Reconstruction = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(Self_S, Z_S), 2.0))

        KL_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
        self.KL_div = KL_loss(self.F_Aug, self.Coef_F)

        # overall losses
        self.loss = self.reconst_cost + 1.0*self.reg_losses +  30*self.selfexpress_losses  \
            + reg_constant1 * self.C_S_normalization + reg_constant2 * self.C_S_Reconstruction \
                + reg_constant3 * self.KL_div

        tf.compat.v1.summary.scalar("recons_loss", self.reconst_cost)
        tf.compat.v1.summary.scalar("reg_loss", self.reg_losses )
        tf.compat.v1.summary.scalar("selfexpress_loss", self.selfexpress_losses )

        self.merged_summary_op = tf.compat.v1.summary.merge_all()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        
        self.init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(self.init)        
        self.saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.trainable_variables() \
            if not (  v.name.startswith("C_A") or v.name.startswith("C_S")  \
                or v.name.startswith("attion_VE")  )])     

        self.saver2 = tf.compat.v1.train.Saver()       
        self.summary_writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())
        
    def _initialize_weights(self):
        all_weights = dict()
        # encoder
        all_weights['enc_w0'] = tf.compat.v1.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
            initializer=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.compat.v1.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))
        all_weights['enc_w1'] = tf.compat.v1.get_variable("enc_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
            initializer=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
        all_weights['enc_b1'] = tf.compat.v1.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))
        all_weights['enc_w2'] = tf.compat.v1.get_variable("enc_w2", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
        all_weights['enc_b2'] = tf.compat.v1.Variable(tf.zeros([self.n_hidden[2]], dtype = tf.float32))        

        # affinity matrix  
        all_weights['C_A']   = tf.compat.v1.Variable(1.0e-5 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'C_A')
        all_weights['C_S']   = tf.compat.v1.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'C_S')
        all_weights['attion_VE']   = tf.compat.v1.Variable(tf.compat.v1.ones([2*self.batch_size, 2]), name = 'attion_VE')
        
        # decoder
        all_weights['dec_w0'] = tf.compat.v1.get_variable("dec_w0", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],self.n_hidden[2]],
            initializer=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
        all_weights['dec_b0'] = tf.compat.v1.Variable(tf.zeros([self.n_hidden[1]], dtype = tf.float32))
        all_weights['dec_w1'] = tf.compat.v1.get_variable("dec_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],self.n_hidden[1]],
            initializer=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
        all_weights['dec_b1'] = tf.compat.v1.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))
        all_weights['dec_w2'] = tf.compat.v1.get_variable("dec_w2", shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
            initializer=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
        all_weights['dec_b2'] = tf.compat.v1.Variable(tf.zeros([1], dtype = tf.float32))
        
        return all_weights

    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        shapes.append(layer1.get_shape().as_list())
        layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, weights['enc_w1'], strides=[1,2,2,1],padding='SAME'),weights['enc_b1'])
        layer2 = tf.nn.relu(layer2)
        shapes.append(layer2.get_shape().as_list())
        layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2, weights['enc_w2'], strides=[1,2,2,1],padding='SAME'),weights['enc_b2'])
        layer3 = tf.nn.relu(layer3)
        return  layer3, shapes

    # Building the decoder
    def decoder(self,z, weights, shapes):
        shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(self.x)[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)
        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['dec_w1'], tf.stack([tf.shape(self.x)[0],shape_de2[1],shape_de2[2],shape_de2[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b1'])
        layer2 = tf.nn.relu(layer2)
        shape_de3= shapes[0]
        layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['dec_w2'], tf.stack([tf.shape(self.x)[0],shape_de3[1],shape_de3[2],shape_de3[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b2'])
        layer3 = tf.nn.relu(layer3)
        return layer3
    
    def attention_fusion(self, C_A, C_S, weights):
        n_x = C_A.get_shape()[0]
        x1 = tf.concat([C_A, C_S], 1)
        p_nml = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(x1, weights['attion_VE'])), axis=1)
        p = tf.math.l2_normalize(p_nml,axis=1)
        p1 = tf.reshape(p[:,0], [n_x, 1])
        p2 = tf.reshape(p[:,1], [n_x, 1])
        p1_broadcast = tf.tile(p1, [1, n_x])
        p2_broadcast = tf.tile(p2, [1, n_x])
        Coef_Fs = tf.multiply(p1_broadcast, C_A) + tf.multiply(p2_broadcast, C_S)
        return Coef_Fs

    def partial_fit(self, X, lr):
        z, cost, summary, _, C_A, C_S, Coef_Fs, KL_div, sim_mat3, F_Aug = self.sess.run((\
            self.z, self.reconst_cost, self.merged_summary_op, self.optimizer, \
                self.C_A, self.C_S, self.Coef_F, self.KL_div, self.sim_mat3, self.F_Aug\
                    ), feed_dict = {self.x: X, self.learning_rate: lr})
        self.iter = self.iter + 1
        return z, cost, C_A, C_S, Coef_Fs, KL_div,sim_mat3, F_Aug
    
    def initlization(self):
        self.sess.run(self.init)
    
    def reconstruct(self,X):
        return self.sess.run(self.x_r, feed_dict = {self.x:X})

    def save_model(self,model_path):
        save_path = self.saver2.save(self.sess,model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")
        
def best_map(L1,L2):
    try:
        Label1 = np.unique(L1)
        nClass1 = len(Label1)
        Label2 = np.unique(L2)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1,nClass2)
        G = np.zeros((nClass,nClass))
        for i in range(nClass1):
            ind_cla1 = L1 == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = L2 == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i,j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:,1]
        newL2 = np.zeros(L2.shape)
        for i in range(0, nClass2):
            newL2[L2 == Label2[i]] = Label1[c[i]]
    except:
        newL2 = np.zeros(L2.shape)
    return newL2 

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        return 0, 0, 0
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]
        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro, f1_micro

def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C
    return Cp

def post_proC(C, K, d, alpha):
    try:
        C = 0.5*(C + C.T)
        r = d*K + 1
        U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
        U = U[:,::-1]    
        S = np.sqrt(S[::-1])
        S = np.diag(S)    
        U = U.dot(S)    
        U = normalize(U, norm='l2', axis = 1)       
        Z = U.dot(U.T)
        Z = Z * (Z>0)    
        L = np.abs(Z ** alpha) 
        L = L/L.max()   
        L = 0.5 * (L + L.T)    
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        return grp, L
    except:
        wrong_grep = np.zeros(C.shape[0])
        return wrong_grep, wrong_grep

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate 

def test_face(Img, Label, CAE, num_class, reg1, reg2, reg3, alpha, lr):    

    dataname = 'Our-Umist'
    eprm_state = 'results'

    file_out = open('./R_output/'+dataname+'_'+eprm_state+'.out', 'a')

    print("\nreg1:",reg1, "reg2:",reg2, "reg3:",reg3, "lr:",lr, "alpha:",alpha, file=file_out)

    # Fusion (Fs)
    iters10_ACC_Fs = []
    iters10_NMI_Fs = []
    iters10_PUR_Fs = []
    iters10_ARI_Fs = []
    iters10_RI_Fs = []
    iters10_F1_macro_Fs = []
    iters10_PRC_macro_Fs = []
    iters10_RC_macro_Fs = []
    iters10_F1_micro_Fs = []
    iters10_PRC_micro_Fs = []
    iters10_RC_micro_Fs = []

    for n_iter in range(0, 13):
        print("n_iter:",n_iter)

        ACC_iters_Fs = [0]
        NMI_iters_Fs = [0]
        PUR_iters_Fs = [0]
        ARI_iters_Fs = [0]
        RI_iters_Fs = [0]
        F1_macro_iters_Fs = [0]
        PRC_macro_iters_Fs = [0]
        RC_macro_iters_Fs = [0]
        F1_micro_iters_Fs = [0]
        PRC_micro_iters_Fs = [0]
        RC_micro_iters_Fs = [0]

        for i in range(0,21-num_class):
            face_10_subjs = np.array(Img[24*i:24*(i+num_class),:])
            face_10_subjs = face_10_subjs.astype(float)        
            label_10_subjs = np.array(Label[24*i:24*(i+num_class)]) 
            label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
            label_10_subjs = np.squeeze(label_10_subjs)    

            CAE.initlization()        
            CAE.restore() 

            max_step = 800 
            start_step = 550 
            display_step = 10

            epoch = 0
            while epoch < max_step:
                epoch = epoch + 1           
                _, _, C_A, C_S, Coef_Fs, _,_,_ = CAE.partial_fit(face_10_subjs, lr)                                  
                if epoch > start_step and epoch % display_step == 0:

                    threshold = thrC(Coef_Fs,alpha)     
 
                    y_x_threshold, _ = post_proC(threshold, label_10_subjs.max(), 10, 8)
                    acc,nmi,pur, ari,ri,  f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro  = \
                        eva(label_10_subjs, y_x_threshold)       

                    Coef_Fs = threshold
                    ACC_iters_Fs.append(acc)
                    NMI_iters_Fs.append(nmi)
                    PUR_iters_Fs.append(pur)
                    ARI_iters_Fs.append(ari)
                    RI_iters_Fs.append(ri)
                    F1_macro_iters_Fs.append(f1_macro)
                    PRC_macro_iters_Fs.append(precision_macro)
                    RC_macro_iters_Fs.append(recall_macro)
                    F1_micro_iters_Fs.append(f1_micro)    
                    PRC_micro_iters_Fs.append(precision_micro) 
                    RC_micro_iters_Fs.append(recall_micro) 

                    # C_A
                    C_A = thrC(C_A,alpha)                           
                    y_x_C_A, _ = post_proC(C_A, label_10_subjs.max(), 10, 8)
                    acc,nmi,pur, ari,ri,  f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro  = eva(label_10_subjs, y_x_C_A) 

                    # C_S
                    C_S = thrC(C_S,alpha)                                
                    y_x_C_S, _ = post_proC(C_S, label_10_subjs.max(), 10, 8)
                    acc,nmi,pur, ari,ri,  f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro  = eva(label_10_subjs, y_x_C_S)

            # Coef_Fs
            ACC_max = np.max(ACC_iters_Fs)
            nmi_max = np.max(NMI_iters_Fs)
            pur_max = np.max(PUR_iters_Fs)
            ari_max = np.max(ARI_iters_Fs)
            ri_max = np.max(RI_iters_Fs)
            F1_macro_max = np.max(F1_macro_iters_Fs)
            PRC_macro_max = np.max(PRC_macro_iters_Fs)
            RC_macro_max = np.max(RC_macro_iters_Fs)
            F1_micro_max = np.max(F1_micro_iters_Fs)
            PRC_micro_max = np.max(PRC_micro_iters_Fs)
            RC_micro_max = np.max(RC_micro_iters_Fs)
            iters10_ACC_Fs.append(round(ACC_max,5))
            iters10_NMI_Fs.append(round(nmi_max,5))
            iters10_PUR_Fs.append(round(pur_max,5))
            iters10_ARI_Fs.append(round(ari_max,5))
            iters10_RI_Fs.append(round(ri_max,5))
            iters10_F1_macro_Fs.append(round(F1_macro_max,5))
            iters10_PRC_macro_Fs.append(round(PRC_macro_max,5))
            iters10_RC_macro_Fs.append(round(RC_macro_max,5))
            iters10_F1_micro_Fs.append(round(F1_micro_max,5))
            iters10_PRC_micro_Fs.append(round(PRC_micro_max,5))
            iters10_RC_micro_Fs.append(round(RC_micro_max,5))

    print("#################       Fusion         ####################", file=file_out)
    print('[MAX] ===acc, nmi, pur, ari, ri,  f1_macro, precision_macro, recall_macro,  f1_micro, precision_micro, recall_micro: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(\
        round(np.max(iters10_ACC_Fs),5), \
        round(iters10_NMI_Fs[np.argmax(iters10_ACC_Fs)],5), round(iters10_PUR_Fs[np.argmax(iters10_ACC_Fs)],5),      \
        round(iters10_ARI_Fs[np.argmax(iters10_ACC_Fs)],5), round(iters10_RI_Fs[np.argmax(iters10_ACC_Fs)],5),    \
        round(iters10_F1_macro_Fs[np.argmax(iters10_ACC_Fs)],5), round(iters10_PRC_macro_Fs[np.argmax(iters10_ACC_Fs)],5),\
        round(iters10_RC_macro_Fs[np.argmax(iters10_ACC_Fs)],5), round(iters10_F1_micro_Fs[np.argmax(iters10_ACC_Fs)],5),\
        round(iters10_PRC_micro_Fs[np.argmax(iters10_ACC_Fs)],5),round(iters10_RC_micro_Fs[np.argmax(iters10_ACC_Fs)],5)), file=file_out)

    m = np.mean(iters10_ACC_Fs)
    me = np.median(iters10_ACC_Fs)
    
    return (1-m), (1-me)  

if __name__ == '__main__':

    data = sio.loadmat('./Data/umist-32-32.mat')
    
    dataname = 'Umist'
    Img = data['img']
    Label = data['label'] 

    n_input = [32,32]
    n_hidden = [15, 10, 5]
    kernel_size = [5,3,3]

    Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1],1]) 

    # Umist
    all_subjects = [ 20 ]
    avg = []
    med = []

    iter_loop = 0
    while iter_loop < len(all_subjects):
        num_class = all_subjects[iter_loop]
        batch_size = num_class * 24
        # [Fine-tune]
        reg1s   = [1000]    #   [0.001, 0.01, 0.1, 1, 10, 100, 1000] # 
        reg2s   = [0.001]   #   [0.001, 0.01, 0.1, 1, 10, 100, 1000] # 
        reg3s   = [0.1]     #   [0.001, 0.01, 0.1, 1, 10, 100, 1000] # 
        lrs     = [0.5e-4]  #   [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4] 
        alphas  = [0.08]    #   
        for lr in lrs:
            for reg1 in reg1s:
                for reg2 in reg2s:
                    for reg3 in reg3s:
                        for alpha in alphas:
                            model_path = './pretrain-model-umist/model-32x32-umist.ckpt' 
                            restore_path = './pretrain-model-umist/model-32x32-umist.ckpt' 
                            logs_path = './logs_' + dataname
                            tf.compat.v1.reset_default_graph()
                            CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, reg_constant2=reg2, reg_constant3=reg3, \
                                        kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, restore_path=restore_path, logs_path=logs_path)
                            avg_i, med_i = test_face(Img, Label, CAE, num_class,\
                                reg1, reg2, reg3, alpha, lr)
                            avg.append(avg_i)
                            med.append(med_i)
                            iter_loop = iter_loop + 1    

    toc = time.time()
    print("Time:", (toc - tic))