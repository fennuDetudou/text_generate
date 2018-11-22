#! /usr/vin/env python3
# coding=utf-8
# author: tudou

import tensorflow as tf
import numpy as np
import time
import os

def get_n_top(probs,vocab,n=5):
    p=np.squeeze(probs)
    #### 1. 只选取概率最高的n个词，所以需要将其他位置的概率置为0
    #### 2. 并重新计算n个词的取值概率，即归一化
    # 置0
    p[np.argsort(p)[:-n]]=0
    # 归一化
    p=p/np.sum(p)
    c=np.random.choice(vocab, 1 , p=p)
    return c

class Char_RNN(object):
    def __init__(self,num_classes,num_seqs=32,num_steps=50,lstm_size=128,num_layers=2,
                 use_embedding=False,embedding_size=64,
                 is_train=False,learning_rate=0.001,sampling=False):

        if sampling is True:
            self.num_seqs,self.num_steps=1,1
        else:
            self.num_seqs = num_seqs
            self.num_steps = num_steps

        self.num_classes = num_classes
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.use_embeding = use_embedding
        self.embedding_size = embedding_size
        self.is_train = is_train
        self.learning_rate = learning_rate

        if self.is_train is True:
            self.keep_probs = 0.5
        else:
            self.keep_probs = 1.0

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver=tf.train.Saver()

    def build_inputs(self):
        with tf.variable_scope('inputs'):
            #### 输入要转化为one_hot向量，要用int而不是float
            self.inputs=tf.placeholder(tf.int32,shape=(self.num_seqs,self.num_steps))
            self.labels=tf.placeholder(tf.int32,shape=(self.num_seqs,self.num_steps))
            self.keep_prob=tf.placeholder(tf.float32,shape=None)

            if self.use_embeding is True:
                #### tf embedding层
                ### 1. 要在 TensorFlow 中创建 embeddings，我们首先将文本拆分成单词，然后为词汇表中的每个单词分配一个整数
                ### 2. 利用整数组成的向量训练embedding层，输出shape为vocabulary_size,embedding_size
                ### 3. 利用embedding_lookup(embedding层，整数向量)获得inputs的分布式表示
                embedding=tf.get_variable('embeddings',shape=(self.num_classes,self.embedding_size))
                self.lstm_inputs=tf.nn.embedding_lookup(embedding,self.inputs)
            else:
                self.lstm_inputs=tf.one_hot(self.inputs,self.num_classes)

    def build_lstm(self):
        def _get_cell():
            cell=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            drop=tf.nn.rnn_cell.DropoutWrapper(cell,self.keep_prob)
            return drop

        with tf.variable_scope('lstm'):
            multi_cell=tf.nn.rnn_cell.MultiRNNCell([_get_cell() for _ in range(self.num_layers)])
            self.initial_state=multi_cell.zero_state(self.num_seqs,tf.float32)
            outputs,self.final_state=tf.nn.dynamic_rnn(multi_cell,self.lstm_inputs,
                                                       initial_state=self.initial_state)
            #### dynamic_rnn输出为(h1,h2,h3,...),所以将它们拼接成一个矩阵
            self.outputs=tf.concat(outputs,1)
            x=tf.reshape(self.outputs,[-1,self.lstm_size])

        with tf.variable_scope("softmax"):
            w=tf.get_variable('softmax_w',shape=[self.lstm_size,self.num_classes],dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer)
            b=tf.get_variable('softmax_b',self.num_classes,dtype=tf.float32)

            tf.summary.histogram('softmax_w',w)
            tf.summary.histogram('softmax_b',b)

            self.logits=tf.matmul(x,w)+b
            self.prob=tf.nn.softmax(self.logits)
            tf.summary.histogram('prob',self.prob)

    def build_loss(self):
        with tf.variable_scope('loss'):
            #### softmax标签要处理
            one_hot_labels=tf.one_hot(self.labels,self.num_classes)
            labels=tf.reshape(one_hot_labels,self.logits.get_shape())
            loss=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=self.logits)
            self.loss=tf.reduce_mean(loss)
            tf.summary.scalar('loss',self.loss)
    def build_optimizer(self):
        #### rnn容易导致梯度消失，所以需要进行梯度裁剪
        optimizer=tf.train.AdamOptimizer(self.learning_rate)
        gradient_vars=optimizer.compute_gradients(self.loss)
        cropped_gradient=[(tf.clip_by_value(grad,-1,1),var) for grad,var in gradient_vars]
        self.train_op=optimizer.apply_gradients(cropped_gradient)

    def train(self,batch_generator,max_steps,save_path,log_every_n,save_every_n):
        #### 标准流程
        self.sess=tf.Session()
        mergerd=tf.summary.merge_all()
        init=(tf.global_variables_initializer(),tf.local_variables_initializer())
        with self.sess as sess:
            trian_writer = tf.summary.FileWriter(save_path + '/logdir', sess.graph)
            sess.run(init)
            step=0
            new_state=sess.run(self.initial_state)
            for x,y in batch_generator:
                step+=1
                start = time.time()
                feed={
                    self.inputs:x,
                    self.labels:y,
                    self.keep_prob:self.keep_probs,
                    self.initial_state:new_state
                }
                ####
                batch_loss,new_state,_,summ=sess.run([self.loss,self.final_state,self.train_op,mergerd],
                                                feed_dict=feed)
                trian_writer.add_summary(summ,step)
                end=time.time()
                if step%log_every_n==0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if step%save_every_n==0:
                    self.saver.save(sess,os.path.join(save_path,'model'),step)
                if step>max_steps:
                    break
            self.saver.save(sess,os.path.join(save_path,'model'),step)

    def sample(self,n_samples,prime,vacab_size):

        samples=[c for c in prime]
        sess=self.sess
        new_state=sess.run(self.initial_state)
        #### 初始化概率
        preds=np.ones((vacab_size,))
        for c in prime:
            #### 输入单个字符
            x=np.zeros((1,1))
            x[0,0]=c
            feed={self.inputs:x,
                  self.keep_prob:self.keep_probs,
                  self.initial_state:new_state
            }
            preds,new_state=sess.run([self.prob,self.final_state],feed_dict=feed)

        c=get_n_top(preds,vocab=vacab_size)
        samples.append(c)

        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: self.keep_probs,
                    self.initial_state: new_state
                    }
            preds, new_state = sess.run([self.prob, self.final_state], feed_dict=feed)
            c = get_n_top(preds, vocab=vacab_size)
            samples.append(c)

        return np.array(samples)

    def load(self,checkpoint):
        self.sess=tf.Session()
        self.saver.restore(self.sess,checkpoint)
        print("restore from {}".format(checkpoint))




