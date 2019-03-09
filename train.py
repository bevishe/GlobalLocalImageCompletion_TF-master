# _*_ coding:utf-8 _*_
import tensorflow as tf
import config 
import network
import os


def train(args, sess, model):
    # Adam optimizers are used instead of AdaDelta
    # parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, help='learning rate of the optimizer')
    # parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum of the optimizer')
    """
    :param args: 预先设置的一些参数，学习率
    :param sess:
    :param model: 模型
    :return:
    """
    d_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_D").minimize(model.d_loss, var_list=model.d_vars)
    c_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_C").minimize(model.recon_loss, var_list=model.c_vars)
    
    global_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_C").minimize(model.loss_all, var_list=model.c_vars)

    epoch = 0
    step = 0
    global_step = 0

    # saver
    # 创建saver对象实例用来保存训练好的模型和调用已保存的模型
    saver = tf.train.Saver()

    # 默认continue_training设置的是false
    if args.continue_training:
        tf.local_variables_initializer().run()
        last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
        saver.restore(sess, last_ckpt)
        ckpt_name = str(last_ckpt)
        print ("Loaded model file from " + ckpt_name)
        epoch = int(ckpt_name.split('-')[-1])
    else:
        # init = tf.initialize_all_variables()
        '''替换上面的函数,作用是返回一个初始化所有全局变量的操作(op),要是你把图投放到一个session后,你可以通过run
        操作来初始化所有的全局变量,本质相当于variable_initializers(global_variables())
        '''
        tf.global_variables_initializer().run()
        '''
        返回一个初始化所有局部变量的操作(op),要是你把图投放到一个session之中,你就能通过run这个操作来初始化所有的
        局部变量,本质相当有variable_initliazers(local_variables())
        '''
        tf.local_variables_initializer().run()
    '''
    tensorflow的session是支持多线程的,可以在同一个session中创建多个线程,并行执行.在Session中的所有线程都必须能够
    同步终止,异常都必须能正确捕获并报告,session终止的时候,队列必须能被正确的关闭.tensorflow提供了两个类来实现对
    session中多线程的管理:tf.Coordinator和tf.QueueRunner,这两个类往往一起使用
    Coordinator类用来管理在Session中的多线程,可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告
    异常,该线程捕获到这个异常之后就会终止所有的线程.使用tf.train.Coordinator()来创建一个线程管理器(协调器)对象
    QueueRunner类来启动tensor的入队线程,可以用来启动多个工作线程同时将多个tensor推入到文件名称队列中,具体的执行函数是
    tf.train.start_queue_runners,只有调用该函数之后才会真正把tensor推入到内存序列中,供计算单元调用,否则会由于内存
    序列为空,数据流程图会处于一直等待状态
    https://blog.csdn.net/dcrmg/article/details/79780331
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    # summary init
    all_summary = tf.summary.merge([model.recon_loss_sum,
                                    model.d_loss_sum,
                                    model.loss_all_sum,
                                    model.input_img_sum, 
                                    model.real_img_sum,
                                    model.recon_img_sum,
                                    model.g_local_imgs_sum,
                                    model.r_local_imgs_sum])
    writer = tf.summary.FileWriter(args.graph_path, sess.graph)

    # training starts here

    # first train completion network
    while epoch < args.train_step:

        # Training Stage 1 (Completion Network)
        if epoch < args.Tc:
            summary, c_loss, _ = sess.run([all_summary, model.recon_loss, c_optimizer])
            if epoch%200 == 0:
                writer.add_summary(summary, global_step)
                global_step += 1
            print ("Epoch [%d] Step [%d] C Loss: [%.4f]" % (epoch, step, c_loss))
        elif epoch < args.Tc + args.Td:
            # Training Stage 2 (Discriminator Network)
            summary, d_loss, _ = sess.run([all_summary, model.d_loss, d_optimizer])
            if epoch%200 == 0:
                writer.add_summary(summary, global_step)
                global_step += 1
            print ("Epoch [%d] Step [%d] D Loss: [%.4f]" % (epoch, step, d_loss))
        else:
            # Training Stage 3 (Completion Network)
            summary, g_loss, _ = sess.run([all_summary, model.loss_all, global_optimizer])
            if epoch%200 == 0:
                writer.add_summary(summary, global_step)
                global_step += 1
            print ("Epoch [%d] Step [%d] G Loss: [%.4f]" % (epoch, step, g_loss) )
        # Check Test image results every time epoch is finished
        if step*args.batch_size >= model.data_count:
            saver.save(sess, args.checkpoints_path + "/model")
            # global_step=epoch)

            # res_img = sess.run(model.test_res_imgs)
            
            # save test img result
            # img_tile(epoch, args, res_img)
            step = 0
            epoch += 1 

        step += 1
    coord.request_stop()
    coord.join(threads)
    sess.close()            
    print("Done.")


def main(args):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    # create graph, images, and checkpoints folder if they don't exist
    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    if not os.path.exists(args.graph_path):
        os.makedirs(args.graph_path)
    if not os.path.exists(args.images_path):
        os.makedirs(args.images_path)

    with tf.Session(config=run_config) as sess:
        # network模块中有network对象，用来定义网络结构
        model = network.network(args)


        print('Start Training...')
        train(args, sess, model)

main(config.args)

# Still Working....
