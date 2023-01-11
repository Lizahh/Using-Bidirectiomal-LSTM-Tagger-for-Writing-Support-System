if __name__ == "__main__":

    # python    re_models.py    gpu_id       source_dir     target_dir        model_id     train/test     multi_factor_count
    # python    sys.argv[0]     sys.argv[1]  sys.argv[2]    sys.argv[3]      sys.argv[4]    sys.argv[5]   sys.argv[6]
    
    # Use ""model_id"" as 1 for CNN,;; 2 for PCNN;; 3 for EA,;; 4 for BGWA;;;, and 5 for our models.
    # os.environ in Python is a mapping object that represents the user’s environmental variables. It returns a dictionary 
    # having user’s environmental variable as key and their values as value.
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    
    # the following 6 lines will be used to manually reproduce the same result each time 
    random_seed = 1023    
    np.random.seed(random_seed) # it is the way to use the numpy to generate random number to reproduce 
    random.seed(random_seed)
    torch.manual_seed(random_seed) # we have manually specified it to reproduce the same result each time
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed) # cuda will also manully reproduce the result

    src_data_folder = sys.argv[2]
    trg_data_folder = sys.argv[3]
    model_name = int(sys.argv[4])
    job_mode = sys.argv[5]  # job mode means konsi job krni eh training or testing 
    
    # mfc = it is multi factor count. which by default will be 1 q k mfc = 1 gives the highest F1 score on NYT10 and for NYT11, mfc = 4 gives best
    mfc = 1
    
    # agr apka model name 5 eh. to iska mtlb eh hmara apna model run hony lga eh so sys.argv[6] which is the factor count, usko 0 se le kr 5 tk set krdo. 
    if model_name == 5:
        mfc = int(sys.argv[6])
        assert 0 <= mfc <= 5

    # agr target folder exist ni krta to khud se create krdo directory for target
    if not os.path.exists(trg_data_folder):
        os.mkdir(trg_data_folder)

    # these are the hyperparameters
    batch_size = 50
    num_epoch = 50
    max_sent_len = 100
    dep_win_size_rel = 5
    use_dep_dist = True
    dim_size = 50
    
    # mtlb abi ni btaya hm ne k konsi softmax lgani eh. adaptive ya konsi koi or.
    softmax_type = 0
    word_embed_dim = 50
    
    # conv_filter_cnt = convolutions filter count for feature extraction
    conv_filter_cnt = 230
    
    # distance_embed_dim shows the depenedency distance of the words from the entities
    distance_embed_dim = 5
    
    # the below is: entity indicator embedding dimensions
    ent_indicator_embed_dim = 10
    
   # embeddings pe dropout apply ni krna
    apply_embed_dropout = False
    
    # the file of the embeddings
    embedding_file = os.path.join(src_data_folder, 'w2v.txt')
    
    # word density means total words to take into consideration: 5 words before entity and 5 words after entity
    
    word_density = 10
    drop_out = 0.5
    # ctx_len means context length. 5 words 
    ctx_len = 5
    
    # Too many epochs can lead to overfitting of the training dataset, whereas too few may result in an underfit model. 
    #Early stopping is a method that allows you to specify an arbitrary large number  of training epochs and stop training 
    #once the model performance stops improving on a hold out validation dataset.
    
    # so here it is set as 3 mtlb har after 3 epoch k check kry if the imporvement has stopped or not
    early_stop_cnt = 3
    
    # mtlb bi-directional lstm use krna eh
    lstm_direction = 2
    
    # jin relations ko ignore krna eh: none, NA, and other
    ignore_rel_list = ['None', 'NA', 'Other']
    
    # to check if the required type is available or not. filhaal it is set to False
    is_type_available = False
    
    # os.path.join(src_data_folder, 'relations.txt') kr k pehly in dono folders ko join krdo
    # then usko aik function "get_class_label_map" ko day do as input. 
    # that function will return  2 values. as you can see in the function below.
    # basic purpose of this function is to get the class label of a particular relation
    relation_cls_label_map, rel_label_cls_map = get_class_label_map(os.path.join(src_data_folder, 'relations.txt'))
    
    <<<<<<def get_class_label_map(rel_file):
    # 2 ordered dictionaries bna lo: one for class labels map and other for labels class map
    # cls_label_map is used for the label index e.g, label 0, label 1 etc
    # label_cls_map is used to put the relation from the relation.txt file 
    cls_label_map = collections.OrderedDict()
    label_cls_map = collections.OrderedDict()
    
    # now open the relation.txt file to read
    reader = open(rel_file)
    
    # take the lines of relations.txt and one by one add in "lines"
    lines = reader.readlines()
    reader.close()
    
    # now add values in the ordered dictionaries
    label = 0
    for line in lines:
        # for each line in lines, strip() the line mtlb k line k left and right se white space remove krdo
        line = line.strip()
        
        #put the label index in the cls_label_map dict
        # put the relation in the label_cls_map dict  
        cls_label_map[line] = label
        label_cls_map[label] = line
        # increment the label index by 1
        label += 1
        
        # return both the dictionaries
    return cls_label_map, label_cls_map>>>>>>
    
    # shaid is ka mtlb eh k for each argument, the max word size of the argument should be 30 characters long
    max_word_arg_head_dist = 30
    dist_vocab_size = 2 * max_word_arg_head_dist + 1

    # iska ni pta k yh kya hai
    QASample = recordclass("QASample", "UID Id Len Text Arg1 Arg2 Words WordsMask WordsArg1Dist WordsArg2Dist "
                                       "WordsEntIndicator WordsArg1DepDist WordsArg2DepDist WordsDepDist "
                                       "Arg1Mask Arg2Mask Piece1Mask Piece2Mask Piece3Mask RelationName")

    # now we will start writing for the training
    # train a model
    if job_mode == 'train':
    
        # sb se pehly trg_data_folder waly folder me aik folder mazeed bnayen gay with name training.log and usko writing mode me 
        # open kren gay or us k ander sary log messages jo trianing k doran print hon gay, wo write krty jayen gay
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')
        
        <<<def custom_print(*msg):
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))>>>>>
            
        custom_print(sys.argv)  # sb se pehly pury arguments show kro mtlb same yei line print krdo: ['re_models.py', '0', 'NYT11/', 'NYT11/Best/', '5', 'test', '4']
        custom_print(dep_win_size_rel)
        custom_print(ctx_len)
        custom_print(drop_out)
        custom_print('loading data......')
        
        # the below line will save the best model in the model.h5py in the trg_data_folder folder
        best_model_file_name = os.path.join(trg_data_folder, 'model.h5py')
        
        # the below are the output, train, dev and test files
        out_file_name = os.path.join(trg_data_folder, 'dev-relation-out.txt')
        train_file = os.path.join(src_data_folder, 'train.json')
        dev_file = os.path.join(src_data_folder, 'dev.json')
        test_file = os.path.join(src_data_folder, 'test.json')
        
        # the below are the dependency files for training, developement and testing
        train_dep_file = os.path.join(src_data_folder, 'train.dep')
        dev_dep_file = os.path.join(src_data_folder, 'dev.dep')
        test_dep_file = os.path.join(src_data_folder, 'test.dep')
        
        # give the training file, along with the dep file to read_Data from it. same for dev and test
        train_data = read_data(train_file, train_dep_file, is_training_data=True)
        dev_data = read_data(dev_file, dev_dep_file)
        test_data = read_data(test_file, test_dep_file)

        # print the training, testing and dev data sizes
        custom_print('Training data size:', len(train_data))
        custom_print('Development data size:', len(dev_data))
        custom_print('Test data size:', len(test_data))
        
        # NOW prepare the vocab and save in target folder
        custom_print("preparing vocabulary......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        all_data = train_data + dev_data + test_data
        
        # now we will build our vocabulary from the training, dev and testing dataset to gain the maximum words
        word_vocab, word_embed_matrix = build_vocab(train_data, dev_data, test_data, vocab_file_name, embedding_file)
        
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        def build_vocab(train, dev, test, vocab_file, embedding_file):
        # make a dictionary to save vocab in that
    vocab = OrderedDict()
    
    # join the training, dev and testing data
    for d in train + dev + test:
        
        # split the data and get one word each time in each iteration of for loop
        for word in d.Text.split():
            # strip the word to remove the white space from the left and right side of the word
            word = word.strip()
            # if the length of the word > 0 means kch characters to hain is me at least
            if len(word) > 0:
                # agr wo word pehly se vocabulary me ni eh to usko add krdo is trha: e.g. word = "my", in vocab dict it will become: my = 1. mtlb 1 bar e aya eh abi
                if word not in vocab:
                    vocab[word] = 1
                    # agr vocab me pehly se tha e.g. pehly se e my 3 bar likha hua tha to us me 1 plus krdo k lo g aik bar or agya. so it will become my = 3 + 1 = 4
                else:
                    vocab[word] += 1


<<<<<<<<<<<<<
     def load_word_embedding(embed_file, vocab):
    custom_print('vocab length:', len(vocab))
    # custom_print('entity vocab length', len(entity_vocab))
    embed_vocab = OrderedDict()
    embed_matrix = list()
    embed_vocab['<PAD>'] = 0    # mtlb 0 se padding krdo
    embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))     # mtlb bs 50 zeros add krdo matrix me of type float
    embed_vocab['<UNK>'] = 1
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim)) # mtlb append krdo is me: 5 columns and 10 rows jin ki values randomly in me se kch hon gi: -0.25 se 0.25
    word_idx = 2        # word index is already two because 1 is for paddding and 1 is for unknown

    # open embedding file in reading mode
    with open(embed_file, "r") as f:
        for line in f:  
            # for each line in file, split it into its parts/words
            parts = line.split()
            if len(parts) < word_embed_dim + 1:     # agr word ka size 50 + 1 = 51 se less eh. to skip krdo usko or next command pe chaly jao
                continue
            word = parts[0]     # word ka 0th token lo word k andr
            if word in vocab and vocab[word] >= word_density:    # mtlb agr wo word "vocab" k andr mojood eh or wo 10 se zada martaba hai (q k word density =10)
                    vec = [np.float32(val) for val in parts[1:]]    # to embedding file me us k 0th index ko chor k baki puri us word ki embedding le k usko float me convert krdo
                    embed_matrix.append(vec)    # ab usko embed_matrix me add krdo
                    embed_vocab[word] = word_idx    # us word k 0th index pe word_index rakh do
                    word_idx += 1   # increment the word index by one

    custom_print('embed vocab length:', len(embed_vocab))   # now print the size of the embedding vocab

    for word in vocab:
        if word not in embed_vocab and vocab[word] >= word_density: # agr wo word pehly se "embedding vocab" me ni eh or 10 se zada martaba hai wo vocab me
            # custom_print(word)
            embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim)) # jab jab esa word aye ga koi tb tb hm ne embedding matrix me jab 10 rows and 5 columns -0.25 to 0.25 k darmyan add kr deny hain
            embed_vocab[word] = word_idx
            word_idx += 1

    custom_print('embed vocab length:', len(embed_vocab))       # ab dubara embedding vocabulary ki length print krdo
    return embed_vocab, np.array(embed_matrix, dtype=np.float32)    #mtlb embedding vocab return krdo and embedding matrix return krdo. np.array does nothing special here



    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    embed_vocab, embed_matrix = load_word_embedding(embedding_file, vocab)  # that embedding matrix has become vocabulary here.
    output = open(vocab_file, 'wb') # now open the "vocab.pk1 file here"
    # Once the file is opened for writing, you can use pickle. dump() , which takes two arguments: 
    #the object you want to pickle and the file to which the object has to be saved. 
    #In this case, the former will be dogs_dict , while the latter will be outfile
    pickle.dump(embed_vocab, output)    # usko dubara dump kr k values get krlo us me mojood
    output.close()
    return embed_vocab, embed_matrix
    
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    


        custom_print('vocab size:', len(word_vocab))    # now print the final size of the vocabulary

        custom_print("Training started......")  # start doing the training
        torch_train(model_name, train_data, dev_data, test_data, best_model_file_name)
        
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# here the arguments are like the following:
    # "model_id" = model_name i.e Use model_id as 1 for CNN, 2 for PCNN, 3 for EA, 4 for BGWA, and 5 for our models.
    # "train_samples" = train_data
    # "dev_samples" = dev_data
    # "test_samples" = test_data
    # "best_model_file" =  best_model_file_name which has the model.h5py file in target folder
        def torch_train(model_id, train_samples, dev_samples, test_samples, best_model_file):
        
     # take the size of the training data
    train_size = len(train_samples)
    
    # mtlb if batch_count = int(math.ceil(10000/50)) = then batch_count will be equal to 200.
    batch_count = int(math.ceil(train_size/batch_size))
    move_last_batch = False
    if len(train_samples) - batch_size * (batch_count - 1) == 1:    # mtlb agr hm akhri batch pe pohnch chukay hain
        move_last_batch = True # set this flag as true
        batch_count -= 1        # batch count me se 1 minus krdo
    custom_print(batch_count)   # print the batch count
    model = get_model(model_id) # get the model 
    
    <<<<<<<<<<<<<<<<
    def get_model(model_id):
    if model_id == 1:
        return CNN()
    if model_id == 2:
        return PCNN()
    if model_id == 3:
        return EA()
    if model_id == 4:
        return BGWA()
    if model_id == 5:
        return MFA()

    >>>>>>>>>>>>>>>>>>

    custom_print(model)  # print the whole form of the model.
    if torch.cuda.is_available():
        model.cuda()    # shift model on cuda

    rel_loss_func = nn.NLLLoss() # The Negative Log Likelihood loss. It is useful to train a classification problem .   
    optimizer = optim.Adagrad(model.parameters())   # optimize the model parameters
    custom_print(optimizer) #print the optimizers

    best_dev_acc = -1.0     
    best_epoch_idx = -1
    best_epoch_seed = -1
    for epoch_idx in range(0, num_epoch):   # run the following for each epoch
        model.train()
        custom_print('Epoch:', epoch_idx + 1)   # print the epoch number
        cur_seed = random_seed + epoch_idx + 1  # add random_Seed which is manual in our case because we want to produce same result each time
        np.random.seed(cur_seed)
        torch.cuda.manual_seed(cur_seed)
        random.seed(cur_seed)
        cur_shuffled_train_data = shuffle_data(train_samples)# SHUFFLE THE TRAINING SAMPLES
        start_time = datetime.datetime.now() # CALCULATE THE STARTING TIME OF THE EPOCH TO KEEP A CHECK ON HOW MUCH TIME IT TAKES TO COMPLETE AN EPOCH
        train_loss_val = 0.0    # TRAINING LOSS VALUE WILL BE 0 AT START
        for batch_idx in tqdm(range(0, batch_count)):   # MTLB TQDM ki value 0 se le kr 200 tk chaly gi if batch_count will be 200.
            batch_start = batch_idx * batch_size # get the starting index of the batch 
            batch_end = min(len(cur_shuffled_train_data), batch_start + batch_size) # and get the end index as well
            if batch_idx == batch_count - 1 and move_last_batch:
                batch_end = len(cur_shuffled_train_data)

            # it showed the current batch
            cur_batch = cur_shuffled_train_data[batch_start:batch_end]  # get the shuffled training data from batch start till batch end
            cur_seq_len, cur_input, cur_target = get_batch_data(cur_batch, True)
            
            # the following all commands will convert the numpy array 'words' into tensor with the ability of auto gradient and transfer them to CUDA
            words_seq = autograd.Variable(torch.from_numpy(cur_input['words'].astype('long')).cuda())
            words_mask = autograd.Variable(torch.from_numpy(cur_input['wordsMask'].astype('float32')).cuda())
            arg1_lin_dist = autograd.Variable(torch.from_numpy(cur_input['arg1LinDist'].astype('long')).cuda())
            arg2_lin_dist = autograd.Variable(torch.from_numpy(cur_input['arg2LinDist'].astype('long')).cuda())
            ent_ind_seq = autograd.Variable(torch.from_numpy(cur_input['entIndicator'].astype('long')).cuda())

            arg_dep_dist = autograd.Variable(torch.from_numpy(cur_input['argDepDist'].astype('float32')).cuda())
            arg_dist_mask = autograd.Variable(torch.from_numpy(cur_input['argDistMask'].astype('uint8')).cuda())
            arg1_dep_dist = autograd.Variable(torch.from_numpy(cur_input['arg1DepDist'].astype('float32')).cuda())
            arg1_dist_mask = autograd.Variable(torch.from_numpy(cur_input['arg1DistMask'].astype('uint8')).cuda())
            arg2_dep_dist = autograd.Variable(torch.from_numpy(cur_input['arg2DepDist'].astype('float32')).cuda())
            arg2_dist_mask = autograd.Variable(torch.from_numpy(cur_input['arg2DistMask'].astype('uint8')).cuda())

            arg1 = autograd.Variable(torch.from_numpy(cur_input['arg1'].astype('long')).cuda())
            arg2 = autograd.Variable(torch.from_numpy(cur_input['arg2'].astype('long')).cuda())
            arg1_mask = autograd.Variable(torch.from_numpy(cur_input['arg1Mask'].astype('float32')).cuda())
            arg2_mask = autograd.Variable(torch.from_numpy(cur_input['arg2Mask'].astype('float32')).cuda())

            piece1mask_seq = autograd.Variable(torch.from_numpy(cur_input['piece1Mask'].astype('float32')).cuda())
            piece2mask_seq = autograd.Variable(torch.from_numpy(cur_input['piece2Mask'].astype('float32')).cuda())
            piece3mask_seq = autograd.Variable(torch.from_numpy(cur_input['piece3Mask'].astype('float32')).cuda())

            
            target = autograd.Variable(torch.from_numpy(cur_target['relation'].astype('long')).cuda())
                
            # THIS WILL GIVE THE ACCORDING INPUTS TO THE MODELS according to model_id
            if model_id in [1]:
                outputs = model(words_seq, words_mask, arg1_lin_dist, arg2_lin_dist, True)
            elif model_id == 2:
                outputs = model(words_seq, words_mask, arg1_lin_dist, arg2_lin_dist,
                                piece1mask_seq, piece2mask_seq, piece3mask_seq, True)
            elif model_id in [3]:
                outputs = model(words_seq, words_mask, arg1_lin_dist, arg2_lin_dist,
                                piece1mask_seq, piece2mask_seq, piece3mask_seq, arg1, arg2, True)
            elif model_id in [4]:
                outputs = model(words_seq, words_mask, arg1_lin_dist, arg2_lin_dist,
                                piece1mask_seq, piece2mask_seq, piece3mask_seq, True)
            elif model_id in [5]:
                outputs = model(words_seq, words_mask, ent_ind_seq, arg1_lin_dist, arg2_lin_dist,
                                arg_dep_dist, arg1_dep_dist, arg2_dep_dist, arg1_mask, arg2_mask, arg_dist_mask,
                                arg1_dist_mask, arg2_dist_mask, True)

            # give the outputs and targets to the rel_loss_func = nLLLoss()
            loss = rel_loss_func(outputs, target)
            # go backward to perform optimization and computation of the gradients
            loss.backward()
            
            torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step()
            train_loss_val += loss.data[0]
            model.zero_grad()

        train_loss_val /= batch_count
        # calculate the end - time of the epoch
        end_time = datetime.datetime.now()
        custom_print('Training Loss:', train_loss_val)
        custom_print('Time:', end_time - start_time)

        custom_print('\nDev Results\n')
        torch.cuda.manual_seed(random_seed)
        dev_preds = predict(dev_samples, model, model_id)
        # get the f1 SCORE
        pred_pos, gt_pos, correct_pos = get_F1(dev_samples, dev_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        
        # The following is the predicted for computing accuracy
        p = float(correct_pos) / (pred_pos + 1e-8)
        
        # following is the real or ground truth (gt) for computing accuracy
        r = float(correct_pos) / (gt_pos + 1e-8)
        dev_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('F1:', round(dev_acc, 3))

        if dev_acc >= best_dev_acc:
            best_epoch_idx = epoch_idx + 1
            best_epoch_seed = cur_seed
            custom_print('model saved......')
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_file)

        custom_print('\nTest Results\n')
        torch.cuda.manual_seed(random_seed)
        test_preds = predict(test_samples, model, model_id)

        pred_pos, gt_pos, correct_pos = get_F1(test_samples, test_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        test_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('F1:', round(test_acc, 3))

        custom_print('\n\n')
        if epoch_idx + 1 - best_epoch_idx >= early_stop_cnt:
            break

    custom_print('*******')
    custom_print('Best Epoch:', best_epoch_idx)
    custom_print('Best Epoch Seed:', best_epoch_seed)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        logger.close()

    if job_mode == 'test':
        logger = open(os.path.join(trg_data_folder, 'test.log'), 'w')
        custom_print(sys.argv)
        custom_print("loading word vectors......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab = load_vocab(vocab_file_name)

        word_embed_matrix = np.zeros((len(word_vocab), word_embed_dim), dtype=np.float32)
        custom_print('vocab size:', len(word_vocab))

        custom_print('seed:', random_seed)
        model_file = os.path.join(trg_data_folder, 'model.h5py')

        best_model = get_model(model_name)
        custom_print(best_model)
        if torch.cuda.is_available():
            best_model.cuda()
        best_model.load_state_dict(torch.load(model_file))

        # prediction on dev data

        dev_file = os.path.join(src_data_folder, 'dev.json')
        dev_dep_file = os.path.join(src_data_folder, 'dev.dep')
        dev_data = read_data(dev_file, dev_dep_file)
        custom_print('Dev data size:', len(dev_data))
        torch.cuda.manual_seed(random_seed)
        dev_preds = predict(dev_data, best_model, model_name)

        custom_print('\nDev Results')
        pred_pos, gt_pos, correct_pos = get_F1(dev_data, dev_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('P:', round(p, 3))
        custom_print('R:', round(r, 3))
        custom_print('F1 Before Thresholding:', round(acc, 3))

        threshold = get_threshold(dev_data, dev_preds)
        custom_print('\nThreshold:', round(threshold, 3))
        print()
        pred_pos, gt_pos, correct_pos = get_F1(dev_data, dev_preds, threshold)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('P:', round(p, 3))
        custom_print('R:', round(r, 3))
        custom_print('F1 After Thresholding:', round(acc, 3))

        test_files = ['test']
        for file_name in test_files:
            custom_print('\n\n\nTest Results:', file_name)
            test_input_file = os.path.join(src_data_folder, file_name + '.json')
            test_dep_file = os.path.join(src_data_folder, file_name + '.dep')
            test_data = read_data(test_input_file, test_dep_file)
            out_file_name = os.path.join(trg_data_folder, file_name + '-output.json')

            custom_print('Test data size:', len(test_data))
            torch.cuda.manual_seed(random_seed)
            test_preds = predict(test_data, best_model, model_name)

            pred_pos, gt_pos, correct_pos = get_F1(test_data, test_preds)
            custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
            p = float(correct_pos) / (pred_pos + 1e-8)
            r = float(correct_pos) / (gt_pos + 1e-8)
            test_acc = (2 * p * r) / (p + r + 1e-8)
            custom_print('F1 Before Thresholding:', round(test_acc, 3))
            print()

            pred_pos, gt_pos, correct_pos = get_F1(test_data, test_preds, threshold)
            custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
            p = float(correct_pos) / (pred_pos + 1e-8)
            r = float(correct_pos) / (gt_pos + 1e-8)
            test_acc = (2 * p * r) / (p + r + 1e-8)
            custom_print('P:', round(p, 3))
            custom_print('R:', round(r, 3))
            custom_print('F1 After Thresholding:', round(test_acc, 3))
            print()
            # write_pred_file(test_data, test_preds, out_file_name, threshold)
            # write_PR_curve(test_data, test_preds, os.path.join(trg_data_folder, file_name + '_pr_data.csv'))
            # pr_curve(os.path.join(trg_data_folder, file_name + '_pr_data.csv'),
            #          os.path.join(trg_data_folder, file_name + '_pr_curve.csv'))

        logger.close()