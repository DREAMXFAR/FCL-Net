function eval_epoch_rcf_png(exp_dir, epoch) 
    addpath('matlab_code/edges/')
    addpath('matlab_code/edges/my')
    addpath('matlab_code/edges/models')
    addpath('matlab_code/toolbox/')
    addpath('matlab_code/toolbox/channels')
    addpath('matlab_code/toolbox/classify')
    addpath('matlab_code/toolbox/detector')
    addpath('matlab_code/toolbox/filters')
    addpath('matlab_code/toolbox/images')
    addpath('matlab_code/toolbox/matlab')
    addpath('matlab_code/toolbox/videos')

    root_root = 'ckpt/standard_RCF/log'; %standard_RCF
    % root_root = 'ckpt/standard_RCF_PASCAL/log';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %exp_dir = new_github_adam_lr1e-4_Aug_0.3_savemodel_Sep12_16-23-54
    %epoch = 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    root = fullfile(root_root, exp_dir, 'results_mat',  num2str(epoch));
    save_root = fullfile(root_root, exp_dir, 'results_mat',  [num2str(epoch), '_nms']);
    mkdir(save_root);

    dsn_folders = dir(root);
    dsn_folders = dsn_folders(3:end, :);

    for dsn_folder_ind = 1:size(dsn_folders,1)

        cur_dsn_folder = dsn_folders(dsn_folder_ind).name;

        cur_save_root = fullfile(save_root, cur_dsn_folder);
        mkdir(cur_save_root);
        
        files = dir( fullfile(root, cur_dsn_folder) );
        files = files(3:end, :);
        
        for image_ind = 1:size(files,1)
            file_name = fullfile(root, cur_dsn_folder, files(image_ind).name);
%             matObj = matfile(file_name);
%             varlist = who(matObj);
%             x = matObj.(char(varlist));
            x = imread(file_name);
            x = double(x)/255.0;

            E=convTri(single(x),1);
            [Ox,Oy]=gradient2(convTri(E,4));
            [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
            O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
            E=edgesNmsMex(E,O,1,5,1.01,4);

            [~, name, ~] = fileparts(files(image_ind).name);
            save_path = fullfile( save_root, cur_dsn_folder, [name,'.png'] );
            imwrite( uint8(E*255), save_path );
        end
    end
    
    %%%% merge two dsn (delete)


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gtDir = 'data/BSR/BSDS500/data/groundTruth/test';
    
    dst = 0.0075;
    
    resDir6 = fullfile(save_root, 'dsn6');
    [ODS_F_fuse6, ~, ~, ~, OIS_F_fuse6, ~, ~, AP_fuse6, ~] = edgesEvalDir('resDir',resDir6,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',dst);

    resDir7 = fullfile(save_root, 'dsn7');
    [ODS_F_fuse7, ~, ~, ~, OIS_F_fuse7, ~, ~, AP_fuse7, ~] = edgesEvalDir('resDir',resDir7,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',dst);
    
    resDir2 = fullfile(save_root, 'dsn2');
    [ODS_F_fuse2, ~, ~, ~, OIS_F_fuse2, ~, ~, AP_fuse2, ~] = edgesEvalDir('resDir',resDir2,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',dst);
    
%     resDir = fullfile(save_root, 'dsn1');
%     [ODS_F_fuse, ~, ~, ~, OIS_F_fuse, ~, ~, AP_fuse, ~] = edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',dst);
%     
%     resDir3 = fullfile(save_root, 'dsn3');
%     [ODS_F_fuse3, ~, ~, ~, OIS_F_fuse3, ~, ~, AP_fuse3, ~] = edgesEvalDir('resDir',resDir3,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',dst);
%     
%     resDir4 = fullfile(save_root, 'dsn4');
%     [ODS_F_fuse4, ~, ~, ~, OIS_F_fuse4, ~, ~, AP_fuse4, ~] = edgesEvalDir('resDir',resDir4,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',dst);
%     
%     resDir5 = fullfile(save_root, 'dsn5');
%     [ODS_F_fuse5, ~, ~, ~, OIS_F_fuse5, ~, ~, AP_fuse5, ~] = edgesEvalDir('resDir',resDir5,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',dst);
    
     % add fusion 0425

    result_txt_path = fullfile(root_root, exp_dir, 'result.txt');
    if exist(result_txt_path)==2
        f = fopen(result_txt_path, 'a');
    else
        f = fopen(result_txt_path, 'w');
    end
    
    fprintf(f, '%d \n',  str2num(epoch)); 
%     fprintf(f, '1: %0.5f %0.5f %0.5f \n', ODS_F_fuse, OIS_F_fuse, AP_fuse);
    fprintf(f, '2: %0.5f %0.5f %0.5f \n', ODS_F_fuse2, OIS_F_fuse2, AP_fuse2);
%     fprintf(f, '3: %0.5f %0.5f %0.5f \n', ODS_F_fuse3, OIS_F_fuse3, AP_fuse3);
%     fprintf(f, '4: %0.5f %0.5f %0.5f \n', ODS_F_fuse4, OIS_F_fuse4, AP_fuse4);
%     fprintf(f, '5: %0.5f %0.5f %0.5f \n', ODS_F_fuse5, OIS_F_fuse5, AP_fuse5);
    fprintf(f, '6: %0.5f %0.5f %0.5f \n', ODS_F_fuse6, OIS_F_fuse6, AP_fuse6);  
    fprintf(f, '7: %0.5f %0.5f %0.5f \n', ODS_F_fuse7, OIS_F_fuse7, AP_fuse7);  % add fusion 0425
    
    fprintf('cur_test: %s %d\n', exp_dir, str2num(epoch));
    
    exit
end





