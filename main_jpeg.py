# -*- coding:utf-8  -*-
from libs.utils import *
from libs.MCTS import *

global MAX_DEPTH
global seed
global state
global sorted_idx
import glob
import shutil

if __name__ == '__main__':

    #Please note that tfilb should be placed in the home directory, i.e. /home/mxb/tflib/
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default="./data/cover_decompressed/", required=False, type=str, help='Path of dataset')
    parser.add_argument('--model_path', '-p', required=True, type=str, help='environmental_model_path')
    parser.add_argument('--sublattice_size', '-s', default=128, type=int, help='size of sublattice')
    parser.add_argument('--alpha', '-a', default=1.5, type=float, help='alpha')
    
    args = parser.parse_args()
    data_path = args.dataset
    
    baseline_folder = 'juniward'
    
    model_path = args.model_path
    print("Loading model: " + model_path)
    sublattice_size = args.sublattice_size
    MAX_DEPTH = sublattice_size * sublattice_size
    alpha = args.alpha
    print("Alpha: " + str(alpha))
    model, con, sess = build_srnet(model_path)
    print("Loading SrNet successfully!")
    dct_sess, res, DCT = buildDCT()
    file_list = glob.glob(data_path + '*.mat')
    idx_list = get_portion_idx()
    for path in file_list:
        cover_decompress = load_mat(path)
        
        img_name = path.split('/')[-1].split('.')[0]
        seed = int(img_name)
        jpeg_name = img_name + ".jpg"
        cover_jpeg_path = path.replace("decompressed", "jpeg")
        cover_jpeg_path = cover_jpeg_path.replace("mat","jpg")
        
        
        cover_dct_path = path.replace("decompressed", "DCT")
        C_COEFFS = load_mat(cover_dct_path)
        stego_decompress_path = path.replace("cover", baseline_folder)
        stego_additive = load_mat(stego_decompress_path)
        con_additive = get_confidence(model, con, sess, cover_decompress, stego_additive)
        # stego_nonadditive = np.zeros([256, 256])
        # stego_nonadditive = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # stego_nonadditive = stego_nonadditive.astype(np.double)
        stego_dct = np.zeros([256,256])
        best_coef = np.zeros([256, 256])
        best_coef[:][:] = C_COEFFS
        stego_dct[:][:] = C_COEFFS
        
        mct_path = "./data/mcts_juniward/" + jpeg_name
        shutil.copy(cover_jpeg_path, mct_path)
        stego_nonadditive = np.zeros([256, 256])
        stego_nonadditive[:][:] = cover_decompress
        for sublattice_idx in range(4):
            best = -9999
            print("idx: " + str(sublattice_idx))
            part_coef = get_idx_portion(C_COEFFS, sublattice_idx, idx_list)
            print("processing img:", path)
            root = Node()
            # stego = substitude_sublattice(sublattice_idx, cover, stego)
            # COST_MAP = HILL(stego_nonadditive)
            COST_MAP, nzAC = JUNIWARD(mct_path, 0.2)
            print("nzAC: "+str(nzAC))
            # part_cover = cover

            part_cost = get_idx_portion(COST_MAP, sublattice_idx, idx_list)
            tmp_cost = part_cost.reshape([-1])
            sorted_idx = tmp_cost.argsort()
            set_idx(sorted_idx)
            # modi = cover - stego_cmd
            # pad_modi = np.zeros([260,260])
            # pad_modi[2:258, 2:258] = modi
            # pad_cost = np.zeros([260,260])
            # pad_cost[2:258, 2:258] = COST_MAP

            SUM_SUC = 0

            # root.gen_psb()
            j = 0
            if sublattice_idx > 0:
                while j < 128:
                    print("iter: " + str(j))
                    #state = np.random.randint(-1, 2, [128, 128])
                    set_state(sublattice_size)
                    j += 1
                    leaf = root.UCTSearch()
                    # tmp = modi == -1
                    # sum_1 = sum(sum(tmp))
                    # print(sum_1)
                    # tmp = modi == 1
                    # sum_1 = sum(sum(tmp))
                    # print(sum_1)
                    state = get_state()
                    coef_mcts = embed_j(part_coef, state, part_cost, alpha, 0.2)
                    # diff = stego_nonadditive - cover
                    # print(sum(sum(diff == 1)))
                    # print(sum(sum(diff == -1)))
                    stego_dct[np.ix_(idx_list[sublattice_idx/2], idx_list[sublattice_idx%2])] = coef_mcts
                    stego_nonadditive = dct2spacial(stego_dct, dct_sess, res, DCT)
                    con_nonadditive = get_confidence(model, con, sess, cover_decompress, stego_nonadditive)
                    if con_nonadditive > 0.98:
                        best_coef[np.ix_(idx_list[sublattice_idx/2], idx_list[sublattice_idx%2])] = coef_mcts
                        break
                    diff = (con_nonadditive - con_additive)
                    if diff > 0:
                        diff *= 1000
                    else:
                        diff *= 100
                    print("Diff: " + str(diff))
                    leaf.backpropagation(diff)
                    if diff > best:
                        best = diff
                        best_coef[np.ix_(idx_list[sublattice_idx/2], idx_list[sublattice_idx%2])] = coef_mcts
                # stego_nonadditive = best_stego
                write_jpeg(best_coef, mct_path)
                stego_dct = best_coef[:,:]
            else:
                state = np.zeros([128,128])
                coef_mcts = embed_j(part_coef, state, part_cost, alpha, 0.2)
                best_coef[np.ix_(idx_list[sublattice_idx/2], idx_list[sublattice_idx%2])] = coef_mcts
                stego_dct[np.ix_(idx_list[sublattice_idx/2], idx_list[sublattice_idx%2])] = coef_mcts
                write_jpeg(best_coef, mct_path)
                stego_nonadditive = dct2spacial(best_coef, dct_sess, res, DCT)

        # diff = best_stego - cover
        # print(sum(sum(diff == 1)))
        # print(sum(sum(diff == 0)))
        # data = []
        # label = []
        # cost = []
        # for row in range(2,258):
        # for lin in range(2,258):
        # nei = np.zeros([5,5,2])
        # nei[:,:,0] = get_neighbor(row*2+2+row_idx, lin*2+2+lin_idx, nei_size)
        # nei[:,:,1] = get_cost_neighbor(row*2+2+row_idx, lin*2+2+lin_idx, nei_size)
        # data.append(nei)

        # label.append(int(best_modi[row][lin]+1))
        # cv2.imwrite("./stego/"+name.split('/')[-1], )
        # save(img_name, data, label)

    eng.quit()



