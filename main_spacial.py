# -*- coding:utf-8  -*-
from libs.utils import *
from libs.MCTS import *
global MAX_DEPTH
global seed
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default="./data/cover_spatial/", required=False, type=str, help='Path of dataset')
    parser.add_argument('--model_path', '-p', required=True, type=str, help='absolute_model_path')
    parser.add_argument('--sublattice_size', '-s', default=128, type=int, help='size of sublattice')
    parser.add_argument('--alpha', '-a', default=1.5, type=float, help='alpha')
    args = parser.parse_args()
    data_path = args.dataset
    
    baseline_folder = "hill"
    model_path = args.model_path
    print("Loading model: "+model_path)
    sublattice_size = args.sublattice_size
    MAX_DEPTH = sublattice_size * sublattice_size
    alpha = args.alpha
    print("Alpha: "+str(alpha))
    model, con, sess = build_srnet(model_path)
    print("Loading SrNet successfully!")
    file_list = glob.glob(data_path + '*.pgm')
    for path in file_list:

        img_name = path.split('/')[-1].split('.')[0]
        seed = int(img_name)
        cover = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cover = cover.astype(np.double)
        baseline_path = path.replace("cover_spatial", baseline_folder)
        stego_additive = cv2.imread(baseline_path, cv2.IMREAD_GRAYSCALE)
        stego_additive = stego_additive.astype(np.double)
        con_additive = get_confidence(model, con, sess, cover, stego_additive)
        stego_nonadditive = np.zeros([256, 256])
        stego_nonadditive = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        stego_nonadditive = stego_nonadditive.astype(np.double)
        best_stego = np.zeros([256, 256])
        #initial_state = np.random.randint(-1, 1, [sublattice_size, sublattice_size])
        for sublattice_idx in range(0, 4, 1):
            best = -9999
            row_idx = int(int(sublattice_idx) / 2)
            lin_idx = int(int(sublattice_idx) % 2)
            #state[:][:] = initial_state
            print("processing img: ", path)
            root = Node()
            # stego = substitude_sublattice(sublattice_idx, cover, stego)
            #COST_MAP = SUNIWARD(stego_nonadditive, 0.2)
            COST_MAP = HILL(stego_nonadditive)
            part_cover = cover[row_idx::2, lin_idx::2]
            part_cost = COST_MAP[row_idx::2, lin_idx::2]
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
                    #state = np.random.randint(-1, 2, [sublattice_size, sublattice_size])
                    set_state(sublattice_size)
                    print("idx:" + str(sublattice_idx) + "  iter: " + str(j))
                    j += 1
                    leaf = root.UCTSearch()
                    # tmp = modi == -1
                    # sum_1 = sum(sum(tmp))
                    # print(sum_1)
                    # tmp = modi == 1
                    # sum_1 = sum(sum(tmp))
                    # print(sum_1)
                    state = get_state()
                    stego_mct = embed(part_cover, state, part_cost, seed, alpha, 0.2)
                    stego_nonadditive[row_idx::2, lin_idx::2] = stego_mct.astype(np.double)
                    # diff = stego_nonadditive - cover
                    # print(sum(sum(diff == 1)))
                    # print(sum(sum(diff == -1)))
                    con_nonadditive = get_confidence(model, con, sess, cover, stego_nonadditive)
                    if con_nonadditive > 0.98:
                        best_stego = stego_nonadditive
                        break
                    diff = (con_nonadditive - con_additive)
                    if diff > 0:
                        diff *= 1000
                    else:
                        diff *= 100
                    print("Diff: " + str(diff))
                    leaf.backpropagation(diff)
                    if diff > best:
                        best_stego = stego_nonadditive
                        best = diff
                stego_nonadditive = best_stego
            else:
                modi = np.zeros([sublattice_size, sublattice_size])
                stego_mct = embed(part_cover, modi, part_cost, seed, alpha, 0.2)
                stego_nonadditive[row_idx::2, lin_idx::2] = stego_mct.astype(np.double)

        # diff = best_stego - cover
        # print(sum(sum(diff == 1)))
        # print(sum(sum(diff == 0)))
        best_stego = best_stego.astype(np.uint8)
        cv2.imwrite("./data/mcts_hill/" + path.split('/')[-1], best_stego)
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


